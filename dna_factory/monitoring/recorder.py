"""Background recorder that keeps telemetry work off the training thread."""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dna_factory.monitoring.gpu import NvidiaGpuSampler
from dna_factory.monitoring.storage import MonitoringStore, monitor_db_path

logger = logging.getLogger(__name__)


class TrainingMonitorRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        trainer_type: str,
        model_name: str | None = None,
        gpu_interval_seconds: float = 1.0,
        queue_size: int = 10_000,
        gpu_sampler_factory: Callable[[], Any] = NvidiaGpuSampler,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.db_path = monitor_db_path(self.output_dir)
        self.run_id = uuid.uuid4().hex
        self.trainer_type = trainer_type
        self.model_name = model_name
        self.gpu_interval_seconds = max(0.2, float(gpu_interval_seconds))
        self.events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(
            maxsize=max(1, int(queue_size))
        )
        self.dropped_events = 0
        self.error: str | None = None
        self._gpu_sampler_factory = gpu_sampler_factory
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._ready = threading.Event()
        self._final_status = "completed"
        self._final_error: str | None = None
        self._last_step: int | None = None
        self._last_step_time: float | None = None
        self._step_seconds_ewma: float | None = None

    def start(self, initial_state: dict[str, Any] | None = None) -> None:
        if self._thread is not None:
            return
        state = initial_state or {}
        self._thread = threading.Thread(
            target=self._run,
            args=(state,),
            name="dna-training-monitor",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        if self._stop_requested.is_set() or self.error is not None:
            return
        item = (kind, {**(payload or {}), "wall_time": time.time()})
        try:
            self.events.put_nowait(item)
        except queue.Full:
            self.dropped_events += 1

    def close(self, status: str = "completed", error: str | None = None) -> None:
        if self._thread is None or self._stop_requested.is_set():
            return
        self._final_status = status
        self._final_error = error
        self._stop_requested.set()
        self._thread.join(timeout=10.0)

    def _add_step_timing(self, payload: dict[str, Any]) -> dict[str, Any]:
        step = int(payload.get("global_step", 0))
        wall_time = float(payload["wall_time"])
        if (
            self._last_step is not None
            and step > self._last_step
            and self._last_step_time is not None
        ):
            sample = (wall_time - self._last_step_time) / (step - self._last_step)
            self._step_seconds_ewma = (
                sample
                if self._step_seconds_ewma is None
                else 0.2 * sample + 0.8 * self._step_seconds_ewma
            )
        if self._last_step is None or step > self._last_step:
            self._last_step = step
            self._last_step_time = wall_time
        payload["step_seconds_ewma"] = self._step_seconds_ewma
        payload["dropped_events"] = self.dropped_events
        return payload

    def _run(self, initial_state: dict[str, Any]) -> None:
        store = None
        sampler = None
        hostname = socket.gethostname()
        try:
            store = MonitoringStore(self.db_path)
            store.start_run(
                {
                    "run_id": self.run_id,
                    "output_dir": str(self.output_dir),
                    "trainer_type": self.trainer_type,
                    "model_name": self.model_name,
                    "hostname": hostname,
                    "pid": os.getpid(),
                    "started_at": time.time(),
                    **initial_state,
                }
            )
            sampler = self._gpu_sampler_factory()
            if getattr(sampler, "error", None):
                store.add_event(
                    self.run_id,
                    {
                        "wall_time": time.time(),
                        "step": 0,
                        "kind": "gpu_unavailable",
                        "payload": {"error": sampler.error},
                    },
                )
            next_gpu_sample = time.monotonic()
            self._ready.set()

            while not self._stop_requested.is_set() or not self.events.empty():
                timeout = max(0.0, min(0.5, next_gpu_sample - time.monotonic()))
                try:
                    kind, payload = self.events.get(timeout=timeout)
                except queue.Empty:
                    kind = ""
                    payload = {}

                if kind == "progress":
                    store.update_progress(self.run_id, self._add_step_timing(payload))
                elif kind == "metrics":
                    store.add_metrics(self.run_id, payload)
                elif kind == "event":
                    store.add_event(self.run_id, payload)

                now = time.monotonic()
                if now >= next_gpu_sample:
                    samples = sampler.sample()
                    store.add_gpu_samples(self.run_id, hostname, time.time(), samples)
                    store.heartbeat(self.run_id, self.dropped_events)
                    next_gpu_sample = now + self.gpu_interval_seconds

            store.finish_run(
                self.run_id, self._final_status, self._final_error, self.dropped_events
            )
        except Exception as exc:  # noqa: BLE001 - monitoring must fail open.
            self.error = str(exc)
            logger.warning("Training monitor disabled after recorder error: %s", exc)
        finally:
            self._ready.set()
            if sampler is not None:
                sampler.close()
            if store is not None:
                store.close()
