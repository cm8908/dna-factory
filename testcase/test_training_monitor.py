"""Tests for the non-blocking live training monitor."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from dna_factory.monitoring.callback import TrainingMonitorCallback
from dna_factory.monitoring.recorder import TrainingMonitorRecorder
from dna_factory.monitoring.storage import (
    MonitoringStore,
    MonitorReader,
    monitor_db_path,
)

RUN_ID = "test-run"


def _metadata(output_dir: Path) -> dict:
    return {
        "run_id": RUN_ID,
        "output_dir": str(output_dir),
        "trainer_type": "GRPO",
        "model_name": "test/model",
        "hostname": "test-host",
        "pid": 123,
        "started_at": time.time(),
        "global_step": 0,
        "max_steps": 100,
    }


class FakeGpuSampler:
    error = None

    def sample(self):
        return [
            {
                "gpu_uuid": "GPU-test",
                "physical_index": 0,
                "cuda_index": 0,
                "name": "Fake GPU",
                "utilization": 91,
                "memory_used": 8 * 2**30,
                "memory_total": 16 * 2**30,
                "temperature": 65,
                "power_watts": 250.0,
            }
        ]

    def close(self):
        pass


def test_store_round_trip(tmp_path):
    db_path = monitor_db_path(tmp_path)
    store = MonitoringStore(db_path)
    store.start_run(_metadata(tmp_path))
    now = time.time()
    store.update_progress(
        RUN_ID,
        {
            "wall_time": now,
            "global_step": 10,
            "max_steps": 100,
            "epoch": 0.5,
            "step_seconds_ewma": 2.0,
            "best_metric": 0.7,
            "best_checkpoint": "checkpoint-10",
            "dropped_events": 0,
        },
    )
    store.add_metrics(
        RUN_ID,
        {
            "wall_time": now,
            "step": 10,
            "epoch": 0.5,
            "split": "train",
            "metrics": {"loss": 1.25, "ignored": "text", "not_finite": float("nan")},
        },
    )
    store.add_gpu_samples(RUN_ID, "test-host", now, FakeGpuSampler().sample())
    store.heartbeat(RUN_ID, 2)
    store.add_event(
        RUN_ID,
        {
            "wall_time": now,
            "step": 10,
            "kind": "checkpoint",
            "payload": {"path": "x"},
        },
    )
    store.finish_run(RUN_ID, "completed", None, 2)
    store.close()

    reader = MonitorReader(db_path)
    run = reader.latest_run()
    assert run["status"] == "completed"
    assert run["global_step"] == 10
    assert reader.metric_names(RUN_ID) == ["loss"]
    assert reader.metric_series(RUN_ID, "loss")[0]["value"] == 1.25
    assert reader.latest_gpu_samples(RUN_ID)[0]["utilization"] == 91
    assert reader.latest_run()["dropped_events"] == 2
    assert reader.recent_events(RUN_ID)[0]["kind"] == "checkpoint"
    reader.close()


def test_recorder_flushes_events_and_finishes(tmp_path):
    recorder = TrainingMonitorRecorder(
        tmp_path,
        "SFT",
        "test/model",
        gpu_interval_seconds=0.2,
        gpu_sampler_factory=FakeGpuSampler,
    )
    recorder.start({"global_step": 0, "max_steps": 2})
    recorder.emit("progress", {"global_step": 1, "max_steps": 2, "epoch": 0.5})
    recorder.emit(
        "metrics",
        {
            "step": 1,
            "epoch": 0.5,
            "split": "train",
            "metrics": {"loss": 2.0},
        },
    )
    recorder.close("completed")

    reader = MonitorReader(recorder.db_path)
    assert reader.latest_run()["status"] == "completed"
    assert reader.latest_run()["global_step"] == 1
    assert reader.latest_metrics(recorder.run_id)[0]["name"] == "loss"
    assert reader.latest_gpu_samples(recorder.run_id)[0]["gpu_uuid"] == "GPU-test"
    reader.close()


class FakeRecorder:
    def __init__(self):
        self.calls = []
        self.db_path = Path("monitor.db")

    def start(self, payload):
        self.calls.append(("start", payload))

    def emit(self, kind, payload):
        self.calls.append((kind, payload))

    def close(self, status="completed", error=None):
        self.calls.append(("close", {"status": status, "error": error}))


def test_callback_records_only_world_process_zero(tmp_path):
    callback = TrainingMonitorCallback(tmp_path, "DPO", "test/model")
    callback.recorder = FakeRecorder()
    state = SimpleNamespace(
        is_world_process_zero=True,
        global_step=4,
        max_steps=10,
        epoch=0.4,
        best_metric=None,
        best_model_checkpoint=None,
    )
    callback.on_train_begin(None, state, None)
    callback.on_step_end(None, state, None)
    callback.on_log(None, state, None, logs={"loss": 1.0})
    callback.on_save(None, state, None)
    callback.on_train_end(None, state, None)
    assert [kind for kind, _ in callback.recorder.calls] == [
        "start",
        "progress",
        "metrics",
        "event",
        "progress",
        "close",
    ]

    callback.recorder.calls.clear()
    state.is_world_process_zero = False
    callback.on_train_begin(None, state, None)
    callback.on_log(None, state, None, logs={"loss": 1.0})
    callback.on_train_end(None, state, None)
    assert callback.recorder.calls == []
