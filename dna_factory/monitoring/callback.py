"""Transformers callback that publishes training state to the background recorder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from dna_factory.monitoring.recorder import TrainingMonitorRecorder

logger = logging.getLogger(__name__)


class TrainingMonitorCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: str | Path,
        trainer_type: str,
        model_name: str | None,
        gpu_interval_seconds: float = 1.0,
        queue_size: int = 10_000,
    ):
        self.recorder = TrainingMonitorRecorder(
            output_dir=output_dir,
            trainer_type=trainer_type,
            model_name=model_name,
            gpu_interval_seconds=gpu_interval_seconds,
            queue_size=queue_size,
        )

    @property
    def db_path(self) -> Path:
        return self.recorder.db_path

    @staticmethod
    def _is_primary(state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    @staticmethod
    def _state_payload(state) -> dict[str, Any]:
        return {
            "global_step": int(getattr(state, "global_step", 0)),
            "max_steps": int(getattr(state, "max_steps", 0)),
            "epoch": getattr(state, "epoch", None),
            "best_metric": getattr(state, "best_metric", None),
            "best_checkpoint": getattr(state, "best_model_checkpoint", None),
        }

    def on_train_begin(self, args, state, control, **kwargs):
        if self._is_primary(state):
            try:
                self.recorder.start(self._state_payload(state))
            except Exception as exc:  # noqa: BLE001 - monitoring must fail open.
                logger.warning("Could not start training monitor: %s", exc)

    def on_step_end(self, args, state, control, **kwargs):
        if self._is_primary(state):
            self.recorder.emit("progress", self._state_payload(state))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._is_primary(state):
            self.recorder.emit(
                "metrics",
                {
                    "step": int(getattr(state, "global_step", 0)),
                    "epoch": getattr(state, "epoch", None),
                    "split": "eval"
                    if any(str(key).startswith("eval_") for key in (logs or {}))
                    else "train",
                    "metrics": dict(logs or {}),
                },
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._is_primary(state):
            self.recorder.emit(
                "event",
                {
                    "step": int(getattr(state, "global_step", 0)),
                    "kind": "evaluation",
                    "payload": dict(metrics or {}),
                },
            )

    def on_save(self, args, state, control, **kwargs):
        if self._is_primary(state):
            self.recorder.emit(
                "event",
                {
                    "step": int(getattr(state, "global_step", 0)),
                    "kind": "checkpoint",
                    "payload": {
                        "best_checkpoint": getattr(state, "best_model_checkpoint", None)
                    },
                },
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self._is_primary(state):
            self.recorder.emit("progress", self._state_payload(state))
            self.close("completed")

    def close(self, status: str = "completed", error: str | None = None) -> None:
        self.recorder.close(status=status, error=error)
