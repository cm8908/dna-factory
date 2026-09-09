"""Optional NVIDIA GPU sampling backed by NVML."""

from __future__ import annotations

import os
from typing import Any


class NvidiaGpuSampler:
    """Samples visible NVIDIA GPUs; quietly disables itself when NVML is unavailable."""

    def __init__(self):
        self._pynvml = None
        self._devices: list[tuple[int, int | None, Any]] = []
        self.error: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._devices = self._resolve_devices(pynvml)
        except Exception as exc:  # noqa: BLE001 - GPU telemetry is optional.
            self.error = str(exc)

    @staticmethod
    def _decode(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    def _resolve_devices(self, pynvml) -> list[tuple[int, int | None, Any]]:
        count = pynvml.nvmlDeviceGetCount()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or not visible.strip():
            physical_indices = list(range(count))
        else:
            physical_indices = []
            tokens = [token.strip() for token in visible.split(",") if token.strip()]
            for token in tokens:
                if token.isdigit() and int(token) < count:
                    physical_indices.append(int(token))
                    continue
                for index in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    uuid = self._decode(pynvml.nvmlDeviceGetUUID(handle))
                    if uuid == token or uuid.startswith(token):
                        physical_indices.append(index)
                        break
        return [
            (
                physical_index,
                cuda_index,
                pynvml.nvmlDeviceGetHandleByIndex(physical_index),
            )
            for cuda_index, physical_index in enumerate(physical_indices)
        ]

    @staticmethod
    def _optional(call, default=None):
        try:
            return call()
        except Exception:  # noqa: BLE001 - NVML feature support varies by GPU.
            return default

    def sample(self) -> list[dict[str, Any]]:
        if self._pynvml is None:
            return []
        pynvml = self._pynvml
        samples = []
        for physical_index, cuda_index, handle in self._devices:
            memory = self._optional(
                lambda handle=handle: pynvml.nvmlDeviceGetMemoryInfo(handle)
            )
            utilization = self._optional(
                lambda handle=handle: pynvml.nvmlDeviceGetUtilizationRates(handle)
            )
            power_mw = self._optional(
                lambda handle=handle: pynvml.nvmlDeviceGetPowerUsage(handle)
            )
            samples.append(
                {
                    "gpu_uuid": self._decode(pynvml.nvmlDeviceGetUUID(handle)),
                    "physical_index": physical_index,
                    "cuda_index": cuda_index,
                    "name": self._decode(pynvml.nvmlDeviceGetName(handle)),
                    "utilization": getattr(utilization, "gpu", None),
                    "memory_used": getattr(memory, "used", None),
                    "memory_total": getattr(memory, "total", None),
                    "temperature": self._optional(
                        lambda handle=handle: pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "power_watts": power_mw / 1000.0 if power_mw is not None else None,
                }
            )
        return samples

    def close(self) -> None:
        if self._pynvml is not None:
            self._optional(self._pynvml.nvmlShutdown)
