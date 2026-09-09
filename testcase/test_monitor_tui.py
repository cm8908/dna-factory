"""Smoke-test the Textual dashboard against a real monitor database."""

from __future__ import annotations

import asyncio
import time

from textual.widgets import DataTable, ProgressBar, SelectionList, TabbedContent

import monitor as monitor_cli
from dna_factory.monitoring.storage import MonitoringStore, monitor_db_path
from dna_factory.tui.app import TrainingMonitorApp


def test_tui_loads_progress_metrics_and_gpu(tmp_path, monkeypatch):
    db_path = monitor_db_path(tmp_path)
    store = MonitoringStore(db_path)
    now = time.time()
    store.start_run(
        {
            "run_id": "ui-run",
            "output_dir": str(tmp_path),
            "trainer_type": "GRPO",
            "model_name": "test/model",
            "hostname": "test",
            "pid": 42,
            "started_at": now - 10,
            "global_step": 0,
            "max_steps": 100,
        }
    )
    store.update_progress(
        "ui-run",
        {
            "wall_time": now,
            "global_step": 25,
            "max_steps": 100,
            "epoch": 0.25,
            "step_seconds_ewma": 1.0,
            "best_metric": None,
            "best_checkpoint": None,
            "dropped_events": 0,
        },
    )
    store.add_metrics(
        "ui-run",
        {
            "wall_time": now,
            "step": 25,
            "epoch": 0.25,
            "split": "train",
            "metrics": {"loss": 1.5, "reward": 0.7},
        },
    )
    store.add_gpu_samples(
        "ui-run",
        "test",
        now,
        [
            {
                "gpu_uuid": "GPU-1",
                "physical_index": 0,
                "cuda_index": 0,
                "name": "Fake GPU",
                "utilization": 90,
                "memory_used": 8 * 2**30,
                "memory_total": 16 * 2**30,
                "temperature": 60,
                "power_watts": 200,
            }
        ],
    )
    store.close()

    async def run_test():
        app = TrainingMonitorApp(db_path, refresh_seconds=60)
        assert callable(app.run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#progress", ProgressBar).progress == 25
            assert app.query_one("#metric-current", DataTable).row_count == 2
            assert set(app.query_one("#metric-picker", SelectionList).selected) == {
                "loss",
                "reward",
            }
            assert app.query_one("#gpu-table", DataTable).row_count == 1
            await pilot.click("#--content-tab-curves")
            assert app.query_one(TabbedContent).active == "curves"
            await pilot.click("#metric-picker", offset=(2, 1))
            assert len(app.query_one("#metric-picker", SelectionList).selected) == 1

    asyncio.run(run_test())

    launched = []

    def fake_run(app):
        launched.append(app.run_id)
        app.reader.close()

    monkeypatch.setattr(TrainingMonitorApp, "run", fake_run)
    assert monitor_cli.main([str(tmp_path)]) == 0
    assert launched == ["ui-run"]
