"""Attachable Textual dashboard for a DNA Factory monitor database."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)
from textual_plotext import PlotextPlot

from dna_factory.monitoring.storage import MonitorReader

PREFERRED_METRICS = (
    "loss",
    "eval_loss",
    "reward",
    "reward_std",
    "kl",
    "rewards/accuracies",
    "rewards/margins",
    "learning_rate",
)


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _number(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


class TrainingMonitorApp(App):
    TITLE = "DNA Factory Training Monitor"
    SUB_TITLE = "read-only live view"
    BINDINGS: ClassVar = [("q", "quit", "Quit"), ("r", "refresh_now", "Refresh")]
    CSS = """
    Screen { background: #0b1020; }
    #summary { height: 5; padding: 0 1; background: #121a2f; }
    #status-line { height: 2; content-align: left middle; }
    #progress { height: 3; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1; }
    #overview-grid { height: 1fr; }
    #metric-current { width: 1fr; height: 1fr; }
    #run-details { width: 1fr; height: 1fr; padding: 1; border: round #345; }
    #curve-layout { height: 1fr; }
    #metric-picker { width: 34; height: 1fr; border: round #345; }
    #metric-plot { width: 1fr; height: 1fr; border: round #345; }
    DataTable { height: 1fr; }
    .empty-note { padding: 1; color: #94a3b8; }
    """

    def __init__(self, db_path: str | Path, refresh_seconds: float = 1.0):
        super().__init__()
        self.reader = MonitorReader(db_path)
        self.refresh_seconds = max(0.25, refresh_seconds)
        self.run_state = self.reader.latest_run()
        if not self.run_state:
            self.reader.close()
            raise RuntimeError(f"No training run found in {db_path}")
        self.run_id = self.run_state["run_id"]
        self.known_metrics = self.reader.metric_names(self.run_id)

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="summary"):
            yield Static("Waiting for training state...", id="status-line")
            yield ProgressBar(total=100, show_eta=False, id="progress")
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"), Horizontal(id="overview-grid"):
                yield DataTable(id="metric-current", cursor_type="row")
                yield Static("", id="run-details")
            with TabPane("Curves", id="curves"), Horizontal(id="curve-layout"):
                yield SelectionList(*self._metric_options(), id="metric-picker")
                yield PlotextPlot(id="metric-plot")
            with TabPane("GPU", id="gpu"):
                yield DataTable(id="gpu-table", cursor_type="row")
                yield Label(
                    "NVML data appears here when NVIDIA GPUs are available.",
                    classes="empty-note",
                )
            with TabPane("Events", id="events"):
                yield DataTable(id="event-table", cursor_type="row")
        yield Footer()

    def _metric_options(self):
        defaults = {name for name in PREFERRED_METRICS if name in self.known_metrics}
        if not defaults and self.known_metrics:
            defaults.add(self.known_metrics[0])
        return [(name, name, name in defaults) for name in self.known_metrics]

    def on_mount(self) -> None:
        metrics = self.query_one("#metric-current", DataTable)
        metrics.add_columns("Metric", "Value", "Step", "Split")
        gpu = self.query_one("#gpu-table", DataTable)
        gpu.add_columns("GPU", "Name", "Util", "VRAM", "Temp", "Power", "Sampled")
        events = self.query_one("#event-table", DataTable)
        events.add_columns("Time", "Step", "Event", "Details")
        self.refresh_data()
        self.set_interval(self.refresh_seconds, self.refresh_data)

    def action_refresh_now(self) -> None:
        self.refresh_data()

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        self._update_plot()

    def refresh_data(self) -> None:
        try:
            self.run_state = self.reader.latest_run()
            self._update_summary()
            self._update_metrics()
            self._update_gpu()
            self._update_events()
            self._update_plot()
        except Exception as exc:  # noqa: BLE001 - keep the TUI alive during partial writes.
            self.query_one("#status-line", Static).update(
                f"[red]Monitor read error:[/] {exc}"
            )

    def _update_summary(self) -> None:
        step = int(self.run_state.get("global_step") or 0)
        max_steps = int(self.run_state.get("max_steps") or 0)
        status = str(self.run_state.get("status", "unknown")).upper()
        stale = (
            self.run_state.get("status") == "running"
            and time.time() - self.run_state["updated_at"] > 15
        )
        if stale:
            status = "STALE"
        elapsed = (self.run_state.get("ended_at") or time.time()) - self.run_state[
            "started_at"
        ]
        step_seconds = self.run_state.get("step_seconds_ewma")
        eta = (
            step_seconds * max(0, max_steps - step)
            if step_seconds and max_steps
            else None
        )
        epoch = self.run_state.get("epoch")
        line = (
            f"[bold cyan]{self.run_state['trainer_type']}[/]  [bold]{status}[/]  "
            f"step [bold]{step:,}[/] / {max_steps:,}  epoch {_number(epoch)}  "
            f"elapsed {_duration(elapsed)}  ETA {_duration(eta)}  "
            f"dropped {self.run_state.get('dropped_events', 0)}"
        )
        self.query_one("#status-line", Static).update(line)
        progress = self.query_one("#progress", ProgressBar)
        progress.update(total=max(max_steps, 1), progress=min(step, max(max_steps, 1)))
        details = (
            f"[bold]Run[/]\n"
            f"model: {self.run_state.get('model_name') or '-'}\n"
            f"host: {self.run_state.get('hostname')}  pid: {self.run_state.get('pid')}\n"
            f"output: {self.run_state.get('output_dir')}\n"
            f"best metric: {_number(self.run_state.get('best_metric'))}\n"
            f"best checkpoint: {self.run_state.get('best_checkpoint') or '-'}"
        )
        if self.run_state.get("error"):
            details += f"\n[red]error: {self.run_state['error']}[/]"
        self.query_one("#run-details", Static).update(details)

    def _sync_metric_options(self, names: list[str]) -> None:
        picker = self.query_one("#metric-picker", SelectionList)
        for name in names:
            if name not in self.known_metrics:
                picker.add_option((name, name, False))
                self.known_metrics.append(name)

    def _update_metrics(self) -> None:
        rows = self.reader.latest_metrics(self.run_id)
        self._sync_metric_options([row["name"] for row in rows])
        table = self.query_one("#metric-current", DataTable)
        table.clear()
        for row in rows:
            table.add_row(
                row["name"], _number(row["value"]), str(row["step"]), row["split"]
            )

    def _update_plot(self) -> None:
        picker = self.query_one("#metric-picker", SelectionList)
        selected = list(picker.selected)
        plot = self.query_one("#metric-plot", PlotextPlot)
        plot.plt.clear_data()
        for name in selected[:8]:
            series = self.reader.metric_series(self.run_id, name, limit=1000)
            if series:
                plot.plt.plot(
                    [row["step"] for row in series],
                    [row["value"] for row in series],
                    label=name,
                )
        plot.plt.title("Training metrics (last 1,000 points)")
        plot.plt.xlabel("global step")
        plot.plt.theme("dark")
        plot.refresh()

    def _update_gpu(self) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        for row in self.reader.latest_gpu_samples(self.run_id):
            used = row["memory_used"] or 0
            total = row["memory_total"] or 0
            vram = f"{used / 2**30:.1f}/{total / 2**30:.1f} GiB" if total else "-"
            sampled = (
                datetime.fromtimestamp(row["wall_time"], UTC)
                .astimezone()
                .strftime("%H:%M:%S")
            )
            table.add_row(
                str(
                    row["cuda_index"]
                    if row["cuda_index"] is not None
                    else row["physical_index"]
                ),
                row["name"] or "-",
                f"{_number(row['utilization'], 3)}%",
                vram,
                f"{_number(row['temperature'], 3)}°C",
                f"{_number(row['power_watts'], 4)} W",
                sampled,
            )

    def _update_events(self) -> None:
        table = self.query_one("#event-table", DataTable)
        table.clear()
        for row in self.reader.recent_events(self.run_id):
            when = (
                datetime.fromtimestamp(row["wall_time"], UTC)
                .astimezone()
                .strftime("%H:%M:%S")
            )
            try:
                details = json.dumps(
                    json.loads(row["payload_json"]), ensure_ascii=False
                )
            except json.JSONDecodeError:
                details = row["payload_json"]
            table.add_row(when, str(row["step"]), row["kind"], details[:240])

    def on_unmount(self) -> None:
        self.reader.close()
