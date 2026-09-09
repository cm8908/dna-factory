"""SQLite persistence shared by the training recorder and the read-only TUI."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

MONITOR_DIR_NAME = ".dna-monitor"
MONITOR_DB_NAME = "monitor.db"


def monitor_db_path(output_dir: str | Path) -> Path:
    """Return the conventional monitor database path for a training output."""
    return Path(output_dir).expanduser().resolve() / MONITOR_DIR_NAME / MONITOR_DB_NAME


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    output_dir TEXT NOT NULL,
    trainer_type TEXT NOT NULL,
    model_name TEXT,
    hostname TEXT NOT NULL,
    pid INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    ended_at REAL,
    global_step INTEGER NOT NULL DEFAULT 0,
    max_steps INTEGER NOT NULL DEFAULT 0,
    epoch REAL,
    step_seconds_ewma REAL,
    best_metric REAL,
    best_checkpoint TEXT,
    dropped_events INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    wall_time REAL NOT NULL,
    step INTEGER NOT NULL,
    epoch REAL,
    split TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_name_step
    ON metrics(run_id, name, step);

CREATE TABLE IF NOT EXISTS gpu_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    wall_time REAL NOT NULL,
    hostname TEXT NOT NULL,
    gpu_uuid TEXT NOT NULL,
    physical_index INTEGER NOT NULL,
    cuda_index INTEGER,
    name TEXT,
    utilization REAL,
    memory_used INTEGER,
    memory_total INTEGER,
    temperature REAL,
    power_watts REAL
);

CREATE INDEX IF NOT EXISTS idx_gpu_run_time
    ON gpu_samples(run_id, wall_time);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    wall_time REAL NOT NULL,
    step INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_time
    ON events(run_id, wall_time);
"""


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class MonitoringStore:
    """Single-writer store. Create and use it from the recorder thread only."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=5.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def start_run(self, metadata: dict[str, Any]) -> None:
        now = float(metadata.get("started_at", time.time()))
        self.connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, output_dir, trainer_type, model_name, hostname, pid,
                status, started_at, updated_at, global_step, max_steps, epoch
            ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)
            """,
            (
                metadata["run_id"],
                metadata["output_dir"],
                metadata["trainer_type"],
                metadata.get("model_name"),
                metadata["hostname"],
                metadata["pid"],
                now,
                now,
                metadata.get("global_step", 0),
                metadata.get("max_steps", 0),
                metadata.get("epoch"),
            ),
        )
        self.connection.commit()

    def update_progress(self, run_id: str, payload: dict[str, Any]) -> None:
        self.connection.execute(
            """
            UPDATE runs SET updated_at=?, global_step=?, max_steps=?, epoch=?,
                step_seconds_ewma=?, best_metric=?, best_checkpoint=?, dropped_events=?
            WHERE run_id=?
            """,
            (
                payload["wall_time"],
                payload.get("global_step", 0),
                payload.get("max_steps", 0),
                payload.get("epoch"),
                payload.get("step_seconds_ewma"),
                payload.get("best_metric"),
                payload.get("best_checkpoint"),
                payload.get("dropped_events", 0),
                run_id,
            ),
        )
        self.connection.commit()

    def heartbeat(self, run_id: str, dropped_events: int) -> None:
        self.connection.execute(
            "UPDATE runs SET updated_at=?, dropped_events=? WHERE run_id=?",
            (time.time(), dropped_events, run_id),
        )
        self.connection.commit()

    def add_metrics(self, run_id: str, payload: dict[str, Any]) -> None:
        split = payload.get("split", "train")
        rows = []
        for name, value in payload.get("metrics", {}).items():
            number = _numeric(value)
            if number is not None:
                rows.append(
                    (
                        run_id,
                        payload["wall_time"],
                        payload.get("step", 0),
                        payload.get("epoch"),
                        split,
                        str(name),
                        number,
                    )
                )
        if rows:
            self.connection.executemany(
                """
                INSERT INTO metrics (run_id, wall_time, step, epoch, split, name, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.connection.commit()

    def add_gpu_samples(
        self,
        run_id: str,
        hostname: str,
        wall_time: float,
        samples: list[dict[str, Any]],
    ) -> None:
        rows = [
            (
                run_id,
                wall_time,
                hostname,
                sample["gpu_uuid"],
                sample["physical_index"],
                sample.get("cuda_index"),
                sample.get("name"),
                sample.get("utilization"),
                sample.get("memory_used"),
                sample.get("memory_total"),
                sample.get("temperature"),
                sample.get("power_watts"),
            )
            for sample in samples
        ]
        if rows:
            self.connection.executemany(
                """
                INSERT INTO gpu_samples (
                    run_id, wall_time, hostname, gpu_uuid, physical_index, cuda_index,
                    name, utilization, memory_used, memory_total, temperature, power_watts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.connection.commit()

    def add_event(self, run_id: str, payload: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO events (run_id, wall_time, step, kind, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                payload["wall_time"],
                payload.get("step", 0),
                payload["kind"],
                json.dumps(payload.get("payload", {}), default=str, ensure_ascii=False),
            ),
        )
        self.connection.commit()

    def finish_run(
        self, run_id: str, status: str, error: str | None, dropped_events: int
    ) -> None:
        now = time.time()
        self.connection.execute(
            """
            UPDATE runs SET status=?, updated_at=?, ended_at=?, error=?, dropped_events=?
            WHERE run_id=?
            """,
            (status, now, now, error, dropped_events, run_id),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class MonitorReader:
    """Small read-only query facade for the TUI."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Monitor database not found: {self.path}")
        uri = f"file:{self.path}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        self.connection.row_factory = sqlite3.Row

    def latest_run(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else {}

    def metric_names(self, run_id: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT name FROM metrics WHERE run_id=? ORDER BY name", (run_id,)
        ).fetchall()
        return [row[0] for row in rows]

    def latest_metrics(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT m.name, m.value, m.step, m.split
            FROM metrics m
            JOIN (
                SELECT name, MAX(id) AS id FROM metrics WHERE run_id=? GROUP BY name
            ) latest ON latest.id=m.id
            ORDER BY m.name
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def metric_series(
        self, run_id: str, name: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT step, wall_time, value, split FROM (
                SELECT step, wall_time, value, split, id FROM metrics
                WHERE run_id=? AND name=? ORDER BY id DESC LIMIT ?
            ) ORDER BY id
            """,
            (run_id, name, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def latest_gpu_samples(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT g.* FROM gpu_samples g
            JOIN (
                SELECT gpu_uuid, MAX(id) AS id FROM gpu_samples
                WHERE run_id=? GROUP BY gpu_uuid
            ) latest ON latest.id=g.id
            ORDER BY COALESCE(g.cuda_index, g.physical_index)
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_events(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT wall_time, step, kind, payload_json FROM events
            WHERE run_id=? ORDER BY id DESC LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def close(self) -> None:
        self.connection.close()
