"""Launch the interactive DNA Factory training monitor."""

from __future__ import annotations

import argparse
from pathlib import Path

from dna_factory.monitoring.storage import (
    MONITOR_DB_NAME,
    MONITOR_DIR_NAME,
    monitor_db_path,
)


def discover_runs(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()

    def activity_time(path: Path) -> float:
        wal_path = path.with_name(f"{path.name}-wal")
        return max(
            path.stat().st_mtime, wal_path.stat().st_mtime if wal_path.exists() else 0
        )

    return sorted(
        root.glob(f"**/{MONITOR_DIR_NAME}/{MONITOR_DB_NAME}"),
        key=activity_time,
        reverse=True,
    )


def resolve_db(target: str | None, root: str | Path) -> Path:
    if target:
        path = Path(target).expanduser().resolve()
        if path.is_file():
            return path
        candidate = monitor_db_path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No monitor database under: {path}")
    runs = discover_runs(root)
    if not runs:
        raise FileNotFoundError(
            f"No {MONITOR_DIR_NAME}/{MONITOR_DB_NAME} found below {Path(root).resolve()}"
        )
    return runs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach to a live DNA Factory training run."
    )
    parser.add_argument(
        "output_dir", nargs="?", help="Training output directory or monitor.db"
    )
    parser.add_argument(
        "--root", default="outputs", help="Run search root when output_dir is omitted"
    )
    parser.add_argument(
        "--list", action="store_true", help="List discovered monitor databases and exit"
    )
    parser.add_argument(
        "--refresh", type=float, default=1.0, help="TUI refresh interval in seconds"
    )
    args = parser.parse_args(argv)

    if args.list:
        for path in discover_runs(args.root):
            print(path)
        return 0

    db_path = resolve_db(args.output_dir, args.root)
    try:
        from dna_factory.tui.app import TrainingMonitorApp
    except ImportError as exc:
        raise SystemExit(
            "Training monitor UI dependencies are missing. Run `uv sync` and retry."
        ) from exc
    TrainingMonitorApp(db_path, refresh_seconds=args.refresh).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
