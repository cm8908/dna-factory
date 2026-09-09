# Live training monitor

DNA Factory records progress, Trainer metrics, evaluation/checkpoint events, and
local NVIDIA GPU telemetry for every training run by default. The recorder runs in
a background thread and drops telemetry when its queue is full, so a slow or closed
monitor does not block training.

## Open the interactive TUI

Start SFT, DPO, GRPO, or distillation normally. The training log prints the exact
attach command after the output directory has been resolved:

```bash
python monitor.py outputs/your-run
```

When the output directory is unknown, attach to the most recently updated run below
`outputs/`:

```bash
python monitor.py
```

Use the mouse or keyboard to switch between Overview, Curves, GPU, and Events.
Select metric names in the Curves pane to change the chart. Press `r` to refresh
immediately and `q` to quit. Closing the TUI does not stop training.

To list all discovered monitor databases without opening the TUI:

```bash
python monitor.py --list
```

## Configuration

The following Dnotitia arguments can be set in YAML or on the command line:

```yaml
monitor_enabled: true
monitor_gpu_interval_seconds: 1.0
monitor_queue_size: 10000
```

Set `monitor_enabled: false` to disable all monitor recording. GPU telemetry is
optional: metrics and progress remain available when NVML or an NVIDIA GPU is not
present.

Monitor data is stored at:

```text
<output_dir>/.dna-monitor/monitor.db
```

The database uses SQLite WAL mode and should live on local storage. Do not put the
active monitor database on NFS or another network filesystem. Multi-node GPU
aggregation is not currently provided; the rank-zero host records the GPUs visible
to that process.
