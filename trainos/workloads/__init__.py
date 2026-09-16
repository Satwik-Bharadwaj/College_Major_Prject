"""Demo workloads for TrainOS.

These are small, self-contained programs that each produce a *distinct*,
scheduler-relevant runtime behaviour so TrainOS can observe them, classify
them, and act on them during a live demo:

  cpu_bound     : a tight compute loop -> high, steady CPU, no I/O
  io_bound      : repeated file read/write -> lots of I/O, low CPU
  interactive   : short bursts of work then sleep -> bursty (high variance) CPU

Run one in the background, then start TrainOS and watch it schedule them:

    python -m trainos.workloads.cpu_bound &
    python -m trainos.workloads.io_bound &
    python -m trainos.workloads.interactive &
    sudo ./venv/bin/python -m trainos.cli run --dry-run

Each script sets its own process name (via setproctitle if available) and
prints its PID on startup so you can point to it in the TrainOS log.

Stop them with `python -m trainos.workloads stop` or just `kill <pid>`.
"""

from __future__ import annotations

__all__ = ["cpu_bound", "io_bound", "interactive"]
