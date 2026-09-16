"""Shared helpers for the demo workloads.

Keeps each workload script tiny and consistent: a standard --duration flag,
a PID/name banner so you can find it in the TrainOS log, best-effort process
renaming, and clean Ctrl-C handling.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def set_process_name(name: str) -> None:
    """Best-effort: make the process show up under a friendly name.

    Uses setproctitle if installed (optional). Falls back silently otherwise;
    TrainOS keys off behaviour, not the name, so this is only for readability.
    """
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle(name)
    except Exception:
        pass


def parse_args(default_duration: float = 120.0) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--duration",
        type=float,
        default=default_duration,
        help="seconds to run before exiting (0 = run until killed)",
    )
    return p.parse_args()


def banner(kind: str, duration: float) -> None:
    pid = os.getpid()
    how_long = "until killed" if duration == 0 else f"{duration:.0f}s"
    print(
        f"[workload:{kind}] pid={pid} running for {how_long}. "
        f"Stop with: kill {pid}",
        flush=True,
    )


def install_sigterm_handler() -> None:
    """Exit cleanly on SIGTERM/SIGINT so `kill <pid>` stops us tidily."""

    def _stop(*_):
        print("[workload] stopping", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)


def deadline_from(duration: float) -> float:
    """Return an absolute stop time, or 0.0 meaning 'never'."""
    return 0.0 if duration == 0 else time.time() + duration


def expired(deadline: float) -> bool:
    return deadline != 0.0 and time.time() >= deadline
