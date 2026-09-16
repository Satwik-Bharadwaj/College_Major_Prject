"""Launcher for the demo workloads.

Convenience entry point so you can start/stop all three demo workloads with
one command instead of backgrounding each by hand.

    python -m trainos.workloads start            # start all three (120s each)
    python -m trainos.workloads start --duration 0   # run until stopped
    python -m trainos.workloads stop             # stop any it started
    python -m trainos.workloads list             # show what's running

`start` launches each workload as its own background process (so TrainOS sees
three distinct PIDs) and records their PIDs in a small state file. `stop`
reads that file and terminates them.

Prefer running the workloads BEFORE starting TrainOS, then:

    sudo ./venv/bin/python -m trainos.cli run --dry-run
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile

_WORKLOADS = ("cpu_bound", "io_bound", "interactive")
_STATE_FILE = os.path.join(tempfile.gettempdir(), "trainos_workloads.pids")


def _python() -> str:
    return sys.executable or "python3"


def _start(duration: float) -> int:
    pids = []
    for name in _WORKLOADS:
        proc = subprocess.Popen(
            [_python(), "-m", f"trainos.workloads.{name}", "--duration", str(duration)]
        )
        pids.append((name, proc.pid))
        print(f"started {name:12s} pid={proc.pid}")

    with open(_STATE_FILE, "w") as fh:
        for name, pid in pids:
            fh.write(f"{pid} {name}\n")

    print(f"\n{len(pids)} workloads running. State: {_STATE_FILE}")
    print("Now start TrainOS in another terminal:")
    print("    sudo ./venv/bin/python -m trainos.cli run --dry-run")
    print("Stop the workloads with:  python -m trainos.workloads stop")
    return 0


def _read_state():
    if not os.path.exists(_STATE_FILE):
        return []
    entries = []
    with open(_STATE_FILE) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                entries.append((int(parts[0]), parts[1]))
    return entries


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop() -> int:
    entries = _read_state()
    if not entries:
        print("No recorded workloads to stop.")
        return 0
    stopped = 0
    for pid, name in entries:
        if not _alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"stopped {name:12s} pid={pid}")
            stopped += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"no permission to stop pid={pid} ({name})", file=sys.stderr)
    try:
        os.remove(_STATE_FILE)
    except OSError:
        pass
    print(f"stopped {stopped} workload(s).")
    return 0


def _list() -> int:
    entries = _read_state()
    if not entries:
        print("No recorded workloads.")
        return 0
    for pid, name in entries:
        state = "running" if _alive(pid) else "exited"
        print(f"{name:12s} pid={pid:<8} {state}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trainos.workloads")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("start", help="start all demo workloads in background")
    s.add_argument("--duration", type=float, default=120.0,
                   help="seconds each workload runs (0 = until stopped)")

    sub.add_parser("stop", help="stop workloads started by `start`")
    sub.add_parser("list", help="list workloads and their state")

    args = p.parse_args(argv)
    if args.command == "start":
        return _start(args.duration)
    if args.command == "stop":
        return _stop()
    if args.command == "list":
        return _list()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
