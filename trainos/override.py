"""User overrides for workload classification.

The classifier infers a workload's class from behaviour, but early in a job's
life (before its signature is clear) or for jobs the user simply knows are
important, the user can *force* a class. An override always wins over the
classifier.

Overrides are stored in a small JSON file so they work across processes: the
user tags a job in one terminal, and the separately-launched scheduler daemon
picks it up on its next tick. Two kinds of override are supported:

  by PID    : force a specific process id -> class
  by name   : force any process whose name/cmdline contains a substring

Typical use:

    # mark a training job as heavy before/after launching it
    trainos flag --pid 12345 --class heavy_compute
    trainos flag --name train.py --class heavy_compute
    trainos flag --list
    trainos flag --clear-pid 12345
    trainos flag --clear-all

The scheduler consults this on every tick, so flags take effect live.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Optional

from . import WORKLOAD_CLASSES

# Shared location so the CLI and a root-run scheduler agree. Overridable via
# env var for tests / non-default deployments.
OVERRIDE_FILE = os.environ.get(
    "TRAINOS_OVERRIDE_FILE",
    os.path.join(tempfile.gettempdir(), "trainos_overrides.json"),
)


def _empty() -> Dict:
    return {"by_pid": {}, "by_name": {}}


def load() -> Dict:
    """Load the override store, tolerating a missing or corrupt file."""
    if not os.path.exists(OVERRIDE_FILE):
        return _empty()
    try:
        with open(OVERRIDE_FILE) as fh:
            data = json.load(fh)
        # normalise shape
        data.setdefault("by_pid", {})
        data.setdefault("by_name", {})
        return data
    except (json.JSONDecodeError, OSError):
        return _empty()


def _save(data: Dict) -> None:
    # atomic-ish write so a concurrent reader never sees a half file
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(OVERRIDE_FILE) or ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, OVERRIDE_FILE)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _validate_class(workload_class: str) -> None:
    if workload_class not in WORKLOAD_CLASSES:
        raise ValueError(
            f"unknown class {workload_class!r}; expected one of {WORKLOAD_CLASSES}"
        )


def set_pid(pid: int, workload_class: str) -> None:
    _validate_class(workload_class)
    data = load()
    data["by_pid"][str(pid)] = workload_class
    _save(data)


def set_name(pattern: str, workload_class: str) -> None:
    _validate_class(workload_class)
    data = load()
    data["by_name"][pattern] = workload_class
    _save(data)


def clear_pid(pid: int) -> bool:
    data = load()
    removed = data["by_pid"].pop(str(pid), None) is not None
    _save(data)
    return removed


def clear_name(pattern: str) -> bool:
    data = load()
    removed = data["by_name"].pop(pattern, None) is not None
    _save(data)
    return removed


def clear_all() -> None:
    _save(_empty())


def lookup(pid: int, name: str = "") -> Optional[str]:
    """Return the forced class for this process, or None if not overridden.

    PID overrides take precedence over name-pattern overrides.
    """
    data = load()
    by_pid = data.get("by_pid", {})
    hit = by_pid.get(str(pid))
    if hit:
        return hit
    if name:
        for pattern, cls in data.get("by_name", {}).items():
            if pattern in name:
                return cls
    return None
