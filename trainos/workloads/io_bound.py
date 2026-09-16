"""I/O-bound demo workload.

Repeatedly writes a chunk to a temp file, flushes it to disk, and reads it
back, with only trivial CPU work in between. This produces the `io_bound`
signature: low/medium CPU, lots of I/O activity while CPU is low (the
io_wait proxy in features.py), and bursty throughput.

TrainOS should classify this as `io_bound` and leave it at default fairness
(nice 0, SCHED_OTHER) since it spends its time waiting on I/O anyway.

    python -m trainos.workloads.io_bound
    python -m trainos.workloads.io_bound --duration 0   # until killed

We fsync each write so the I/O actually hits the device and shows up in
per-process io_counters; without fsync the OS page cache would hide it.
"""

from __future__ import annotations

import os
import tempfile
import time

from ._common import (
    banner,
    deadline_from,
    expired,
    install_sigterm_handler,
    parse_args,
    set_process_name,
)

# 1 MiB payload per iteration — enough to register as real I/O without
# filling the disk (we truncate/reuse the same file each cycle).
_CHUNK = b"x" * (1024 * 1024)


def main() -> int:
    args = parse_args()
    install_sigterm_handler()
    set_process_name("trainos-io-bound")
    banner("io_bound", args.duration)

    deadline = deadline_from(args.duration)
    fd, path = tempfile.mkstemp(prefix="trainos_io_", suffix=".tmp")
    os.close(fd)
    try:
        while not expired(deadline):
            # write + flush to disk
            with open(path, "wb") as fh:
                fh.write(_CHUNK)
                fh.flush()
                os.fsync(fh.fileno())
            # read it back
            with open(path, "rb") as fh:
                _ = fh.read()
            # short pause: I/O-bound work isn't burning CPU, it's waiting.
            time.sleep(0.05)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
