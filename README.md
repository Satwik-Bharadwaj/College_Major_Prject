# TrainOS — Workload-Aware CPU Scheduling Layer

TrainOS is a **user-space adaptive CPU scheduling layer** for Linux. It observes
running processes, classifies each one's runtime behavior with a trained
**Random Forest** model, and applies real scheduling controls (priority, CPU
affinity, scheduling policy) on top of the Linux scheduler (CFS/EEVDF).

It is **not** a kernel patch. It runs as a normal background process/daemon and
uses standard Linux syscalls, so it ships to any Ubuntu machine and runs after a
one-line install.

```
┌──────────────┐   metrics   ┌───────────────┐  class   ┌────────────────┐
│  Processes   │ ──────────▶ │ Random Forest │ ───────▶ │ Policy Enforcer │
│ (via psutil) │             │  classifier   │          │ nice/affinity  │
└──────────────┘             └───────────────┘          └────────────────┘
        ▲                                                        │
        └────────────────── control loop (every N sec) ──────────┘
```

## What it does

1. **Collects** runtime metrics per process (CPU %, CPU variance, memory %,
   I/O wait, runtime, throughput stability) — the same features from the report.
2. **Classifies** each process into a scheduling-relevant class:
   `cpu_bound`, `io_bound`, or `interactive`.
3. **Acts** on that classification by adjusting `nice` value, CPU affinity, and
   scheduling policy — the part that makes it a scheduler, not just a classifier.

## Quick start (Ubuntu)

```bash
./install.sh                      # installs deps into a venv
sudo ./venv/bin/python -m trainos.cli run --dry-run   # observe only, no changes
sudo ./venv/bin/python -m trainos.cli run             # real scheduling
```

## Testing anywhere (macOS/Linux, no root)

```bash
./install.sh
./venv/bin/python -m trainos.cli selftest    # end-to-end check
./venv/bin/python -m trainos.cli train       # (re)train the model
./venv/bin/python -m trainos.cli benchmark   # simulated CFS vs TrainOS comparison
```

See `docs/USAGE.md` for the full guide and `docs/PHASE3.md` for how this maps to
your project report.
