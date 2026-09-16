# TrainOS Usage Guide

## 0. What you're shipping

A Python package `trainos` with a one-line installer. On the target Ubuntu box
you run `./install.sh` once, then start the scheduler as a background service.
No kernel changes, no reboot.

## 1. Install (Ubuntu or macOS)

```bash
cd College_Major_Prject
./install.sh
```

This creates a `venv/`, installs dependencies, installs the package, and trains
the model on first run. It needs internet the first time (to fetch pip
packages). After that everything is local.

If your Ubuntu is offline, on a machine WITH internet run:
```bash
pip download -r requirements.txt -d wheels/
```
copy the project (with `wheels/`) across, then:
```bash
python3 -m venv venv && source venv/bin/activate
pip install --no-index --find-links wheels -r requirements.txt
pip install -e .
python -m trainos.cli train
```

## 2. Verify it works (do this before your demo)

```bash
# a) dependency-free logic check (works even without a full install)
python3 tests/test_stdlib.py

# b) full end-to-end check (needs install)
./venv/bin/python -m trainos.cli selftest

# c) see the CFS-vs-TrainOS numbers
./venv/bin/python -m trainos.cli benchmark

# d) environment + model status
./venv/bin/python -m trainos.cli info
```

`selftest` exercises: dataset -> train+select -> save/load -> live metric
collection -> one scheduler tick. If it prints `ALL CHECKS PASSED`, the whole
pipeline works on that machine.

## 3. Train / retrain the model

```bash
# synthetic (reproducible, no data collection needed)
./venv/bin/python -m trainos.cli train --save-dataset dataset.csv

# from a real trace you collected (see step 5)
./venv/bin/python -m trainos.cli train --csv labelled_trace.csv
```

Output shows the model comparison table (RandomForest vs GradientBoosting vs
DecisionTree vs KNN), the chosen model, and feature importances with a leakage
warning if any single feature dominates.

## 4. Run the scheduler

Always start with `--dry-run` — it classifies and logs but changes nothing:

```bash
sudo ./venv/bin/python -m trainos.cli run --dry-run
```

When you're happy, run for real (root needed to renice other users' processes):

```bash
sudo ./venv/bin/python -m trainos.cli run --interval 3
# add --affinity to also pin CPU-bound jobs off core 0
# add --ticks 20 to auto-stop after 20 cycles (handy for a timed demo)
```

On macOS the actions are simulated (logged, not applied) so you can develop and
demo the loop without Linux.

## 4b. Demo workloads (show TrainOS scheduling live)

The package ships three demo workloads that each produce a distinct,
scheduler-relevant behaviour so you can watch TrainOS observe, classify, and
act on them. Run them first, then start TrainOS.

  - `cpu_bound`    tight compute loop  -> high steady CPU, no I/O
  - `io_bound`     fsync'd read/write  -> lots of I/O, low CPU
  - `interactive`  burst-then-sleep    -> bursty (high-variance) CPU

Start all three in the background, then run TrainOS in another terminal:

```bash
# terminal 1: start the workloads (each runs 120s; use --duration 0 for no limit)
./venv/bin/python -m trainos.workloads start

# terminal 2: watch TrainOS classify and (with sudo, on Linux) schedule them
sudo ./venv/bin/python -m trainos.cli run --dry-run --interval 2
# then for real:
sudo ./venv/bin/python -m trainos.cli run --interval 2

# when done:
./venv/bin/python -m trainos.workloads stop
```

You can also launch a single workload by hand:

```bash
./venv/bin/python -m trainos.workloads.cpu_bound &
./venv/bin/python -m trainos.workloads.io_bound --duration 0 &
./venv/bin/python -m trainos.workloads.interactive &
```

**Important — run the demo on Linux, not macOS.** macOS does not expose
per-process I/O counters, so the `io_bound` workload's I/O is invisible there
and it gets misclassified as `interactive`. On Linux, `io_counters` is
available and all three workloads classify correctly. The `cpu_bound` workload
classifies correctly on both.

To confirm expected classes during a demo, look at the per-tick log line
(`classes={...}`) and the session summary's `class breakdown` when you stop.

## 5. Collect real training data on Linux (optional, for stronger results)

```bash
# run some workloads (a training script, a file copy, a text editor), then:
./venv/bin/python -m trainos.cli collect --ticks 30 --interval 2 --out trace.csv
```

Add a `label` column to `trace.csv` (cpu_bound / io_bound / interactive), then
`train --csv trace.csv`. This turns the synthetic-trained demo into a
data-backed result for your report.

## 6. Run as a permanent background daemon (systemd)

```bash
sudo mkdir -p /opt/trainos
sudo cp -r . /opt/trainos/
sudo cp deploy/trainos.service /etc/systemd/system/trainos.service
sudo systemctl daemon-reload
sudo systemctl enable --now trainos
journalctl -u trainos -f          # watch it work
sudo systemctl stop trainos       # stop
```

## Safety notes

- The enforcer never touches PID 0/1 or TrainOS itself.
- `nice` changes are always reversible (they reset when a process exits; you can
  also just stop TrainOS).
- Switching a process to `SCHED_BATCH`/`SCHED_IDLE` is reversible and low risk.
- Raising priority (negative nice) and real-time policies need root; without it
  TrainOS still lowers priority of batch jobs, which needs no privilege.
