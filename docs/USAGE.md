# TrainOS Usage Guide

## 0. What you're shipping

A Python package `trainos` with a one-line installer. On the target Ubuntu box
you run `./install.sh` once, then start the scheduler as a background service.
No kernel changes, no reboot.

**Goal.** Let a demanding job -- an ML training run, or any sustained
heavy-compute workload -- run well on a machine that was not designed for it.
TrainOS classifies each process as `heavy_compute` (the demanding job we
protect) or `normal` (ordinary background work), then boosts the heavy job and
de-prioritises the background churn so the heavy job gets the CPU.

**Two ways a job becomes `heavy_compute`:**
1. The classifier infers it from behaviour (sustained high CPU + memory, steady).
2. You flag it explicitly with `trainos flag` -- useful when the classifier is
   still unsure (e.g. right after launch). An explicit flag always wins.

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
# add --affinity to also confine background jobs off core 0 (heavy job gets it)
# add --ticks 20 to auto-stop after 20 cycles (handy for a timed demo)
```

On macOS the actions are simulated (logged, not applied) so you can develop and
demo the loop without Linux.

## 4a. Flag a job as heavy_compute (user override)

If you want a specific job protected regardless of what the classifier infers,
flag it. The override wins over the classifier and takes effect on the running
scheduler's next tick.

```bash
# by PID (e.g. a training job you just launched)
./venv/bin/python -m trainos.cli flag --pid 12345 --class heavy_compute

# by name/cmdline substring (matches any process whose name contains it)
./venv/bin/python -m trainos.cli flag --name train.py --class heavy_compute

./venv/bin/python -m trainos.cli flag --list          # show active overrides
./venv/bin/python -m trainos.cli flag --clear-pid 12345
./venv/bin/python -m trainos.cli flag --clear-all
```

## 4b. Generate background load (many jobs at once)

Instead of a script per job, one generator spawns hundreds of mixed background
jobs (all `normal`) to create realistic contention:

```bash
# spawn 100 mixed background jobs, ~90s each
./venv/bin/python -m trainos.cli generate --count 100 --duration 90
./venv/bin/python -m trainos.cli generate --status     # how many alive
./venv/bin/python -m trainos.cli generate --stop        # stop them all
```

Each job is drawn from a cpu / io / interactive profile (`--cpu/--io/--interactive`
proportions), all of which are ordinary `normal` work that should yield to a
heavy job.

## 4c. Measure it: A/B evaluation (the real result)

`evaluate` runs the actual experiment behind the project's goal. It spawns
background load plus ONE tracked ML training job, runs it once under the default
OS scheduler and once under TrainOS (with the ML job flagged heavy_compute), and
reports the ML job's completion time and throughput for each:

```bash
# on the Ubuntu box, ideally under sudo so boosting takes full effect
sudo ./venv/bin/python -m trainos.cli evaluate --count 100 --rounds 20
```

Output is a table: rounds completed, wall seconds (lower is better), and
rounds/sec (higher is better) for baseline vs TrainOS, plus the percentage
reduction in the ML job's completion time. That is your measured, defensible
result -- a real before/after on real processes, not a simulation.

**Important — measure on Linux, ideally with root.** Only on Linux do `nice`,
`SCHED_BATCH`, and affinity actually change scheduling; off Linux the enforcer
simulates, so both phases look the same by design. Negative-nice boosting needs
root; without root TrainOS can still de-prioritise the background jobs (positive
nice), which is the larger lever, so you still see an effect. macOS also hides
per-process I/O counters, another reason to run the real evaluation on Ubuntu.

## 4d. Quick visual demo (optional)

To just watch classification live, the original per-behaviour workloads still
exist and can be launched together:

```bash
./venv/bin/python -m trainos.workloads start   # cpu + io + interactive
sudo ./venv/bin/python -m trainos.cli run --dry-run --interval 2
./venv/bin/python -m trainos.workloads stop
```

These are ordinary `normal` workloads under the new model; for the priority
side of the story use `flag` + a training job, or just run `evaluate`.

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
