"""TrainOS: Workload-Aware CPU Scheduling Layer for ML Training Jobs.

A user-space adaptive scheduler that classifies process behavior with a
Random Forest model and applies Linux scheduling controls accordingly.
"""

__version__ = "1.0.0"

# Scheduling-relevant workload classes.
#
# TrainOS's goal is to let a demanding job (an ML training run, or any
# sustained heavy-compute workload) run well on a machine that was not built
# for it. So the classifier separates the *demanding* workload from ordinary
# background work, and the policy protects the former by getting the latter
# out of its way.
#
#   heavy_compute : sustained high-CPU + high-memory, long-running, steady.
#                   ML training is the target case; the honest claim is
#                   "sustained heavy compute", not "detects ML specifically".
#   normal        : everything else -- bursty, low-CPU, short-lived, or
#                   I/O-waiting background work.
WORKLOAD_CLASSES = ("heavy_compute", "normal")

# The class TrainOS protects (boosts). A user override can force a process
# into this class regardless of what the classifier infers.
PRIORITY_CLASS = "heavy_compute"

# Feature order MUST be stable across training and inference.
FEATURE_NAMES = (
    "cpu_usage",           # X1: mean CPU utilisation (%)
    "cpu_variance",        # X2: variance of CPU utilisation over the window
    "memory_usage",        # X3: memory utilisation (%)
    "io_wait",             # X4: fraction of time spent waiting on I/O
    "runtime_duration",    # X5: seconds the process has been alive
    "throughput_stability" # X6: 1.0 = perfectly steady, 0.0 = bursty
)
