"""TrainOS: Workload-Aware CPU Scheduling Layer for ML Training Jobs.

A user-space adaptive scheduler that classifies process behavior with a
Random Forest model and applies Linux scheduling controls accordingly.
"""

__version__ = "1.0.0"

# Scheduling-relevant workload classes. These map to concrete scheduler
# actions, unlike the "ML vs non-ML" label which does not.
WORKLOAD_CLASSES = ("cpu_bound", "io_bound", "interactive")

# Feature order MUST be stable across training and inference.
FEATURE_NAMES = (
    "cpu_usage",           # X1: mean CPU utilisation (%)
    "cpu_variance",        # X2: variance of CPU utilisation over the window
    "memory_usage",        # X3: memory utilisation (%)
    "io_wait",             # X4: fraction of time spent waiting on I/O
    "runtime_duration",    # X5: seconds the process has been alive
    "throughput_stability" # X6: 1.0 = perfectly steady, 0.0 = bursty
)
