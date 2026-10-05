# Recorded computational costs

These are historical measurements, not model predictions. Regenerated training outputs can be deleted and recomputed, whereas the exact measured durations and sampled memory peaks cannot.

- `geometry_timing.json`: 18 controlled S1/T1 F/G/U runs with common uniform refresh. Includes geometry/point construction, initialization, block preparation, optimization, checkpoint output, inference and memory observations.
- `geometry_hardware.json`: machine and thread settings for those measurements.
- `c1_reference_timing.json`: 32 C1 runs with reference evaluations in the measurement workflow. Solver and observation scopes are recorded separately.
- `c1_plain_gap_timing.json`: 24 C1 runs measuring the solver without reference-field access. These timings have a different scope from the preceding dataset.
- `benchmark_hardware.json`: hardware description for the controlled C1 comparison.
- `model_cost_summary.csv` and `time_to_accuracy_per_run.csv`: recorded baseline resource and first-observed accuracy-crossing data. These use the original observation schedule, not the new launcher's output overhead.

Use each dataset's explicit timing scope. Do not pool solver-only, observed end-to-end and geometry-pipeline times. CUDA allocated/reserved bytes describe the PyTorch allocator; sampled process RSS is not a guaranteed instantaneous peak. An accuracy crossing between observations is interval-censored, and a missed threshold is not a measured crossing time.
