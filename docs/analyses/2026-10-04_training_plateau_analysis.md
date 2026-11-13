# MicroDuck training plateau check

Scope: iterations 0-17679 of W&B run qktrrkg5. Read through the public SDK with an authorised account; no training was started.

Both groups are 500-iteration averages, 2000-2499 against 17000-17499, to reduce single-point noise.

| Metric | 2000-2499 | 17000-17499 | Change |
|---|---:|---:|---:|
| Mean return | 113.0012 | 124.8662 | +10.50% |
| Mean episode length (control steps) | 955.8387 | 973.9114 | +1.89% |
| Planar velocity error, logged value | 0.4265 | 0.4218 | -1.08% |
| Yaw rate error, logged value | 1.1185 | 0.9862 | -11.84% |
| Absolute second difference of the action | 0.3461 | 0.3208 | -7.31% |
| Mean fall-termination count (not a fall rate) | 0.3958 | 0.2358 | -40.43% |

Main observations: the planar velocity error is close to a plateau; yaw tracking, smoothness, return and the termination count are still improving slowly. The curriculum parameters reached their final stage by about iteration 2000, so the total return before and after that is not a direct comparison at the same difficulty.

Limits of the metrics: mean_action_acc is the second difference of the action, not an accuracy; 1000 control steps equal the 20-second episode limit. The velocity error is a cumulative value that upstream normalises by the maximum command time, and does not equal the mean physical error per step. fell_over is an averaged count from the log, not a fall rate from an independent evaluation.

Data check: the first paged export had 17 exact duplicate records and 17 missing iteration numbers; the original file was kept, the missing iterations were read again and a cleaned version saved separately, giving 17680 unique consecutive iterations and 32 columns with nothing missing.

Suggestion: first compare deterministic fixed-command replays of model_2000.pt, model_5000.pt and model_17500.pt, with the same seed, the same initial state and the same command sequence, and count straight / turn tracking, stopping and 60 seconds without falling. Choose the checkpoint on validation scenes and keep the formal test scenes until the configuration is frozen. Nothing in the curves calls for running on to 50000 iterations.

This curve analysis cannot replace MuJoCo replay or closed-loop acceptance on the RK3588 board.
