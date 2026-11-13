# MicroDuck heading hold when walking straight (2026-10-04)

## Change

The original CPU replay is wrapped separately, without changing the microduck_rl repository, its dependencies, the ONNX or the training. The 20 ms action delay is kept.

When walking straight, an outer controller locks the current relative heading. It estimates heading from the simulated gyroscope and the gravity direction the original policy already has, computes a small turning correction and writes it into the turn-command slot of the 61-dimensional observation. The user's command and the automatic correction are stored separately. The controller reads no world position or world heading; that truth is only evaluated offline in the diagnostic program.

P feedback was verified first, then a small integral term was added to reduce the constant offset. Current parameters: Kp=1.0, Ki=0.05, correction limit +-0.3 rad/s, integral limit +-0.8 rad·s. The library's default config is still Ki=0; the launcher explicitly selects the Ki=0.05 verified here. The policy's 14-dimensional action is not filtered or modified.

## Comparison under the same conditions

Same initial pose: a 0.1 m/s command for 5 seconds, then a 0.3 m/s command forward for 60 seconds, with the user's turn command zero throughout. Policy: 17,500 iterations; BAM M6; kp=200; vin=7.4 V; sag=0.1; 20 ms control step, 5 ms physics step. One deterministic CPU simulation per condition; not a test on the board.

| Metric | 20 ms delay only | Plus heading hold |
|---|---:|---:|
| Absolute heading error at the end | 12.09° | 0.18° |
| Absolute sideways offset at the end | 1.830 m | 0.090 m |
| Forward position at the end | 10.38 m | 10.30 m |
| Largest body tilt | 3.78° | 3.84° |
| Reached the 70° fall threshold | no | no |

These are end values, not the largest error over the run; there is still sway within the gait cycle. Heading hold clearly improves this case but does not guarantee zero position error. Randomised scenes and behaviour on the board must be verified separately later with the same configuration.

## Acceptance of the control states

A 60-second sequence "straight -> left turn -> stop -> straight -> right turn -> stop -> straight" was run: the correction drops out during manual turning and stopping and locks the direction again when straight walking restarts; both turns went the right way, the mean forward speed in the stop segments was below 0.002 m/s, and there was no fall. 6 unit checks verify the feedback sign, the tilt coordinate transform, manual priority, pausing and re-locking in reverse, the limits, and config / sensor faults.

After a pause with T, resuming locks the current direction again; it does not try to make up for rotation that went unobserved during the pause. A/E are continuous commands; to go back to straight walking, stop with the space bar first, then press ↑.

## Retest

Run in a terminal:

```bash
zsh "/Users/ethan/Documents/RK3588 Microduck/work/run_straight_hold.zsh"
```

Wait 5 seconds, then press ↑ in the terminal and watch for 60 seconds; space stops, Q quits. Logs and CSV are saved with a timestamp under task/evidence/G2/. Look at the "[航向保持]" (heading hold) field in the terminal: with the user's turn at zero, the turning correction actually sent changes with the error.

## Limits

An ideal simulated gyroscope is used for now; over a long time a real gyroscope's bias would still make the relative heading drift. Heading control alone cannot remove sideways slip either; holding position strictly needs reliable odometry or an external observation. This wrapper is for local verification and should not be connected to visual following without testing: the outer visual turning and the heading hold must be given clearly separate control authority.

The formal baseline of the original policy should keep its record with heading hold off; the added controller should be noted separately in the model and experiment lists.
