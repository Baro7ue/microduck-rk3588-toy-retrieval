# MicroDuck circling to the left: diagnosis (2026-10-04)

User report: the 17,500-iteration policy walks upright; after pressing ↑ to raise the forward speed it circles to the left.

## Fixed conditions

- Repository commit: cb70b792312d559a4da09064d92009079671815f, left unmodified.
- Policy: walk_17500.onnx, checked to match model_17500.pt in weights and normalisation.
- Reuses upstream PolicyInference, BAM M6, motor kp=200, voltage 7.4 V, sag=0.1.
- Control step 20 ms, physics step 5 ms, 4 physics updates per control step.
- Main comparison: run at vx=0.1 for 5 seconds, then switch to [vx,vy,wz]=[0.3,0,0] for 30 seconds, imitating a press of ↑.
- No rendering, no keyboard intervention, the same initial state. Each item is one deterministic CPU simulation, not a randomised evaluation or a board result.

## Measured

| Configuration | Steady actual forward speed | Steady actual turn rate | Total heading change | Reached the 70° fall threshold |
|---|---:|---:|---:|---|
| Default, no action delay | 0.1859 m/s | +0.1871 rad/s | +315.63° | no |
| Only --delay 1 added | 0.1801 m/s | -0.0056 rad/s | -10.03° | no |

The steady averages start 2 seconds after the command switch (seconds 7-35 of the run); the heading change includes the start-up and the switch and comes from the unwrapped world heading of the body. A positive turn is to the left.

## Diagnosis

The training BAM actuator has delay_min_lag=3 and delay_max_lag=6, in physics steps: 15-30 ms. The default CPU replay has no action delay. The replay's existing --delay 1 means one control step, 20 ms, which is inside the training delay range, but it is a fixed control-step delay and not quite the same as the training's random per-actuator delay in physics steps.

In the single-variable comparison above, adding the delay clearly reduces the circling, which supports "a mismatch in actuator delay between training and replay is the main factor in this case". The policy tracks body-frame velocity and turn rate and has no lock on world heading; residual turning adds up, and it cannot be called tracking a perfectly straight line.

The default scene.xml and the training scene_walk.xml contain different collision models; tested at vx=0.1 and 0.3, the trajectories and metrics are exactly the same, so the scene difference does not explain the turning under the default condition.

Turn commands of +0.5/-0.5 rad/s were also tested with --delay 1: the mean actual turn rate is about +0.489/-0.564 rad/s, both runs lasted 30 seconds without reaching the 70° fall threshold. Within what was verified, turning left and right is kept.

## Next step

Use the original replay command with only --delay 1 added, still start at vx=0.1, then press ↑ to switch to 0.3 and watch for 30 seconds. Keep the start-up log and the per-second [vel 1s avg] output, and check that the turn command is 0 and that the actual turning is smaller.

For now, add no cloud training and no fixed counter-turn compensation. The replay still has a heading offset at start-up, and the actual forward speed is below the 0.3 command; later checkpoint selection and the closed loop on the board need separate verification.

Note: walk_5000.onnx was exported successfully on the Mac CPU with the official scripts/export.py of the locked version; the full export log is kept in this folder. This is a model export, not another training run, and this checkpoint has not been through this walking comparison.
