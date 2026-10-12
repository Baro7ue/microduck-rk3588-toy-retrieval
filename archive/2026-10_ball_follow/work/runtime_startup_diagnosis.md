# Mac runtime startup diagnosis

The user's original trial failed at 1.078 s, while every observed policy label was
`homing`. The two-second home ramp had not finished, so neither the standing nor
the self-trained walking network drove those recorded ticks. The control-loop
rate remained approximately 50 Hz and the missed counter stayed at 22 throughout
the trial. Evidence is in
`task/evidence/G2/runtime_stand_20261004_183934_420167/`.

The physical body releases its fixed initial pose on torque enable, while the
daemon performs a two-second unbalanced pose ramp. An isolated reproduction with
the same pinned daemon, parameters, model files and XML physics fell at 1.063 s.

The separate `work/runtime_bootstrap_body.py` entry keeps the initial HOME pose
fixed only during startup. It subscribes to the native daemon's state, and ends
support permanently when `stand` or `walk` first reports active inference. A
five-second startup timeout stops the simulated body if takeover cannot be
observed. After release, physical motion is unrestricted. The active-state label
does not by itself prove successful target delivery; any bus write failures must
also be inspected in the daemon log.

The actual helper was verified in an isolated headless comparison:
`task/evidence/G2/startup_diagnostic_20261004_185108_848699/`.
The supported interval kept projected gravity exactly `[0, 0, -1]`. After support
ended, the pretrained `velstand.onnx` held the body without falling for 10.023 s.
One read timeout and reconnect occurred during active standing, with four
additional work overruns. This is not a fault-free timing result.

The pinned repositories remain clean. No model, dependency, training parameter,
daemon source, TCP protocol, actuator physics or heading compensation was changed.
This support is an explicit simulation initialization aid, not a successful
unmodified startup. Its duration must be excluded from balancing scores. It is
not a formal G2 result or a real-hardware startup validation. The self-trained
walking policy and GUI timing still require separate verification.

For the next user step, `work/restart_runtime_body_assisted.zsh` stops only the
verified existing local daemon and body and opens the new held body. It leaves
control disabled. The existing daemon startup and bounded probe are subsequent
steps. All original failure evidence remains available.
