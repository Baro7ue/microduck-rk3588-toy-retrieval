# Stage 0 measurements: how to run them (2026-10-13)

Based on `claude-docs/stage0_plan_20261012/MicroDuck阶段0详细计划_20261012.pdf` (sections 2-5).
This step only adds files; not one line of the two upstream repositories or the existing scripts changed.

## What was added

| File | Purpose |
|---|---|
| `src/microduck_toy/sim/probe_body.py` | Sim body = `bootstrap_body.py` + truth recording + a test port on 127.0.0.1:7811 (push, lay down, reset, place objects) |
| `src/microduck_toy/stage0/link.py` | Connects to robotd (one state per control tick; resends the velocity and head commands at 20 Hz) and to the test port |
| `src/microduck_toy/stage0/common.py` | Shared parts: run folder, enable, reset, wait until still, fall check, `--rescore` |
| `src/microduck_toy/stage0/metrics.py` | Scoring (pure functions). The standing rule is copied from the lab's bench_getup: both feet on the ground, nothing else touching it, gravity z < -0.85, trunk height >= standing height - 4.5 cm; "fully standing" also needs the beak height >= 80% of the standing height |
| `src/microduck_toy/stage0/s0c_getup.py` | 0c getting up: c1 stand 10 s at each of four head poses; c2 pushed over while walking (sweep the push force first, then 3 pushes per direction); c3 lying on the back, front and side; after each get-up hold 5 s, then walk 3 s |
| `src/microduck_toy/stage0/s0a_walk.py` | 0a walking with the head down: a5 beak tip and camera position, a1v forward dead zone, a1w turning dead zone (cold/warm), a2 walk 1 m straight (heading PI), a3f/a3t pulses, a4 stopping |
| `src/microduck_toy/stage0/s0b_pick.py` | 0b GroundPick: 10 picks each at head H0 and H2; after a pick stand 3 s, then walk 0.5 m |
| `configs/robotd/stage0/{s1,s1u,s2,s2u}.toml.in` | s1 = alpha_stand + limp_fall; s2 = velstand; the ones with a u are the official tuning (0.9, low-pass 0.5/0.7, stand gain ratio 0.8), used only to find the cause when something fails |
| `scripts/sim/stage0/run_probe_body.zsh` | Starts the sim body above (for robotd on the Mac by default; `--board` for the board) |
| `scripts/sim/stage0/run_robotd.zsh` | Starts robotd on the Mac with the chosen config |
| `scripts/sim/stage0/run_test.zsh` | Runs 0a / 0b / 0c |
| `scripts/board/stage0/run_board_robotd.zsh` | Starts robotd on the board with the chosen config (same flow as `scripts/board/run_board_robotd.zsh`) |
| `tests/test_stage0_*.py` | 22 unit tests (scoring, sim body, config templates, launchers) |

In the cloud I built robotd from the same versions (microduck 43d0967, microduck_rl cb70b79) and ran
quick versions of 0c and 0a with s2 (walk_17500 + velstand). For 0b, velstand stood in for the pick
network, which checked only the flow. s1 and the real 0b wait for the two official models.

## Step 1: the two official models (P1)

The download commands and the commands that record the SHA-256 are in the chat reply (copying
commands out of a PDF or Markdown tends to bring wrong characters along).
After downloading, drag `alpha_ground_pick.onnx` and `alpha_stand.onnx` into the project chat, and I
will first run s1 and 0b in the cloud.

## Step 2: unit tests

```
scripts/tools/run_tests.zsh
```

You should see `OK`. `test_stage0_probe` needs microduck_rl's Python environment (that is `$MD_PY`);
without it the test is skipped.

## Step 3: three terminals for every measurement

`cd` to the repository root in each one first.

| Terminal | Command | Notes |
|---|---|---|
| A | `scripts/sim/stage0/run_probe_body.zsh` | The sim body. Add `--viewer` to watch it |
| B | `scripts/sim/stage0/run_robotd.zsh s2` | robotd; replace `s2` with the config under test |
| C | `scripts/sim/stage0/run_test.zsh 0c --config s2` | The test script; it sends `robot.enable` itself and stops when done |

When changing the config, quit A and B with Ctrl+C and start them again (after robotd changes
networks, the sim body's start-up support only works at the first enable).

Do the smoke test first (P4): `run_test.zsh 0c --config s2 --parts c1 --quick` stands 3 s at each of
the four head poses. It passes if the robot stays up and `truth.csv` has one row every 20 ms.

## Step 4: measure in this order

| Order | Command in terminal C | About |
|---|---|---|
| 0c, S2 | `run_test.zsh 0c --config s2` | 10 minutes |
| 0c, S1 | `run_test.zsh 0c --config s1` (B runs `run_robotd.zsh s1`) | 10 minutes |
| 0a | `run_test.zsh 0a --config <the one 0c selected>` | 20 minutes |
| 0b | `run_test.zsh 0b --config <the one 0c selected>` | 6 minutes |

You can run only some parts: `--parts c1,c3`, `--parts a3f,a3t`, `--parts b_H2`. `--quick` does each
part only once or twice, to check that the script runs.
When a part fails, run the same part with the config of the same name plus u (for example `s1u`) to
see whether the tuning causes it.

## Step 5: repeat on the board (the selected config)

| Terminal | Command |
|---|---|
| A | `scripts/sim/stage0/run_probe_body.zsh --board` |
| B | `scripts/board/stage0/run_board_robotd.zsh <config>` |
| C | see below |

- 0c: `run_test.zsh 0c --config <config> --target board --parts c2 --reps 1 --push <the force the Mac sweep found>`
  (one push in each of 4 directions; for the force, take the first `fell: true` in `c2_sweep` in
  that Mac run's `summary.json`)
- 0a: `run_test.zsh 0a --config <config> --target board --parts a2,a3f,a3t`
- 0b: `run_test.zsh 0b --config <config> --target board --parts b_H2`

## Where the results are

Each run has one folder, `runs/<UTC>_s0<a|b|c>_<config>/`:

| File | Content |
|---|---|
| `summary.json` | `verdict` (the conclusion against the plan's acceptance values, `pass` true/false; null for a quick run or a run of only some parts) and `parts` (the numbers of every trial) |
| `truth.csv` | Sim truth, one row every 20 ms: trunk position/heading/velocity, beak tip `tip_*`, camera `cam_*` (`cam_pitch` = pitch, positive is down), whether the feet and other parts touch the ground, `mark` = the marks the script stamps |
| `state.jsonl` | robotd's state every tick (skeleton and frames removed) |
| `events.jsonl` | Every command the script sent and every test-port operation |
| `trials.jsonl` | The time points of every trial |

After a change to the scoring rules there is no need to run again: `run_test.zsh 0c --rescore runs/<folder>`
rewrites `summary.json` with the new rules.

When a run is done, tell me the folder name. I will read `summary.json`, spot-check `truth.csv` and
compare every item with the acceptance values.

## Two things to know

- The truth is used only for scoring and never reaches the controller: robotd still sees only the
  sensor data of the upstream protocol.
- Laying the robot down and resetting it are "teleports": a reset puts the robot back in the standing
  pose and holds it for 1 s (like the start-up support); a lying pose is not held and is dropped from
  5 mm above the ground. So c3 measures how well the stand network gets up from lying still, the same
  kind of get-up as the lab's bench_getup.
