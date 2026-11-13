# Stage 1, 1E, inching over the last stretch: interface, simulation and the part you write (2026-10-15)

Based on section 7 of the Stage 1 plan (1E) and on the results of 1A-1D and the three reruns on your Mac.
Division of work (on 10-15 you chose "write all three blocks"): I write the interface, the pulse
simulation and the unit tests; you write `decide()`, and I review it when it is done.

## 1. Rerun results (code version 30e6d6d)

| Test | Result | Verdict |
|---|---|---|
| 1B b_fall `runs/20261015T094124Z_s1b_s2` | All 3 toys were released in the fall (bus pushed left at 1.05, Henry pushed right at 0.6, ketchup pushed left at 1.05 m/s); 4 other pushes did not bring it down, and the toy stayed in the beak each time | pass |
| Fall handling (same run) | Zeroing on `safety.fallen` alone: 0.34, 0.35, 0.34 s from the truth fall to velstand taking over (limit 0.35); 0 false alarms during the normal actions of 1a, 1c and c_trim | pass (the plan asked for 4 trials, 3 were run) |
| 1C c_trim `runs/20261015T094236Z_s1c_s2` | With a right turn added the mean yaw gets smaller but the spread doubles (standard deviation 4.2-4.5 deg, was 2.0-2.7 deg); no forward or backward pulse goes straight | recorded in the exit table |
| 1D d_H1 `runs/20261015T095725Z_s1d_s2` | Error -0.77 to -1.17 cm over 0.20-0.60 m, changing smoothly with distance; the toys no longer sit on the tray | pass |

All the numbers are in `configs/control/stage1_facts.json` (the Stage 1 exit table; code from 1E on
reads numbers only from there and from `stage0_facts.json`).

## 2. Two new findings that changed the design of 1E

1. **Every forward pulse drifts left.** F2 (vx 0.25 x 0.3 s) goes 2.1 cm and turns left 4.9 +- 2.7 deg;
   F5 (x 0.5 s) goes 5.3 cm and turns left 4.2 +- 2.1 deg. When the robot turns left, the toy moves
   right in its view: about 1 cm to the right per step at 15 cm. The plan's "aim to the right + only
   turn left" can only correct the toy towards the right, the same direction as this drift, so it
   cannot bring it back.
2. **H2's picture is very narrow on the left.** From 1D's camera pose: 8 cm ahead of the trunk only
   4.2 cm is visible to the left and 7.4 cm to the right; at 20 cm, 6.6 cm left and 10.5 cm right.
   A toy too far left leaves the picture first.

So I added R17 (a turn in place to the right, wz -1.0 x 0.3 s, -17.2 +- 0.9 deg, very steady) to
1E's action table and asked you in the thread whether to keep it. Results of the pulse simulation
(C segment only, 1000 runs per group):

| Action table | Stopped in the window and picked | Expected success per closing |
|---|---|---|
| F2, F5, L2 (left turns only) | about 25-30%; only about 53% even with the B segment's sideways error cut to +-2 cm | about 0.90 |
| F2, F5, L2, R17 | about 90% | about 0.91 |

The B segment's aim offset `aim_right_of_toy_m` changed from the plan's 2 cm to 1 cm (with R17,
0-2 cm are all about the same, and 1 cm is steady at both B-segment accuracies; 1F tunes it again).

## 3. The files this time

| File | Content |
|---|---|
| `src/microduck_toy/control/inching.py` | The interface: `Pulse`, `load_pulses`, `move_toy`, `Target`, `Look`, `InchingController`; **`reset()` and `decide()` are yours to write** |
| `src/microduck_toy/control/pulse_sim.py` | The pulse simulation (no MuJoCo): pulses are sampled from the measured means and standard deviations, ranging gets noise from 1D's errors, and a toy outside H2's picture is not seen |
| `tests/test_stage1_inching.py` | 26 tests: 14 test my part (passing now), 12 test your `decide()` (failing now, all passing once it is written) |
| `configs/control/stage1.json` | New `inching` section: target point, window, budget, available pulses, the B segment's aim offset, and the simulation's parameters in `inching.sim` |
| `configs/control/stage1_facts.json` | The Stage 1 exit table (new file): rerun results, the pulse table `pulses_mac`, the ranging correction table, H2's camera pose |

## 4. What you write: `InchingController.reset()` and `decide(look)`

Each round of the C segment: stand still 1.0 s -> range over 5 frames (the runner has already
applied the correction and the average) -> call `decide(look)` once -> send one pulse or end,
according to the return value.

**Input `Look`** (coordinates in the level trunk frame, metres; x forward, y left):

| Field | Meaning |
|---|---|
| `toy` | The toy's position `(forward, left)`; `None` when none of the 5 frames saw it |
| `seen`, `frames` | Frames that saw the toy / total frames |
| `elapsed_s` | Seconds since this C segment began |
| `fallen` | Whether `safety.fallen` was reported since the last look |

**Output**: one string, which must be one of:
a name in `self.pulses` (`'F2'`, `'F5'`, `'L2'`, `'R17'`), or `PICK` (pick), or `RETREAT` (give up
this try, back off about 10 cm and aim again in the B segment), or `ABORT` (fallen).

**Rules that must hold** (`ControllerRuleTests`, one test per rule):

1. Fallen: return `ABORT`, whatever is seen.
2. Toy not seen: return `RETREAT`.
3. With the toy within +-1.0 cm of the target point both forward and sideways, it must `PICK`, even
   with the budget spent; more than 2.5 cm off in either direction, it must never `PICK`. In between
   is up to you (the config's window is +-1.5 cm).
4. After 12 pulses, or with `elapsed_s` >= 30 s, return `RETREAT`; the count starts again after `reset()`.
5. Only pulses in `self.pulses` may be used (without R17 in the action table, R17 may not be sent).
6. A forward pulse must satisfy the plan's rule: **mean + 2 standard deviations <= remaining
   distance + half the window**, that is `forward_m[0] + 2 * forward_m[1] <= (x - 0.08) + 0.015`.
7. The same input sequence gives the same output (no random numbers).

**The level to reach** (`ClosedLoopTests`, 500 simulated C segments, fixed seed):

| Metric | Threshold |
|---|---|
| Share that ends in a pick (not a retreat), `picked` | >= 0.80 |
| Expected success per closing, `grasp_p_per_close` | >= 0.85 |
| Mean number of pulses, `mean_pulses` | <= 7 |

## 5. Tools you can use directly

- `self.pulses['F2']` is a `Pulse` (like a read-only struct in C): `forward_m`, `left_m` and
  `yaw_rad` are each `(mean, standard deviation)`, and `role` is `'forward'` or `'turn'`.
- `pulse.predict(toy)`: where the toy will be after this pulse, by the means. Behind it is
  `move_toy`: the toy stays and the robot moves, so it is "subtract the translation, then undo the yaw":

```c
/* the robot moved (f, l) and turned left by yaw (radians); the toy in the robot's new frame */
double x = toy_x - f, y = toy_y - l;
double new_x =  cos(yaw) * x + sin(yaw) * y;
double new_y = -sin(yaw) * x + cos(yaw) * y;
```

- `self.target.inside(toy)`, `self.target.error(toy)` (returns `(forward error, sideways error)`).
- `self.max_pulses`, `self.max_s`.

**A gap to deal with**: F2's "mean + 2 standard deviations" is 3.5 cm, more than the window's 3 cm
width, so with the toy 9.5-10.0 cm ahead (just past the window's far edge) no forward pulse satisfies
rule 6. In the cloud pre-run the reference version only kept sending L2 there until it timed out
and retreated. You can pick right there (1.5-2 cm off forward; the weld model's success rate is
about 0.75-0.86), or think of something else.

**A hint (not the answer)**: I wrote the plan's "correct sideways first, then forward" literally in
the simulation: `picked` is about 0.69 and misses 0.80, because it ignores the left drift of the
forward pulses. Working out the landing point of every available pulse with `predict` and choosing
the best one passes the threshold. How to define "best" is up to you.

## 6. Commands

The files are already in the repository on your Mac (around 10-15 10:40Z, md5 checked). Have a look first:

```
cd "/Users/ethan/Documents/RK3588 Microduck"
git status --short
git diff --stat
```

You should see ` M configs/control/stage1.json` (`git diff --stat` shows only 24 more lines) and 5
new `??` files. If not, use the attachment `stage1_1e_20261015.tar.gz`: download it to the Downloads
folder, then `tar -xzf ~/Downloads/stage1_1e_20261015.tar.gz` (this time `stage1.json` must be
overwritten, so no `-k`).

Then run the tests:

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py
```

For now you should see `FAILED (failures=12)`, all of them in `ControllerRuleTests` and
`ClosedLoopTests`, with the message `decide is not written yet`. Once it is written you should see
`OK` and a printed line `1E closed loop: {...}`.

To run the simulation on its own and watch each step:

```
source env.sh
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 500
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 3 --show 3
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 500 --pulses F2,F5,L2
```

`--show 3` prints every step of the first 3 runs (the toy's truth, whether it was seen, your
decision). `--pulses` changes the action table, `--aim` the B segment's aim offset.

## 7. When you are done

Put the code in `inching.py` (change only `reset` and `decide`; add helper functions of your own if
needed) and tell me when the tests are `OK`; I will review it. Meanwhile I write the MuJoCo runner
for 1E/1F: the B segment walks to about 0.25 m with the heading PI, the C segment calls your
`decide()` every round, then it picks and is scored. 1F runs on it.

## 8. Questions to explain (answer in your own words when done)

1. F2 turns the robot 4.9 deg to the left. Why does the toy move right in the picture? By about how much at 15 cm?
2. Why does rule 6 use "mean + 2 standard deviations" rather than the mean alone?
3. The simulation's ranging error has two parts, "one bias per C segment" and "noise per look". Why
   can averaging more frames not remove the first part?

## 9. 1F: approach + pick (run it after your `decide()` tests are `OK`)

New files:

| File | Content |
|---|---|
| `src/microduck_toy/control/approach.py` | The approach (uses only `robot.state` and the boxes in the picture, so it will run unchanged on the board later): the B segment looks once at H1, turns in place towards the aim point or walks forward with the heading PI, and looks again, until the toy is within 0.24 m; the C segment at H2 stands still 1.0 s each round, ranges over 5 frames, calls your `decide()` and sends the pulse it chose |
| `src/microduck_toy/stage1/s1f_approach.py` | The 1F test: starts fixed by seed (robot heading off by +-10 deg, toy 0.25 +- 0.03 m ahead of the beak tip, +-5 cm sideways, heading 0-180 deg); at most 3 closings per start, backing off (B10) and trying again after a miss; scored on truth |
| `configs/control/stage1.json` | New `approach` section: the B segment's turning, straight walking and the distance at which it hands over to the C segment |
| `tests/test_stage1_approach.py` | 8 tests: the correction table, box to position, the starts, the verdict |

Three terminals (as in step 3; A needs `--camera`):

| Terminal | Command |
|---|---|
| A | `scripts/sim/stage1/run_arena_body.zsh --camera` |
| B | `scripts/sim/stage0/run_robotd.zsh s2` |
| C | The smoke test first, `scripts/sim/stage1/run_test.zsh 1f --config s2 --quick` (3 starts), then the formal run `scripts/sim/stage1/run_test.zsh 1f --config s2 --parts f_main` (30 starts), and last the blind-walk comparison `--parts f_blind` (10 starts) |

Verdict (plan section 8): with picks / closings >= 50%, the pick stays the main line. Also checked:
parking error RMS <= 2 cm, starts picked within 3 closings >= 80%, mean time per start <= 40 s,
0 falls. The result is in `verdict.f_main` in `summary.json`.
