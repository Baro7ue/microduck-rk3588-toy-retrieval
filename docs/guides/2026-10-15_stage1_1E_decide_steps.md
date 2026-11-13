# 1E: writing `decide()` in four steps

The only file to change is `src/microduck_toy/control/inching.py`; write from the line
`# ---- Ethan: from here` downwards. Each step is a few lines, followed by the few tests that belong
to that step; go on to the next step once they are all `OK`. Please type the code yourself rather
than copying it. `____` marks the blanks for you to fill.

Rules 1-7 are in section 4 of `docs/guides/2026-10-15_stage1_1E_inching.md`.

To run single tests use `-k` followed by part of the test name (several `-k` are allowed). Run every
command from the repository root:

```
cd "/Users/ethan/Documents/RK3588 Microduck"
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k fallen -k lost
```

A few Python forms, compared with C:

| Python | The same in C |
|---|---|
| `look.toy is None` | `look->toy == NULL` |
| `look.toy[0]`, `look.toy[1]` | the forward and sideways components, like `toy.x`, `toy.y` |
| `pulse.forward_m[0]`, `pulse.forward_m[1]` | mean and standard deviation of this pulse's forward travel (metres) |
| `for name, pulse in self.pulses.items():` | `for (i = 0; i < n; i++) { name = tbl[i].name; pulse = &tbl[i]; ... }` |
| `continue`, `return` | as in C |

## Step 1: handle the accidents first (rules 1, 2)

```python
    def decide(self, look: Look) -> str:
        if ____:                 # rule 1: it has fallen
            return ABORT
        if ____:                 # rule 2: the toy was not seen in this round
            return RETREAT
        return RETREAT           # placeholder; step 2 continues from here
```

Hint: the fields of `Look` are further up in the same file (`toy`, `seen`, `frames`, `elapsed_s`, `fallen`).

Predict before you press Enter: how many tests will the next command run, and with what result?

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k fallen -k lost
```

## Step 2: pick when close enough, retreat when the budget is spent (rules 3, 4)

Replace the placeholder `return RETREAT` at the end of step 1 with:

```python
        if ____:                 # rule 3: the toy is in the window
            return PICK
        if ____ or ____:         # rule 4: 12 pulses sent, or 30 s gone
            return RETREAT
        return RETREAT           # placeholder; step 3 replaces it
```

Hints:

- You do not have to work out "in the window" yourself: `self.target.inside(look.toy)` is already
  written. It uses the config's +-1.5 cm window, which satisfies rule 3 (must pick within +-1.0 cm,
  must never pick more than 2.5 cm off).
- The two limits of the budget are `self.max_pulses` and `self.max_s`. The pulses sent so far are
  counted in `self.pulses_sent` (zeroed in `reset()`). The time since this round began is `look.elapsed_s`.

Think about it: why does "pick" have to come **before** "budget spent"? (One test checks exactly this.)

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k picks_inside -k far_off -k pick_wins -k time_budget
```

## Step 3: choose a pulse (rules 5, 6)

The idea: "pretend to send" each available pulse in turn, use `pulse.predict(toy position)` to work
out where the toy would be afterwards, and choose the one whose landing point is nearest the target
point. A forward pulse must pass rule 6 first; if it does not, skip it.

First add two small functions below `decide` (indented like `decide`):

```python
    def _fits(self, pulse: Pulse, toy) -> bool:
        """Rule 6: a forward pulse's mean + 2 standard deviations <= remaining distance + half the window."""
        room = ____              # remaining distance + half the window (metres)
        return ____ <= room

    def _cost(self, point) -> float:
        """How far the landing point is from the target point; smaller is better."""
        ex, ey = self.target.error(point)      # forward error, sideways error (metres)
        return ____
```

Then replace the placeholder `return RETREAT` at the end of step 2 with:

```python
        best_name, best_cost = None, None
        for name, pulse in self.pulses.items():
            if pulse.role == 'forward' and ____:   # a forward pulse that fails rule 6: skip
                continue
            c = self._cost(____)                   # where the toy will be after this pulse
            if best_cost is None or c < best_cost:
                best_name, best_cost = name, c
        if best_name is None:
            return RETREAT
        ____                                       # book-keeping: one more pulse sent
        return best_name
```

Hints:

- Remaining distance = the toy's forward distance - the target's forward distance. The target's
  forward distance is `self.target.forward_m` (0.08 m); half the window is `self.target.window_m` (0.015 m).
- Distance is Pythagoras, but the square root can be left out: to compare sizes, the sum of squares
  is enough. Think about why.
- Rule 5 (only pulses in `self.pulses`) needs no extra code: the loop only looks at `self.pulses` anyway.
- Why is the book-keeping at the end and not at the start of `decide`? Because PICK, RETREAT and
  ABORT are not pulses and should not count towards the 12.

Predict first: of these 5 tests, which is most likely to fail if the book-keeping is forgotten?

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py -k answers_are -k only_the_allowed -k overshoot -k pulse_budget -k same_looks
```

## Step 4: the closed-loop simulation (the pass line)

```
scripts/tools/run_tests.zsh -p test_stage1_inching.py
```

You should see `Ran 26 tests` and `OK`, with a line `1E closed loop: {...}` printed in between. The
pass line is picked >= 0.80, grasp_p_per_close >= 0.85, mean_pulses <= 7.

I ran this skeleton in the cloud with the blanks filled in the most direct way: picked about 0.86,
grasp_p_per_close about 0.90, 6.0 pulses on average. It passes, but not far above 0.80 or below 7.

To watch each of its steps:

```
source env.sh
"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 3 --show 3
```

## Step 5 (optional, recommended): widen the margin

First run `"$MD_PY" -m microduck_toy.control.pulse_sim --episodes 500` and note the result. Then
change one thing at a time, run it again and fill in this table:

| What changed | picked | parked | grasp_p_per_close | mean_pulses |
|---|---|---|---|---|
| Step 4 as it is | | | | |
| Sideways error x 2 in `_cost` | | | | |
| Sideways error x 3 in `_cost` | | | | |
| An idea of your own | | | | |

Three directions to think in:

1. **Why does sideways matter more?** Every forward pulse turns the robot about 5 deg to the left,
   so the toy moves to the right in the picture; and H2's picture is narrow on the left and wide on the right.
2. **The gap.** With the toy 9.5-10.0 cm ahead, no forward pulse passes rule 6. Find such a case with
   `--show` and see whether the robot keeps turning on the spot.
3. **The biggest loss is `retreat_unseen`**, the toy leaving the picture. See how much of `ends` it
   is, and think about which kind of pulse sends the toy out.

## When you are done

Once the tests are `OK`, say so here. I will read `inching.py` straight from your Mac to review it;
no need to send the file. Then answer in your own words the 3 questions of section 8 of the 1E
guide, plus one more:

4. If the book-keeping of step 3 were the first line of `decide`, which test would fail, and why?
