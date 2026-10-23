"""C segment inching (plan 1E): stand, look, send one short pulse, stand, look again, until the toy
sits where GroundPick's beak lands; then pick.

    ctl = InchingController.from_config(stage1['inching'], stage1_facts)
    ctl.reset()                                   # start of every C segment
    while True:
        look = Look(toy=(0.13, 0.02), seen=5, frames=5, elapsed_s=4.2)   # after settle_s standing
        action = ctl.decide(look)
        if action in (PICK, RETREAT, ABORT):
            break
        pulse = ctl.pulses[action]                # the runner sends move(pulse.vx, 0, pulse.wz)
                                                  # for pulse.seconds, then zero, then waits

Everything is in the level trunk frame at the moment of the look (control/ranging.py): x forward,
y left, metres; yaw > 0 = turned left. A pulse's effect comes from the Mac measurements in
configs/control/stage1_facts.json pulses_mac (1C), as mean and sd over 10 tries.

What is given here: the pulse table (Pulse, load_pulses), the mean effect of a pulse on where the toy
appears (move_toy, Pulse.predict), the target window (Target), the Look the runner hands over, and
the loading from the config. The 1E brain, InchingController.reset and .decide, was planned as
Ethan's; Claude wrote it on 2026-10-23 at his request and he reviews it. The rules decide must keep
are in tests/test_stage1_inching.py; control/pulse_sim.py runs it against the measured pulse spread
in closed loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

PICK = 'pick'          # the toy is in the window: run control.pick.ground_pick
RETREAT = 'retreat'    # give up this C segment: back off (the retreat pulse), aim again in B
ABORT = 'abort'        # the robot is down: zero the move, head H0, wait for velstand


@dataclass(frozen=True)
class Pulse:
    """One measured pulse: command move(vx, 0, wz) for `seconds`, then zero.
    Effects are (mean, sd) in the robot frame before the pulse."""
    name: str
    vx_mps: float
    wz_radps: float
    seconds: float
    role: str                                # 'forward', 'turn' or 'retreat'
    forward_m: tuple[float, float]
    left_m: tuple[float, float]
    yaw_rad: tuple[float, float]
    settle_p90_s: float

    def predict(self, toy: tuple[float, float]) -> tuple[float, float]:
        """Where a toy now at `toy` should appear after this pulse, using the mean effect."""
        return move_toy(toy, self.forward_m[0], self.left_m[0], self.yaw_rad[0])


def move_toy(toy, forward: float, left: float, yaw: float) -> tuple[float, float]:
    """The toy does not move; the robot does. If the trunk goes `forward`, `left` (metres, in its
    own frame before the move) and turns `yaw` (rad, left > 0), a floor point seen at `toy` is seen
    at R(-yaw) (toy - (forward, left)) afterwards."""
    x, y = toy[0] - forward, toy[1] - left
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x + s * y, -s * x + c * y)


def load_pulses(facts: dict, names) -> dict[str, Pulse]:
    """Pulses by name from stage1_facts.json pulses_mac (cm and deg there, m and rad here)."""
    table = facts['pulses_mac']
    out = {}
    for name in names:
        p = table[name]
        out[name] = Pulse(
            name=name, vx_mps=float(p['vx_mps']), wz_radps=float(p['wz_radps']),
            seconds=float(p['seconds']), role=p['role'],
            forward_m=(p['forward_cm'][0] / 100, p['forward_cm'][1] / 100),
            left_m=(p['left_cm'][0] / 100, p['left_cm'][1] / 100),
            yaw_rad=(math.radians(p['yaw_deg'][0]), math.radians(p['yaw_deg'][1])),
            settle_p90_s=float(p['settle_after_stop_s'][1]))
    return out


@dataclass(frozen=True)
class Target:
    """Where the toy must be for the pick: GroundPick lands the beak here (stage0_facts 0b, H2)."""
    forward_m: float
    left_m: float
    window_m: float                           # half width, the same on both axes

    def error(self, toy) -> tuple[float, float]:
        return toy[0] - self.forward_m, toy[1] - self.left_m

    def inside(self, toy) -> bool:
        ex, ey = self.error(toy)
        return abs(ex) <= self.window_m and abs(ey) <= self.window_m


@dataclass(frozen=True)
class Look:
    """What the runner hands decide() after each settle.
    toy        (forward_m, left_m) averaged over the frames that saw it, after the 1D correction
               table; None when no frame saw it
    seen       frames that saw the toy, of `frames` taken
    elapsed_s  since this C segment started (reset)
    fallen     robot.state safety.fallen on any tick since the last look"""
    toy: tuple[float, float] | None
    seen: int
    frames: int
    elapsed_s: float
    fallen: bool = False


class InchingController:
    """The C segment brain. decide() returns one of: a name in self.pulses, PICK, RETREAT, ABORT."""

    def __init__(self, pulses: dict[str, Pulse], target: Target, max_pulses: int, max_s: float):
        self.pulses = pulses
        self.target = target
        self.max_pulses = max_pulses
        self.max_s = max_s
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict, facts: dict) -> 'InchingController':
        """cfg = stage1.json 'inching'; facts = stage1_facts.json."""
        target = Target(cfg['target_m'][0], cfg['target_m'][1], cfg['window_m'])
        return cls(load_pulses(facts, cfg['pulses']), target, int(cfg['max_pulses']),
                   float(cfg['max_s']))

    def actions(self) -> set[str]:
        return set(self.pulses) | {PICK, RETREAT, ABORT}

    # ---- the 1E brain (written by Claude 2026-10-23, Ethan reviews) -----------------------------
    #
    # Same shape as docs/guides/2026-10-15_stage1_1E_decide_steps.md steps 1-3, plus step 5's
    # lateral weight. Rule numbers are those of the 1E guide, section 4.

    # How much worse a left/right error counts than a forward/back one in _cost (step 5).
    # Every forward pulse turns the robot ~5 deg left, so the toy drifts right after it, and H2
    # sees only ~4 cm left of the centre line: a lateral miss tends to lose the toy, a forward
    # one does not. pulse_sim, 500 segments x 3 seeds at aim 1 cm: weight 1 -> picked 0.85,
    # 6.2 pulses; 2 -> 0.91, 5.7; 3 -> 0.92, 5.5 (grasp chance per close 0.90-0.91 for all).
    LATERAL_WEIGHT = 3.0

    def reset(self) -> None:
        """Start of a C segment: forget everything decide() kept from the last one."""
        self.pulses_sent = 0

    def decide(self, look: Look) -> str:
        """One decision per look. See tests/test_stage1_inching.py ControllerRuleTests for the
        rules it must keep and ClosedLoopTests for how well it must do.

        The checks run from most to least urgent and the first one that fires returns, like a
        chain of `if (...) return ...;` at the top of a C function."""
        # Rule 1: the robot fell since the last look. Nothing else matters; the runner zeroes
        # the move and waits for velstand to stand it up.
        if look.fallen:
            return ABORT

        # Rule 2: no frame saw the toy. `is None` is Python's `== NULL`.
        if look.toy is None:
            return RETREAT

        # Rule 3: the toy is in the +-1.5 cm window around where the beak lands: pick.
        # This sits before the budget check on purpose: a toy in the window is picked even if
        # the 12 pulses or 30 s just ran out (test_pick_wins_over_the_budget).
        if self.target.inside(look.toy):
            return PICK

        # Rule 4: out of budget. Back off (B10) and let the B segment aim again.
        if self.pulses_sent >= self.max_pulses or look.elapsed_s >= self.max_s:
            return RETREAT

        # Rules 5 and 6: try every allowed pulse on paper and keep the one whose predicted
        # landing point is closest to the target. Only names in self.pulses are tried (rule 5);
        # a forward pulse that could carry the toy past the window is skipped (rule 6).
        # In C this is the usual min search:
        #     best = -1; best_cost = 0;
        #     for (i = 0; i < n; i++) {
        #         if (is_forward(&tbl[i]) && !fits(&tbl[i], toy)) continue;
        #         c = cost(predict(&tbl[i], toy));
        #         if (best < 0 || c < best_cost) { best = i; best_cost = c; }
        #     }
        # dict.items() yields (key, value) pairs in config order (F2, F5, L2, R17), so on a tie
        # the earlier name wins and the same look always gets the same answer.
        best_name, best_cost = None, None
        for name, pulse in self.pulses.items():
            if pulse.role == 'forward' and not self._fits(pulse, look.toy):
                continue
            c = self._cost(pulse.predict(look.toy))
            if best_cost is None or c < best_cost:
                best_name, best_cost = name, c

        # Only possible when the config lists forward pulses alone and none fits.
        if best_name is None:
            return RETREAT

        # Count the pulse only now: PICK, RETREAT and ABORT above are not pulses and must not
        # use up the 12 (test_pulse_budget_and_reset).
        self.pulses_sent += 1
        return best_name

    def _fits(self, pulse: Pulse, toy) -> bool:
        """Rule 6: a forward pulse's mean + 2 sd must fit in what is left to the target plus the
        window's half width, so even a long pulse (about 1 in 40) stops inside the window.
        Example: toy 12 cm ahead -> room = 12 - 8 + 1.5 = 5.5 cm; F2 needs 2.14 + 2 x 0.69 =
        3.52 cm (fits), F5 needs 5.27 + 2 x 0.68 = 6.63 cm (does not)."""
        room = (toy[0] - self.target.forward_m) + self.target.window_m
        return pulse.forward_m[0] + 2 * pulse.forward_m[1] <= room

    def _cost(self, point) -> float:
        """How far `point` is from the target; smaller is better. Squared distance without the
        square root: sqrt only changes the numbers, not which one is smallest, so the min search
        picks the same pulse. The lateral part counts LATERAL_WEIGHT times."""
        ex, ey = self.target.error(point)                 # forward and left error, metres
        return ex * ex + self.LATERAL_WEIGHT * ey * ey
