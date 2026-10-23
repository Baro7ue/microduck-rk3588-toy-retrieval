"""1E inching: the given parts (pulse table, toy motion, window, the H2 view, the pulse sim) and the
rules and closed-loop bar for InchingController.decide, which Ethan writes.

    python -m unittest tests.test_stage1_inching            # from the repo root, after env.sh

ControllerRuleTests and ClosedLoopTests fail until decide() is written. To try another controller
class without editing the tests: MD_INCHING_CONTROLLER=module:Class (default
microduck_toy.control.inching:InchingController)."""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import random
import unittest

from microduck_toy.control import inching, pulse_sim
from microduck_toy.control.inching import ABORT, PICK, RETREAT, Look, Target, move_toy

REPO = Path(__file__).resolve().parents[1]
STAGE1 = json.loads((REPO / 'configs/control/stage1.json').read_text())
FACTS = json.loads((REPO / 'configs/control/stage1_facts.json').read_text())
INCH = STAGE1['inching']
TX, TY = INCH['target_m']
W = INCH['window_m']
FAR = 0.025          # never pick this far off on either axis


def controller_class():
    spec = os.environ.get('MD_INCHING_CONTROLLER')
    if not spec:
        return inching.InchingController
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)


def make(pulses=None):
    cfg = dict(INCH)
    if pulses is not None:
        cfg['pulses'] = pulses
    return controller_class().from_config(cfg, FACTS)


def look(x, y, seen=5, elapsed=2.0, fallen=False):
    toy = None if x is None else (x, y)
    return Look(toy=toy, seen=seen if toy else 0, frames=INCH['frames'], elapsed_s=elapsed,
                fallen=fallen)


# ---- given parts: these pass now ----------------------------------------------------------------

class PulseTableTests(unittest.TestCase):
    def test_units(self):
        f2 = inching.load_pulses(FACTS, ['F2'])['F2']
        self.assertAlmostEqual(f2.forward_m[0], 0.0214)
        self.assertAlmostEqual(f2.yaw_rad[0], math.radians(4.88))
        self.assertEqual((f2.vx_mps, f2.wz_radps, f2.seconds, f2.role), (0.25, 0.0, 0.3, 'forward'))

    def test_config_names_are_measured(self):
        table = FACTS['pulses_mac']
        for name in INCH['pulses']:
            self.assertIn(table[name]['role'], ('forward', 'turn'), name)
        self.assertEqual(table[INCH['retreat']]['role'], 'retreat')
        self.assertNotIn(INCH['retreat'], INCH['pulses'])


class ToyMotionTests(unittest.TestCase):
    def test_walking_forward_brings_the_toy_closer(self):
        x, y = move_toy((0.10, 0.01), 0.02, 0.0, 0.0)
        self.assertAlmostEqual(x, 0.08)
        self.assertAlmostEqual(y, 0.01)

    def test_turning_left_moves_the_toy_right(self):
        x, y = move_toy((0.10, 0.0), 0.0, 0.0, math.pi / 2)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, -0.10)

    def test_measured_pulses(self):
        p = inching.load_pulses(FACTS, ['F2', 'F5', 'L2', 'R17'])
        x, y = p['F2'].predict((0.15, 0.03))
        self.assertAlmostEqual(x, 0.15 - 0.0214, delta=0.003)
        self.assertLess(y, 0.03 - 0.007)                # F2 turns 4.9 deg left: toy ~0.9 cm right
        self.assertLess(p['L2'].predict((0.15, 0.03))[1], 0.03)
        self.assertGreater(p['R17'].predict((0.12, -0.02))[1], 0.01)


class TargetTests(unittest.TestCase):
    def test_window(self):
        t = Target(TX, TY, W)
        self.assertTrue(t.inside((TX, TY)))
        self.assertTrue(t.inside((TX + W, TY - W)))
        self.assertFalse(t.inside((TX + W + 0.001, TY)))
        self.assertFalse(t.inside((TX, TY + W + 0.001)))
        self.assertEqual(t.error((0.10, 0.0)), (0.10 - TX, -TY))


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.view = pulse_sim.H2View.from_configs(STAGE1, FACTS)

    def test_target_is_in_view(self):
        self.assertTrue(self.view.visible((TX, TY)))

    def test_under_the_head_is_not(self):
        self.assertFalse(self.view.visible((0.06, 0.0)))

    def test_lopsided(self):
        # 1D's camera pose: the H2 image reaches further right of the trunk than left
        self.assertTrue(self.view.visible((0.20, -0.08)))
        self.assertFalse(self.view.visible((0.20, 0.08)))


class Scripted(inching.InchingController):
    """Plays fixed answers, for checking the sim itself."""
    script = ()

    def reset(self):
        self.i = 0

    def decide(self, look):
        answer = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return answer


class PulseSimTests(unittest.TestCase):
    def sim(self, script, episodes=50, seed=3):
        cls = type('S', (Scripted,), {'script': tuple(script)})
        return pulse_sim.simulate(cls.from_config(INCH, FACTS), episodes, seed, STAGE1, FACTS)

    def test_pick_at_once_scores_the_start(self):
        out = self.sim([PICK])
        s = out['summary']
        self.assertEqual((s['picked'], s['mean_pulses']), (1.0, 0.0))
        self.assertEqual(s['parked'], 0.0)              # the B segment never leaves it parked
        for r in out['results']:
            self.assertEqual(r['truth_end_m'], r['start'])

    def test_time_adds_up(self):
        r = self.sim(['F5', RETREAT], episodes=1)['results'][0]
        look_s = INCH['frames'] / INCH['sim']['camera_fps']
        self.assertAlmostEqual(r['elapsed_s'], 2 * look_s + 0.5 + INCH['settle_s'], places=2)
        self.assertEqual((r['end'], r['pulses']), (RETREAT, 1))

    def test_same_seed_same_run(self):
        a = self.sim(['F2', 'L2', 'F5', PICK], seed=7)
        b = self.sim(['F2', 'L2', 'F5', PICK], seed=7)
        self.assertEqual(a, b)

    def test_unknown_action_is_refused(self):
        with self.assertRaises(ValueError):
            self.sim(['B10'])                           # the retreat pulse is the runner's, not decide's

    def test_runaway_is_cut(self):
        s = self.sim(['L2'], episodes=5)['summary']
        self.assertEqual(s['ends'], {'runaway': 5})


# ---- Ethan's part: these fail until decide() is written ------------------------------------------

class ControllerTestCase(unittest.TestCase):
    def decide(self, ctl, lk):
        try:
            return ctl.decide(lk)
        except NotImplementedError:
            self.fail('InchingController.decide is not written yet (1E)')


class ControllerRuleTests(ControllerTestCase):
    def test_fallen_aborts_whatever_it_sees(self):
        for lk in (look(TX, TY, fallen=True), look(0.2, 0.0, fallen=True), look(None, None, fallen=True)):
            ctl = make()
            self.assertEqual(self.decide(ctl, lk), ABORT)

    def test_lost_toy_retreats(self):
        ctl = make()
        self.assertEqual(self.decide(ctl, look(None, None)), RETREAT)

    def test_picks_inside_the_window(self):
        for dx, dy in ((0, 0), (0.01, 0.01), (-0.01, -0.01), (0.01, -0.01), (-0.01, 0.01)):
            ctl = make()
            self.assertEqual(self.decide(ctl, look(TX + dx, TY + dy)), PICK, (dx, dy))

    def test_never_picks_far_off(self):
        # Between the +-1.0 cm that must pick and these +-2.5 cm the choice is yours: just past
        # the window's far edge (9.5-10.0 cm) no forward pulse passes the overshoot rule, so a
        # pick there (grasp chance 0.75-0.86 by the weld model) can beat more pulses.
        for dx, dy in ((FAR, 0), (-FAR, 0), (0, FAR), (0, -FAR)):
            ctl = make()
            self.assertNotEqual(self.decide(ctl, look(TX + dx, TY + dy)), PICK, (dx, dy))

    def test_pick_wins_over_the_budget(self):
        ctl = make()
        self.assertEqual(self.decide(ctl, look(TX, TY, elapsed=INCH['max_s'] + 5)), PICK)

    def test_time_budget(self):
        ctl = make()
        self.assertEqual(self.decide(ctl, look(0.20, TY, elapsed=INCH['max_s'])), RETREAT)

    def test_pulse_budget_and_reset(self):
        ctl = make()
        far = look(0.25, TY)
        for i in range(INCH['max_pulses']):
            self.assertIn(self.decide(ctl, far), ctl.pulses, f'decision {i + 1}')
        self.assertEqual(self.decide(ctl, far), RETREAT)
        ctl.reset()
        self.assertIn(self.decide(ctl, far), ctl.pulses)

    def test_answers_are_actions(self):
        for x, y in grid():
            ctl = make()
            self.assertIn(self.decide(ctl, look(x, y)), ctl.actions(), (x, y))

    def test_only_the_allowed_pulses(self):
        for x, y in grid():
            ctl = make(['F2', 'F5', 'L2'])
            self.assertNotEqual(self.decide(ctl, look(x, y)), 'R17', (x, y))

    def test_forward_pulses_do_not_overshoot(self):
        # plan 1E: a forward pulse only when its mean + 2 sd fits in what is left plus the window
        for x, y in grid():
            ctl = make()
            action = self.decide(ctl, look(x, y))
            pulse = ctl.pulses.get(action)
            if pulse is not None and pulse.role == 'forward':
                room = x - TX + W
                self.assertLessEqual(pulse.forward_m[0] + 2 * pulse.forward_m[1], room + 1e-9,
                                     f'{action} at {x:.3f}, {y:.3f}')

    def test_same_looks_same_answers(self):
        rng = random.Random(4)
        looks = [look(rng.uniform(0.09, 0.26), rng.uniform(-0.04, 0.05), elapsed=i) for i in range(10)]
        a, b = make(), make()
        self.assertEqual([self.decide(a, lk) for lk in looks], [self.decide(b, lk) for lk in looks])


def grid():
    for i in range(18):
        for j in range(10):
            yield 0.09 + 0.01 * i, -0.05 + 0.01 * j


class ClosedLoopTests(ControllerTestCase):
    """The bar for 1F: 500 C segments in control/pulse_sim.py with stage1.json's pulses.
    picked = segments that end in a pick (not a retreat); grasp_p_per_close = the weld model's
    chance at the true spot, averaged over picks; mean_pulses per segment."""

    def test_closed_loop(self):
        ctl = make()
        try:
            out = pulse_sim.simulate(ctl, 500, 1, STAGE1, FACTS)
        except NotImplementedError:
            self.fail('InchingController.decide is not written yet (1E)')
        s = out['summary']
        print('\n1E closed loop:', json.dumps(s))
        self.assertGreaterEqual(s['picked'], 0.80)
        self.assertGreaterEqual(s['grasp_p_per_close'], 0.85)
        self.assertLessEqual(s['mean_pulses'], 7.0)


if __name__ == '__main__':
    unittest.main()
