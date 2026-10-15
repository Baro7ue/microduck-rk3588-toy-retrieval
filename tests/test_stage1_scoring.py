"""Stage 1 scorers changed after the first Mac run (2026-10-15): b_fall tries, 1C pulse table,
1D colour box per toy. Pure functions on hand-made results; no simulator."""
from __future__ import annotations

import unittest

from microduck_toy.stage0.link import Probe, UNLOGGED_REPLY
from microduck_toy.stage1 import s1b_grasp, s1c_back, s1d_range


def fall_try(toy, fell, ok=None, held_at_end=False, direction='right'):
    return {'toy': toy, 'direction': direction, 'speed': 0.6, 'fell': fell,
            'released_on_fall': bool(ok), 'ok': bool(ok), 'held_at_end': held_at_end}


class FallCheckTests(unittest.TestCase):
    def test_every_toy_falls_and_lets_go(self):
        res = [fall_try('bus', True, ok=True),
               fall_try('henry', False, held_at_end=True), fall_try('henry', True, ok=True),
               fall_try('ketchup', True, ok=True)]
        check = s1b_grasp.fall_check(res)
        self.assertTrue(check['pass'])
        self.assertEqual(check['released_on_fall'], '3/3')
        self.assertEqual((check['pushes_recovered'], check['kept_through_recovery']), (1, 1))

    def test_a_fall_that_keeps_the_toy_fails(self):
        res = [fall_try('bus', True, ok=False), fall_try('henry', True, ok=True)]
        self.assertFalse(s1b_grasp.fall_check(res)['pass'])

    def test_no_fall_is_not_a_verdict(self):
        # the first Mac run: three standing pushes at 1.4 m/s, none knocked velstand over
        res = [fall_try(t, False, held_at_end=True) for t in ('bus', 'henry', 'ketchup')]
        check = s1b_grasp.fall_check(res)
        self.assertIsNone(check['pass'])
        self.assertEqual(check['toys_fell'], 0)

    def test_order_starts_on_the_weak_side(self):
        self.assertEqual(s1b_grasp.FALL_ORDER[0], 'right')
        self.assertEqual(set(s1b_grasp.FALL_ORDER), set(s1b_grasp.FALL_DIRECTIONS))


def pulse(label, forward, turn, fell=False):
    return {'label': label, 'forward_m': forward, 'left_m': 0.0, 'dyaw_deg': turn, 'fell': fell}


class PulseTableTests(unittest.TestCase):
    def test_backward_needs_distance_and_little_turn(self):
        table = s1c_back.pulse_table(
            [pulse('vx-0.4x0.5', -0.105, t) for t in (9.4, 2.1, 19.0)]
            + [pulse('vx-0.40wz-0.8x0.5', -0.09, t) for t in (1.0, -2.0, 4.5)]
            + [pulse('vx-0.30wz-0.4x0.5', -0.04, 0.0), pulse('vx-0.30wz-0.4x0.5', -0.06, 0.0)])
        self.assertFalse(table['vx-0.4x0.5']['usable'])           # turns too much
        self.assertTrue(table['vx-0.40wz-0.8x0.5']['usable'])
        self.assertFalse(table['vx-0.30wz-0.4x0.5']['usable'])    # one pulse < 5 cm back
        self.assertNotIn('straight', table['vx-0.4x0.5'])

    def test_forward_straight_by_mean_and_spread(self):
        table = s1c_back.pulse_table(
            [pulse('vx+0.25x0.3', 0.021, t) for t in (4.9, 2.0, 9.2)]
            + [pulse('vx+0.25wz-0.3x0.3', 0.02, t) for t in (1.5, -1.0, 2.5)]
            + [pulse('vx+0.25wz-0.6x0.3', 0.02, t) for t in (0.0, 0.5, -6.0)]
            + [pulse('vx+0.25wz-0.6x0.5', 0.05, 0.0, fell=True)])
        self.assertFalse(table['vx+0.25x0.3']['straight'])
        self.assertTrue(table['vx+0.25wz-0.3x0.3']['straight'])
        self.assertFalse(table['vx+0.25wz-0.6x0.3']['straight'])  # one at -6 deg
        self.assertFalse(table['vx+0.25wz-0.6x0.5']['straight'])  # fell

    def test_trim_labels_parse_as_forward_or_backward(self):
        for vx, wz, seconds in s1c_back.TRIM:
            label = f'vx{vx:+.2f}wz{wz:+.1f}x{seconds}'
            entry = s1c_back.pulse_table([pulse(label, 0.06 if vx > 0 else -0.06, 0.0)])[label]
            self.assertIn('straight' if vx > 0 else 'usable', entry)


class ColourByToyTests(unittest.TestCase):
    def test_px_and_cm_per_toy(self):
        rows = [{'toy': 'bus', 'colour_centre_px': 1.0, 'colour_err_forward_m': -0.004,
                 'colour_err_left_m': 0.0},
                {'toy': 'henry', 'colour_centre_px': 40.0, 'colour_err_forward_m': -0.02,
                 'colour_err_left_m': 0.01},
                {'toy': 'henry', 'colour_centre_px': None}]
        out = s1d_range.colour_by_toy(rows)
        self.assertEqual(out['bus']['centre_px']['mean'], 1.0)
        self.assertEqual(out['henry']['centre_px']['n'], 1)
        self.assertAlmostEqual(out['henry']['err_forward_m']['mean'], -0.02)

    def test_robot_stands_clear_of_the_tray(self):
        far = max(s1d_range.DISTANCES['H1'])
        self.assertLess(s1d_range.ROBOT_AT[0] + far, 0.5 - 0.05)  # tray spans x 0.5-0.8 m


class ProbeLogTests(unittest.TestCase):
    def test_view_frames_are_not_logged(self):
        logged = []

        class Events:
            def log(self, event, **fields):
                logged.append((event, fields))

        probe = Probe.__new__(Probe)
        probe.events, probe.lock = Events(), __import__('threading').Lock()

        class Stream:
            def write(self, data):
                pass

            def flush(self):
                pass

            def readline(self):
                return b'{"seq": 7, "rgb_b64": "AAAA", "boxes": []}\n'

        probe.stream = Stream()
        answer = probe.call('view', after=6, rgb=True)
        self.assertIn('rgb_b64', answer)                   # the caller still gets the frame
        self.assertEqual(logged[0][0], 'probe.view')
        self.assertNotIn('rgb_b64', logged[0][1]['reply'])
        self.assertEqual(UNLOGGED_REPLY, ('rgb_b64',))


if __name__ == '__main__':
    unittest.main()
