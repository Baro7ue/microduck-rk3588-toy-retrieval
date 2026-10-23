"""Stage 1G scorer on hand-made truth rows; no simulator."""
from __future__ import annotations

import math
import unittest

from microduck_toy.stage1 import s1g_carry


def row(t, x=0.0, y=0.0, yaw=0.0, holding='bus', toy_dz=0.0, gz=-1.0, z=0.12):
    return {'mono': t, 'x': x, 'y': y, 'yaw': yaw, 'z': z, 'gx': 0.0, 'gy': 0.0, 'gz': gz,
            'holding': holding, 'tip_x': x + 0.08, 'tip_y': y, 'tip_z': 0.2,
            'bus_x': x + 0.08, 'bus_y': y, 'bus_z': 0.2 + toy_dz}


def walk_rows(t0, t1, speed=0.2, **kw):
    rows, t = [], t0
    while t <= t1 + 1e-9:
        rows.append(row(round(t, 3), x=speed * (t - t0), **kw))
        t += 0.02
    return rows


def step(name, kind, begin, end, held_read=True, head='H0', value=None):
    return {'name': name, 'kind': kind, 'head': head, 'value': value, 'begin': begin, 'end': end,
            'held_read': held_read}


class StepTests(unittest.TestCase):
    def test_walk_distance_and_sway(self):
        rows = walk_rows(0.0, 2.5)
        rows[60]['bus_z'] += 0.004                   # the toy dips 4 mm in the beak once
        st = s1g_carry.score_step(rows, 'bus', step('walk_H0_0.2', 'walk', 0.0, 2.5, value=0.2))
        self.assertTrue(st['held'])
        self.assertAlmostEqual(st['forward_m'], 0.5, places=3)
        self.assertAlmostEqual(st['drift_m'], 0.0)
        self.assertAlmostEqual(st['sway_m'], 0.004, places=6)
        self.assertFalse(st['fell'])
        self.assertAlmostEqual(st['max_tilt_deg'], 0.0)

    def test_turn_angle(self):
        rows = [row(t / 50, yaw=math.radians(90) * t / 100) for t in range(101)]
        st = s1g_carry.score_step(rows, 'bus', step('turn_left', 'turn', 0.0, 2.0, value=90))
        self.assertAlmostEqual(st['turned_deg'], 90.0, places=3)

    def test_drop_and_fall(self):
        rows = walk_rows(0.0, 2.5)
        for r in rows[80:]:
            r.update(holding='', gz=-0.2, z=0.04)     # fell and let go
        st = s1g_carry.score_step(rows, 'bus', step('walk_H0_0.3', 'walk', 0.0, 2.5, held_read=False))
        self.assertFalse(st['held'])
        self.assertTrue(st['fell'])
        self.assertNotIn('sway_m', st)


def result(dropped=False, fell=False, held_at_carry=True, legs=8, by='pick'):
    steps = [{'name': 'stand', 'kind': 'stand', 'fell': False, 'held': True, 'sway_m': 0.001,
              'max_tilt_deg': 2.0}]
    for leg in s1g_carry.LEGS[:legs]:
        name = s1g_carry.leg_name(*leg)
        entry = {'name': name, 'kind': leg[0], 'fell': fell and name == 'walk_H0_0.3',
                 'held': not dropped, 'sway_m': 0.002, 'max_tilt_deg': 5.0, 'seconds': 3.0}
        if leg[0] == 'walk':
            entry.update(forward_m=0.48, drift_m=0.01)
        else:
            entry.update(turned_deg=leg[2] * 0.95)
        steps.append(entry)
    return {'loaded_by': by, 'fell': fell, 'dropped': dropped, 'held_at_carry': held_at_carry,
            'legs_done': legs, 'robot_read_matches': True, 'steps': steps}


class VerdictTests(unittest.TestCase):
    def test_clean_run_passes(self):
        v = s1g_carry.verdict([result() for _ in range(9)], quick=False)
        self.assertTrue(v['pass'])
        self.assertEqual(v['routes_complete'], 9)
        self.assertEqual(v['by_step']['walk_H1_0.3']['n'], 9)
        self.assertAlmostEqual(v['by_step']['turn_right']['turned_deg']['mean'], -85.5)
        self.assertEqual(v['by_step']['turn_left']['n'], 18)

    def test_a_fall_or_a_drop_fails(self):
        self.assertFalse(s1g_carry.verdict([result(), result(fell=True, legs=3)], False)['pass'])
        self.assertFalse(s1g_carry.verdict([result(dropped=True)], False)['pass'])
        self.assertFalse(s1g_carry.verdict([result(held_at_carry=False)], False)['pass'])
        v = s1g_carry.verdict([result(fell=True, legs=3)], False)
        self.assertEqual(v['by_step']['walk_H0_0.3']['falls'], 1)

    def test_quick_is_not_a_verdict(self):
        self.assertIsNone(s1g_carry.verdict([result()], quick=True)['pass'])

    def test_route_stays_in_the_field(self):
        x, y, yaw = s1g_carry.ROUTE_START
        xs, ys = [x], [y]
        for kind, _, value in s1g_carry.LEGS:
            if kind == 'walk':
                x += s1g_carry.WALK_M * math.cos(yaw)
                y += s1g_carry.WALK_M * math.sin(yaw)
                xs.append(x)
                ys.append(y)
            else:
                yaw += math.radians(value)
        self.assertLessEqual(max(map(abs, xs + ys)), 0.6 + 1e-9)     # walls at 1 m, 0.3 m clear
        self.assertLess(max(xs), 0.65 - 0.15 - 0.35)                  # tray centre 0.65, 0.3 m wide
        heads = {(kind, head) for kind, head, _ in s1g_carry.LEGS}
        self.assertEqual({h for k, h in heads if k == 'walk'}, {'H0', 'H1'})
        speeds = sorted(v for k, _, v in s1g_carry.LEGS if k == 'walk')
        self.assertEqual(speeds, [0.2, 0.2, 0.3, 0.3])


if __name__ == '__main__':
    unittest.main()
