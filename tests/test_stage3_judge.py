"""Stage 3 judging without a simulator: the end reasons on truth rows, the gate verdicts, the 3C
choice of d, the dev episode order and the wall distance used for false ToF stops."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from microduck_toy.stage3 import episode

REPO = Path(__file__).resolve().parents[1]
STAGE1 = json.loads((REPO / 'configs/control/stage1.json').read_text())
STAGE3 = json.loads((REPO / 'configs/control/stage3.json').read_text())
CENTRE = STAGE1['arena']['tray']['centre_m']


def row(**toys):
    """Truth row with every toy far from the tray unless given as name=(x, y, z)."""
    out = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    for k, t in enumerate(episode.TOYS):
        x, y, z = toys.get(t, (-0.5, -0.6 + 0.3 * k, 0.012))
        out.update({f'{t}_x': x, f'{t}_y': y, f'{t}_z': z})
    return out


def rec(end, judge_row, target='bus', grasps=()):
    return {'judge_row': judge_row, 'target': target, 'brain': {'end': end}, 'grasps': list(grasps),
            'start': 0.0, 'end_mark': 1.0}


class JudgeTests(unittest.TestCase):
    def judge(self, r):
        return episode.judge_end(r, CENTRE, STAGE3)

    def test_in_the_tray_is_delivered(self):
        out = self.judge(rec('dropped', row(bus=(CENTRE[0] - 0.12, CENTRE[1] + 0.02, 0.02))))
        self.assertEqual(out['outcome'], 'delivered')
        self.assertTrue(out['delivered_main'])
        self.assertTrue(out['delivered_strict'])

    def test_on_the_rim_is_main_but_not_strict(self):
        out = self.judge(rec('dropped', row(bus=(CENTRE[0] - 0.147, CENTRE[1], 0.02))))
        self.assertEqual(out['outcome'], 'delivered')
        self.assertFalse(out['delivered_strict'])

    def test_short_of_the_tray_is_drop_missed(self):
        out = self.judge(rec('dropped', row(bus=(CENTRE[0] - 0.16, CENTRE[1], 0.012))))
        self.assertEqual(out['outcome'], 'drop_missed')

    def test_still_in_the_beak_above_the_tray_is_not_delivered(self):
        out = self.judge(rec('dropped', row(bus=(CENTRE[0] - 0.1, CENTRE[1], 0.17))))
        self.assertEqual(out['outcome'], 'drop_missed')

    def test_wrong_toy_held_or_in_the_tray(self):
        held = rec('held', row(), grasps=[{'result': 'grasped', 'candidate': 'thomas'}])
        self.assertEqual(self.judge(held)['outcome'], 'wrong_toy')
        tray = rec('dropped', row(croc=(CENTRE[0], CENTRE[1], 0.02)))
        out = self.judge(tray)
        self.assertEqual((out['outcome'], out['others_in_tray']), ('wrong_toy', ['croc']))

    def test_other_ends_pass_through(self):
        for end in ('grasp_fail', 'search_fail', 'timeout', 'error', 'held'):
            self.assertEqual(self.judge(rec(end, row()))['outcome'], end)

    def test_target_offset_is_reported(self):
        out = self.judge(rec('dropped', row(bus=(CENTRE[0] - 0.1, CENTRE[1] + 0.03, 0.02))))
        self.assertEqual(out['target_from_tray_m'], [-0.1, 0.03, 0.02])


def e_result(outcome, strict=True, elapsed=60.0):
    return {'outcome': outcome, 'delivered_strict': strict, 'elapsed_s': elapsed, 'falls': [],
            'tof_stops': [], 'states_s': {'SPIN': 5.0, 'CARRY': 10.0}}


class VerdictTests(unittest.TestCase):
    def test_3e_gate_is_21_of_30(self):
        ok = [e_result('delivered')] * 21 + [e_result('drop_missed')] * 9
        self.assertTrue(episode.verdict_3e(ok)['pass'])
        short = [e_result('delivered')] * 20 + [e_result('drop_missed')] * 10
        self.assertFalse(episode.verdict_3e(short)['pass'])

    def test_3e_any_wrong_toy_fails(self):
        rs = [e_result('delivered')] * 29 + [e_result('wrong_toy')]
        self.assertFalse(episode.verdict_3e(rs)['pass'])

    def test_3e_quick_run_has_no_verdict(self):
        out = episode.verdict_3e([e_result('delivered')] * 3)
        self.assertIsNone(out['pass'])
        self.assertEqual(out['state_s_mean'], {'CARRY': 10.0, 'SPIN': 5.0})

    def drop(self, d, delivered, falls=0, stumbled=False):
        return {'d': d, 'delivered_main': delivered, 'delivered_strict': delivered,
                'falls': [{}] * falls, 'tray_look': {'err_m': [0.01, 0.0]}, 'tray_contact': stumbled,
                'trunk_to_centre_m': d, 'toy_in_beak_m': [0.09, 0.0], 'stumbled': stumbled}

    def test_c_drop_picks_most_deliveries_ties_larger(self):
        rs = ([self.drop(0.19, True)] * 9 + [self.drop(0.21, True)] * 9 +
              [self.drop(0.23, True)] * 6 + [self.drop(0.23, False)] * 3)
        out = episode.verdict_c_drop(rs)
        self.assertEqual(out['chosen_d'], '0.21')
        self.assertTrue(out['pass'])

    def test_c_drop_skips_a_d_with_falls(self):
        rs = ([self.drop(0.19, True)] * 8 + [self.drop(0.19, True, falls=1)] +
              [self.drop(0.21, True)] * 7 + [self.drop(0.21, False)] * 2)
        out = episode.verdict_c_drop(rs)
        self.assertEqual(out['chosen_d'], '0.21')
        self.assertFalse(out['pass'])            # 7/9 < 8/9

    def test_c_drop_ties_go_to_fewer_stumbles_first(self):
        rs = ([self.drop(0.19, True)] * 9 + [self.drop(0.21, True)] * 8 + [self.drop(0.21, True, stumbled=True)] +
              [self.drop(0.23, True)] * 8 + [self.drop(0.23, False)])
        out = episode.verdict_c_drop(rs)
        self.assertEqual(out['by_d']['0.21']['stumbles'], 1)
        self.assertEqual(out['chosen_d'], '0.19')

    def test_d_fall_parts_are_grouped(self):
        rs = [{**e_result('delivered'), 'part': f'd_fall_{s}', 'push': {'pushed': True, 'state': s.upper()},
               'falls': [{'up_after_s': 1.2}]} for s in ('goto', 'approach', 'carry')]
        out = episode.verdicts(rs)['3d_fall']
        self.assertEqual((out['n'], out['pushed'], out['fell']), (3, 3, 3))
        self.assertEqual(sorted(out['by_state']), ['APPROACH', 'CARRY', 'GOTO'])

    def test_d_fall_names_the_robots_the_pushes_did_not_fell(self):
        rs = [{**e_result('delivered'), 'part': 'd_fall_goto', 'label': f'd_fall_goto_{k}',
               'push': {'pushed': True, 'state': 'GOTO'}, 'falls': [{'up_after_s': 1.0}] if k else []}
              for k in range(9)]
        out = episode.verdicts(rs)['3d_fall']
        self.assertEqual(out['not_felled'], ['d_fall_goto_0'])
        self.assertFalse(out['pass'])


class FakeRobot:
    """robot.state and wait() as in control/robot.py, with states set by the test."""

    def __init__(self):
        self.cond = threading.Condition()
        self.state = {'policy': 'stand', 'safety': {}}
        self.falls = []

    def set(self, **kw):
        with self.cond:
            self.state = {**self.state, **kw}
            self.cond.notify_all()

    def wait(self, predicate, timeout, what):
        deadline = time.monotonic() + timeout
        with self.cond:
            while not predicate(self.state):
                if time.monotonic() >= deadline:
                    raise TimeoutError(what)
                self.cond.wait(0.02)
            return self.state


def until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError('timed out')
        time.sleep(0.01)


@mock.patch.object(episode, 'PUSH_FALL_S', 0.1)
class PusherTests(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.pushes = []
        self.fall_on = None

        def call(op, **kw):
            self.pushes.append(round(math.hypot(kw['vx'], kw['vy']), 3))
            if len(self.pushes) == self.fall_on:
                self.robot.set(safety={'fallen': True})
            return {'pushed': [kw['vx'], kw['vy']]}

        self.runner = SimpleNamespace(robot=self.robot, mark=lambda name: time.monotonic(),
                                      probe=SimpleNamespace(call=call))
        self.brain = SimpleNamespace(state='APPROACH')

    def test_a_robot_still_up_is_pushed_again_on_its_next_walk(self):
        self.fall_on = 3
        pusher = episode.Pusher(self.runner, self.brain, 'APPROACH', 'forward', 1.05, 0.0)
        pusher.on_state('APPROACH')
        self.robot.set(policy='walk')
        until(lambda: len(self.pushes) == 2)
        time.sleep(0.15)
        self.assertEqual(len(self.pushes), 2)          # two pushes per walk, never a third right after
        self.robot.set(policy='stand')
        time.sleep(0.05)
        self.robot.set(policy='walk')                   # the next B step
        until(lambda: pusher.finished)
        self.assertEqual(self.pushes, [1.05, 1.26, 1.05])
        self.assertEqual(pusher.done['walks'], 2)
        self.assertTrue(pusher.done['pushed'])

    def test_a_slow_fall_after_the_pushes_ends_the_pushing(self):
        pusher = episode.Pusher(self.runner, self.brain, 'APPROACH', 'forward', 1.05, 0.0)
        pusher.on_state('APPROACH')
        self.robot.set(policy='walk')
        until(lambda: len(self.pushes) == 2)
        time.sleep(0.15)
        self.robot.falls.append({'t': 1.0})            # down 2 s after the push, past PUSH_FALL_S
        self.robot.set(policy='stand')
        until(lambda: pusher.finished)
        self.robot.set(policy='walk')
        time.sleep(0.6)
        self.assertEqual(self.pushes, [1.05, 1.26])

    def test_no_push_after_the_episode_ends(self):
        pusher = episode.Pusher(self.runner, self.brain, 'APPROACH', 'left', 0.0, 1.05)
        pusher.on_state('APPROACH')
        pusher.stop()
        self.robot.set(policy='walk')
        until(lambda: pusher.finished)
        time.sleep(0.5)
        self.assertEqual(self.pushes, [])
        self.assertFalse(pusher.done['pushed'])


class EpisodeListTests(unittest.TestCase):
    def test_all_30_layout_major(self):
        eps = episode.dev_episodes(None, quick=False)
        self.assertEqual(len(eps), 30)
        self.assertEqual([(e['seed'], e['target']) for e in eps[:4]],
                         [(3000, 'bus'), (3000, 'henry'), (3000, 'ketchup'), (3001, 'bus')])

    def test_quick_is_one_per_target(self):
        eps = episode.dev_episodes(None, quick=True)
        self.assertEqual([(e['seed'], e['target']) for e in eps],
                         [(3000, 'bus'), (3001, 'henry'), (3002, 'ketchup')])

    def test_named_episodes(self):
        eps = episode.dev_episodes('3004:ketchup,3000:henry', quick=False)
        self.assertEqual([(e['seed'], e['target']) for e in eps], [(3004, 'ketchup'), (3000, 'henry')])

    def test_layout_puts_the_target_in_the_target_slot(self):
        lay = episode.load_layouts()[0]
        out = episode.episode_layout(lay, 'henry')
        self.assertEqual(out['toys']['henry'], lay['slots']['target'])
        self.assertEqual(sorted(out['toys']), ['croc', 'henry', 'thomas'])


class BoardCommandTests(unittest.TestCase):
    def args(self, cpus):
        return SimpleNamespace(board_bundle='/b/bundle', board_py='/b/venv/bin/python', board_socket='/b/run/robotd.sock',
                               sim_host='192.168.50.1', backend='int8', board_cpus=cpus)

    def test_pinned_to_the_big_cores(self):
        cmd = episode.board_command(self.args('4-7'), '/b/run/x/episodes/e', 'bus', (0.1, 0.2, 0.3), 'delivered')
        self.assertIn('taskset -c 4-7 /b/venv/bin/python -u -m microduck_toy.stage3.brain --target bus', cmd)
        self.assertTrue(cmd.startswith('cd /b/bundle && '))

    def test_empty_means_not_pinned(self):
        cmd = episode.board_command(self.args(''), '/b/out', 'bus', (0.0, 0.0, 0.0), 'held')
        self.assertNotIn('taskset', cmd)
        self.assertIn('--until held', cmd)


class WallTests(unittest.TestCase):
    def test_wall_ahead(self):
        half = STAGE1['arena']['half_size_m']
        self.assertAlmostEqual(episode.wall_ahead({'x': 0.5, 'y': 0.0, 'yaw': 0.0}, STAGE1), half - 0.5)
        self.assertAlmostEqual(episode.wall_ahead({'x': 0.0, 'y': 0.0, 'yaw': math.pi / 4}, STAGE1),
                               half * math.sqrt(2))
        self.assertAlmostEqual(episode.wall_ahead({'x': 0.0, 'y': -0.8, 'yaw': -math.pi / 2}, STAGE1),
                               half - 0.8)

    def test_a_slanted_wall_is_seen_by_the_side_zones(self):
        half = STAGE1['arena']['half_size_m']
        square = {'x': half - 0.25, 'y': 0.0, 'yaw': 0.0}
        self.assertAlmostEqual(episode.wall_in_view(square, STAGE1), 0.25)
        slant = {'x': 0.0, 'y': half - 0.2, 'yaw': math.radians(40)}
        self.assertGreater(episode.wall_ahead(slant, STAGE1), 0.3)
        edge = math.radians(40 + episode.TOF_HALF_FOV_DEG)
        self.assertAlmostEqual(episode.wall_in_view(slant, STAGE1),
                               0.2 / math.sin(edge) * math.cos(math.radians(episode.TOF_HALF_FOV_DEG)))


if __name__ == '__main__':
    unittest.main()
