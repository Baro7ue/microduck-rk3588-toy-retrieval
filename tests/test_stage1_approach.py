"""1E/1F approach runner without a simulator: the 1D correction table, ranging a box into a Look,
the seeded 1F starts and the 1F verdict."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from microduck_toy.control import approach, ranging
from microduck_toy.stage1 import s1f_approach

REPO = Path(__file__).resolve().parents[1]
STAGE1 = json.loads((REPO / 'configs/control/stage1.json').read_text())
FACTS = json.loads((REPO / 'configs/control/stage1_facts.json').read_text())


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.c = approach.Correction(FACTS['ranging']['H2']['correction_table_cm'])

    def test_at_a_key(self):
        x, y = self.c.apply((0.15, 0.02))
        self.assertAlmostEqual(x, 0.15 + 0.0064)
        self.assertAlmostEqual(y, 0.02 - 0.0004)

    def test_between_and_beyond(self):
        x, _ = self.c.apply((0.09, 0.0))       # halfway between +0.64 (0.08) and -0.32 (0.10) cm
        self.assertAlmostEqual(x, 0.09 - 0.0016)
        self.assertAlmostEqual(self.c.apply((0.40, 0.0))[0], 0.40 + 0.0075)
        self.assertAlmostEqual(self.c.apply((0.05, 0.0))[0], 0.05 - 0.0064)


class FakeRobot:
    def __init__(self, state):
        self.state = state
        self.falls = []

    def snapshot(self):
        return self.state


def h2_state() -> dict:
    f = FACTS['ranging']['H2']['camera_frame']
    return {'safety': {'gravity': f['gravity'], 'fallen': False},
            'odom': {'position': [0.0, 0.0, f['trunk_z_m']], 'yaw': 0.0},
            'frames': {'camera': {'pos': f['camera_pos_m'], 'quat': f['camera_quat']}}}


class EyesTests(unittest.TestCase):
    def setUp(self):
        self.cam = ranging.CameraModel.from_config(STAGE1['camera'])
        self.state = h2_state()

    def eyes(self, point, every=1):
        col, row = ranging.project(self.cam, self.state, (point[0], point[1], 0.01))
        box = {'name': 'bus', 'xyxy': [col - 20, row - 10, col + 20, row + 10], 'cut': False}
        calls = {'n': 0}

        def next_boxes(after):
            calls['n'] += 1
            return after + 1, ([box] if calls['n'] % every == 0 else [])
        return approach.Eyes(next_boxes, self.cam, FACTS['ranging'])

    def test_ranges_and_corrects(self):
        seen = self.eyes((0.15, 0.02)).look(FakeRobot(self.state), 'H2', 'bus', 0.01, 5)
        self.assertEqual((seen['seen'], seen['frames']), (5, 5))
        raw = seen['raw'][0]
        self.assertAlmostEqual(raw[0], 0.15, places=4)
        self.assertAlmostEqual(seen['toy'][0], 0.15 + 0.0064, places=4)

    def test_other_toys_do_not_count(self):
        seen = self.eyes((0.15, 0.02)).look(FakeRobot(self.state), 'H2', 'henry', 0.01, 5)
        self.assertEqual(seen['seen'], 0)
        self.assertIsNone(seen['toy'])


class StartTests(unittest.TestCase):
    def test_seeded_and_interleaved(self):
        a, b = s1f_approach.starts(10), s1f_approach.starts(10)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 30)
        self.assertEqual([s['toy'] for s in a[:3]], s1f_approach.TARGETS)
        self.assertEqual(s1f_approach.starts(3), a[:9])     # the board's 9 are the Mac's first 9
        for s in a:
            self.assertTrue(0.22 <= s['gap_m'] <= 0.28)
            self.assertLessEqual(abs(s['left_m']), 0.05)
            self.assertLessEqual(abs(s['robot_yaw_deg']), 10.0)
            self.assertTrue(0.0 <= s['toy_yaw_deg'] <= 180.0)


def result(toy, picks, start_s=30.0, fell=False):
    return {'toy': toy, 'picks': picks, 'approach_ends': ['pick'] * len(picks), 'pulses': [5],
            'fell': fell, 'start_s': start_s,
            'held_within_3': any(p['grasped'] for p in picks[:3]),
            'grasp_p_at_trigger': []}


def pick(grasped, err=(0.01, 0.0), ran=True):
    return {'ran': ran, 'grasped': grasped, 'park_err_m': list(err)}


class VerdictTests(unittest.TestCase):
    def test_success_per_close(self):
        rs = [result('bus', [pick(False), pick(True)]), result('henry', [pick(True)]),
              result('ketchup', [pick(False), pick(False), pick(False)])]
        v = s1f_approach.verdict(rs)
        self.assertEqual((v['closes'], v['grasped']), (6, 2))
        self.assertAlmostEqual(v['success_per_close'], 2 / 6)
        self.assertFalse(v['pass'])
        self.assertEqual(v['held_within_3']['k'], 2)
        self.assertAlmostEqual(v['park_err_rms_m'], 0.01)
        self.assertEqual(v['by_toy']['ketchup'], {'closes': 3, 'grasped': 0})

    def test_no_close_is_not_a_verdict(self):
        rs = [result('bus', [pick(False, ran=False)])]
        v = s1f_approach.verdict(rs)
        self.assertEqual(v['closes'], 0)
        self.assertIsNone(v['pass'])
        self.assertAlmostEqual(v['park_err_rms_m'], 0.01)    # the park still counts

    def test_grasp_p(self):
        self.assertAlmostEqual(s1f_approach.grasp_p([0.02, 0.0], 0.04), 0.75)
        self.assertEqual(s1f_approach.grasp_p([0.05, 0.0], 0.04), 0.0)


if __name__ == '__main__':
    unittest.main()
