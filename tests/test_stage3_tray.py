"""Stage 3 tray and ToF without a simulator: the drop side, the tray look from synthetic boxes, the
drop walk, and the ToF stop rule on synthetic walls, floor and the beak."""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from microduck_toy.control import ranging
from microduck_toy.stage3 import tof as tof_lib
from microduck_toy.stage3 import tray
from microduck_toy.stage3.world import Field

REPO = Path(__file__).resolve().parents[1]
STAGE1 = json.loads((REPO / 'configs/control/stage1.json').read_text())
STAGE3 = json.loads((REPO / 'configs/control/stage3.json').read_text())
FIELD = Field.from_config(STAGE1, STAGE3)
CAM = ranging.CameraModel.from_config(STAGE1['camera'])
RIM = STAGE1['arena']['tray']['rim_height_m']
HALF = STAGE1['arena']['tray']['size_m'] / 2


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def axis_quat(axis, angle):
    s = math.sin(angle / 2)
    return [math.cos(angle / 2), *(s * a for a in axis)]


FORWARD = [.5, -.5, .5, -.5]    # OpenCV right/down/forward -> trunk -y/-z/+x


def h1_like(pitch_deg=30.0, tof_pitch_deg=0.0):
    """A standing robot with the camera pitched down like H1 (synthetic, level trunk)."""
    return {'frames': {'camera': {'pos': [0.06, 0.0, 0.10],
                                  'quat': quat_mul(axis_quat((0, 1, 0), math.radians(pitch_deg)), FORWARD)},
                       'tof': {'pos': [0.07, 0.0, 0.09],
                               'quat': axis_quat((0, 1, 0), math.radians(tof_pitch_deg))}},
            'imu': {'quat': [1.0, 0.0, 0.0, 0.0]}, 'odom': {'position': [0.0, 0.0, 0.12], 'yaw': 0.0},
            'safety': {'gravity': [0.0, 0.0, -1.0]}}


def tray_box(state, centre):
    """The tray's box as a detector would draw it: the near rim's foot (z = 0) and the far rim's top
    (z = rim height) at the four corners, projected and clipped to the image."""
    f, l = centre
    near = [ranging.project(CAM, state, (f - HALF, l + s * HALF, 0.0)) for s in (1, -1)]
    far = [ranging.project(CAM, state, (f + HALF, l + s * HALF, RIM)) for s in (1, -1)]
    cols = [p[0] for p in near + far]
    rows = [p[1] for p in near + far]
    clip = lambda v, hi: max(0.0, min(hi, v))  # noqa: E731
    return {'cls': 'tray', 'conf': 0.9, 'xyxy': [clip(min(cols), CAM.width), clip(min(rows), CAM.height),
                                                 clip(max(cols), CAM.width), clip(max(rows), CAM.height)]}


class SideTests(unittest.TestCase):
    cfg = STAGE3['carry']

    def test_staging_points_and_facing(self):
        cx, cy = FIELD.tray_centre
        self.assertEqual(tray.staging_point('x-', FIELD, 0.45), (cx - 0.45, cy))
        self.assertAlmostEqual(tray.facing('x-'), 0.0)
        self.assertAlmostEqual(tray.facing('y+'), -math.pi / 2)
        self.assertAlmostEqual(tray.facing('y-'), math.pi / 2)

    def test_nearest_side_wins(self):
        cx, cy = FIELD.tray_centre
        self.assertEqual(tray.choose_side((cx - 1.0, cy), [], FIELD, self.cfg)['side'], 'x-')
        self.assertEqual(tray.choose_side((cx, cy + 0.7), [], FIELD, self.cfg)['side'], 'y+')
        self.assertEqual(tray.choose_side((cx, cy - 0.7), [], FIELD, self.cfg)['side'], 'y-')

    def test_toy_in_the_lane_moves_the_side(self):
        cx, cy = FIELD.tray_centre
        lane_toy = (cx - 0.3, cy)                 # between the x- staging point and the tray edge
        out = tray.choose_side((cx - 1.0, cy), [lane_toy], FIELD, self.cfg)
        self.assertNotEqual(out['side'], 'x-')
        self.assertEqual([o['blocked'] for o in out['options'] if o['side'] == 'x-'], [1])

    def test_plus_x_is_never_a_side(self):
        self.assertNotIn('x+', self.cfg['sides'])


class TrayLookTests(unittest.TestCase):
    cfg = STAGE3['tray_look']

    def test_both_sides_cut_keeps_odometry_sideways(self):
        st = h1_like()
        box = tray_box(st, (0.45, 0.0))
        self.assertEqual(box['xyxy'][0], 0.0)                    # wider than the image
        self.assertEqual(box['xyxy'][2], CAM.width)
        m = tray.measure_tray([(st, [box])] * 5, CAM, self.cfg, HALF, RIM, expected=(0.40, 0.01))
        self.assertEqual((m['source'], m['used']), ('yolo', 5))
        self.assertAlmostEqual(m['centre'][0], 0.45, places=4)
        self.assertAlmostEqual(m['centre'][1], 0.01, places=6)

    def test_both_sides_cut_clips_odometry_to_the_band(self):
        st = h1_like()
        box = tray_box(st, (0.45, 0.0))
        m = tray.measure_tray([(st, [box])], CAM, self.cfg, HALF, RIM, expected=(0.45, 0.08))
        y1 = box['xyxy'][3]
        right_edge = ranging.ground_point(CAM, st, (CAM.width, y1, CAM.width, y1), 0.0)[1]
        # the right rim is beyond the picture's right edge, so the centre is at most right_edge + HALF
        self.assertAlmostEqual(m['centre'][1], right_edge + HALF, places=6)
        self.assertLess(m['centre'][1], 0.08)

    def test_uncut_side_across_the_axis_uses_its_far_corner(self):
        st = h1_like()
        box = tray_box(st, (0.45, 0.17))       # right rim at +2 cm: left of the axis, widest far away
        self.assertLess(box['xyxy'][2], CAM.width / 2)
        m = tray.measure_tray([(st, [box])], CAM, self.cfg, HALF, RIM, expected=(0.45, 0.10))
        self.assertAlmostEqual(m['centre'][0], 0.45, places=4)
        self.assertAlmostEqual(m['centre'][1], 0.17, places=4)

    def test_uncut_side_on_its_own_side_uses_its_near_corner(self):
        st = h1_like()
        box = tray_box(st, (0.45, -0.24))      # left rim at -9 cm: right of the axis, widest near
        self.assertGreater(box['xyxy'][0], CAM.width / 2)
        m = tray.measure_tray([(st, [box])], CAM, self.cfg, HALF, RIM, expected=(0.45, -0.20))
        self.assertAlmostEqual(m['centre'][1], -0.24, places=4)

    def test_far_box_is_rejected(self):
        st = h1_like()
        box = tray_box(st, (0.45, 0.0))
        m = tray.measure_tray([(st, [box])], CAM, self.cfg, HALF, RIM, expected=(0.90, 0.0))
        self.assertEqual((m['source'], m['rejected']), ('odometry', 1))
        self.assertEqual(m['centre'], [0.90, 0.0])

    def test_no_box_falls_back_to_odometry(self):
        m = tray.measure_tray([(h1_like(), [])] * 3, CAM, self.cfg, HALF, RIM, expected=(0.44, -0.01))
        self.assertEqual((m['source'], m['used'], m['centre']), ('odometry', 0, [0.44, -0.01]))

    def test_median_over_frames(self):
        st = h1_like()
        boxes = [tray_box(st, (f, 0.0)) for f in (0.44, 0.45, 0.46, 0.45, 0.60)]
        m = tray.measure_tray([(st, [b]) for b in boxes], CAM, self.cfg, HALF, RIM, expected=(0.45, 0.0))
        self.assertAlmostEqual(m['centre'][0], 0.45, places=4)


class DropWalkTests(unittest.TestCase):
    def test_walk_until_d(self):
        w = tray.drop_walk((0.45, 0.0), 0.21)
        self.assertAlmostEqual(w['distance'], 0.24)
        self.assertAlmostEqual(w['bearing'], 0.0)

    def test_off_line_turns_first(self):
        w = tray.drop_walk((0.4, 0.3), 0.21)
        self.assertAlmostEqual(w['bearing'], math.atan2(0.3, 0.4))
        self.assertAlmostEqual(w['distance'], 0.29)

    def test_already_closer_than_d_does_not_walk_back(self):
        self.assertEqual(tray.drop_walk((0.15, 0.0), 0.21)['distance'], 0.0)

    def test_sweep_window(self):
        """3P drop geometry: every d in the sweep keeps the toes off the rim and the toy in."""
        m = STAGE3['measured_3p']
        toe = m['reach_toe_m']['H0']
        for d in STAGE3['drop']['d_sweep_m']:
            self.assertGreater(d - toe, HALF, d)                         # toes outside the near rim
            for toy, fwd in m['held_toy_forward_m'].items():
                self.assertLess(d - fwd['H0'], STAGE3['episode']['delivered_half_m'], (d, toy))


def tof_frame(state, hit):
    """A ToF frame: each beam's distance to the first surface `hit(origin, direction)` returns."""
    pose = state['frames']['tof']
    level = ranging.level_matrix(state)
    rot = level @ ranging.quat_matrix(pose['quat'])
    origin = level @ np.asarray(pose['pos'], float) + np.array([0.0, 0.0, state['odom']['position'][2]])
    dist, status = [], []
    for beam in tof_lib.BEAMS:
        d = hit(origin, rot @ beam)
        dist.append(int(round(d * 1000)) if d else 0)
        status.append(tof_lib.STATUS_VALID if d else 255)
    return {'distance_mm': dist, 'status': status}


def wall(x):
    return lambda o, v: (x - o[0]) / v[0] if v[0] > 1e-6 and (x - o[0]) / v[0] < 4.0 else None


def floor(o, v):
    return -o[2] / v[2] if v[2] < -1e-6 else None


class TofTests(unittest.TestCase):
    cfg = STAGE3['tof']

    def test_zone_layout_matches_tofrs(self):
        b = tof_lib.BEAMS
        self.assertEqual(b.shape, (64, 3))
        self.assertGreater(b[0][2], 0)          # row 0 = top
        self.assertGreater(b[0][1], 0)          # col 0 = the sensor's left
        self.assertLess(b[63][2], 0)
        self.assertLess(b[63][1], 0)
        self.assertAlmostEqual(math.degrees(math.atan2(b[0][1], b[0][0])), 22.5 - 45 / 16, places=6)

    def test_wall_inside_stop_range_stops(self):
        st = h1_like()
        hits = tof_lib.obstacles(tof_frame(st, wall(self.cfg['stop_range_m'] - 0.03)), st, self.cfg)
        self.assertTrue(hits)
        self.assertTrue(all(h['range_m'] <= self.cfg['stop_range_m'] for h in hits))

    def test_wall_beyond_stop_range_does_not(self):
        st = h1_like()
        self.assertEqual(tof_lib.obstacles(tof_frame(st, wall(self.cfg['stop_range_m'] + 0.03)), st,
                                           self.cfg), [])

    def test_floor_never_stops(self):
        st = h1_like(tof_pitch_deg=40.0)        # looking down at the floor close ahead
        pts = tof_lib.points(tof_frame(st, floor), st)
        self.assertTrue(pts)
        self.assertTrue(all(abs(p['z']) < 1e-3 for p in pts))
        self.assertEqual(tof_lib.obstacles(tof_frame(st, floor), st, self.cfg), [])

    def test_nothing_while_falling(self):
        st = h1_like()
        st['imu']['quat'] = axis_quat((1, 0, 0), math.radians(35.0))
        self.assertGreater(tof_lib.tilt_deg(st), self.cfg['max_tilt_deg'])
        frame = tof_frame(st, wall(self.cfg['stop_range_m'] - 0.03))
        self.assertEqual(tof_lib.obstacles(frame, st, self.cfg), [])

    def test_returns_behind_the_trunk_are_ignored(self):
        st = h1_like()
        behind = [{'zone': 0, 'slant_m': 0.12, 'x': -0.1, 'y': 0.0, 'z': 0.2, 'range_m': 0.1}]
        with mock.patch.object(tof_lib, 'points', return_value=behind):
            self.assertEqual(tof_lib.obstacles({}, st, self.cfg), [])

    def test_own_beak_is_ignored(self):
        st = h1_like()
        near = self.cfg['self_blind_m'] - 0.02
        beak = lambda o, v: near     # noqa: E731  every zone returns nearer than self_blind_m
        self.assertEqual(tof_lib.obstacles(tof_frame(st, beak), st, self.cfg), [])


if __name__ == '__main__':
    unittest.main()
