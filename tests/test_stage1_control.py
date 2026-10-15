"""Stage 1 control code without a simulator: colour boxes, ranging, the pick timing, truth reading."""
from __future__ import annotations

import colorsys
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from microduck_toy.control import pick as pick_lib
from microduck_toy.control import ranging
from microduck_toy.control.colour import ColourDetector, largest_blob, rgb_to_hsv
from microduck_toy.stage1 import metrics

REPO = Path(__file__).resolve().parents[1]
STAGE1 = json.loads((REPO / 'configs/control/stage1.json').read_text())


def rgb(h_cv: float, s: float, v: float) -> list[int]:
    """OpenCV-range HSV (H 0-180, S and V 0-255) to an RGB pixel."""
    r, g, b = colorsys.hsv_to_rgb(h_cv / 180.0, s / 255.0, v / 255.0)
    return [round(255 * r), round(255 * g), round(255 * b)]


class ColourTests(unittest.TestCase):
    def setUp(self):
        self.det = ColourDetector(STAGE1['colour'])
        self.img = np.zeros((640, 360, 3), np.uint8)
        self.img[:] = rgb(22, 90, 180)                 # light wood: bus hue, too grey for the bus

    def test_hsv_matches_colorsys(self):
        px = np.array([[rgb(25, 200, 220), rgb(60, 150, 120), rgb(176, 220, 200)]], np.uint8)
        hsv = rgb_to_hsv(px)[0]
        for got, want in zip(hsv, [(25, 200, 220), (60, 150, 120), (176, 220, 200)]):
            self.assertTrue(np.allclose(got, want, atol=1.5), (got, want))

    def test_bus_box_ignores_the_wood_and_a_stray_pixel(self):
        self.img[300:340, 150:230] = rgb(25, 200, 220)
        self.img[50, 50] = rgb(25, 200, 220)
        (box,) = self.det.detect(self.img, ['bus'])
        self.assertEqual(box['name'], 'bus')
        self.assertEqual(box['xyxy'], [150, 300, 230, 340])
        self.assertFalse(box['cut'])
        self.assertEqual(self.det.detect(self.img, ['henry', 'ketchup']), [])

    def test_red_wraps_round_the_hue_circle(self):
        self.img[600:640, 10:40] = rgb(178, 200, 180)
        self.img[600:640, 40:70] = rgb(3, 200, 180)
        (box,) = self.det.detect(self.img, ['ketchup'])
        self.assertEqual(box['xyxy'], [10, 600, 70, 640])
        self.assertTrue(box['cut'])                    # touches the bottom edge

    def test_largest_blob(self):
        mask = np.zeros((10, 10), bool)
        mask[0:2, 0:2] = True
        mask[5:9, 5:8] = True
        blob = largest_blob(mask)
        self.assertEqual(int(blob.sum()), 12)
        self.assertIsNone(largest_blob(np.zeros((3, 3), bool)))


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def axis_quat(axis, angle):
    s = math.sin(angle / 2)
    return [math.cos(angle / 2), *(s * a for a in axis)]


FORWARD = [.5, -.5, .5, -.5]    # OpenCV right/down/forward -> trunk -y/-z/+x


class RangingTests(unittest.TestCase):
    cam = ranging.CameraModel.from_config(STAGE1['camera'])

    def state(self, pitch_deg=50.0, roll_deg=2.0, tilt_deg=3.0, yaw_deg=40.0):
        camera = quat_mul(axis_quat((0, 1, 0), math.radians(pitch_deg)), FORWARD)
        imu = quat_mul(axis_quat((0, 0, 1), math.radians(yaw_deg)),
                       quat_mul(axis_quat((0, 1, 0), math.radians(tilt_deg)),
                                axis_quat((1, 0, 0), math.radians(roll_deg))))
        return {'frames': {'camera': {'pos': [0.06, 0.0, 0.09], 'quat': camera}},
                'imu': {'quat': imu}, 'odom': {'position': [1.0, 2.0, 0.117], 'yaw': 0.7},
                'safety': {'gravity': [0.0, 0.0, -1.0]}}

    def test_camera_model_from_config(self):
        self.assertEqual((self.cam.width, self.cam.height), (360, 640))
        self.assertAlmostEqual(self.cam.focal_px, 180 / math.tan(math.radians(22.5)), places=6)

    def test_ground_point_inverts_project(self):
        state = self.state()
        for forward, left, z in ((0.08, 0.015, 0.012), (0.15, -0.03, 0.01), (0.4, 0.1, 0.015)):
            col, row = ranging.project(self.cam, state, (forward, left, z))
            self.assertTrue(0 <= col <= 360 and 0 <= row <= 640, (forward, left, col, row))
            got = ranging.ground_point(self.cam, state, (col, row, col, row), z)
            self.assertAlmostEqual(got[0], forward, places=9)
            self.assertAlmostEqual(got[1], left, places=9)

    def test_heading_does_not_matter(self):
        a = ranging.ground_point(self.cam, self.state(yaw_deg=0), (180, 500, 180, 500), 0.01)
        b = ranging.ground_point(self.cam, self.state(yaw_deg=123), (180, 500, 180, 500), 0.01)
        self.assertAlmostEqual(a[0], b[0], places=9)
        self.assertAlmostEqual(a[1], b[1], places=9)

    def test_gravity_fallback_matches_a_pure_tilt(self):
        state = self.state(roll_deg=0.0, tilt_deg=4.0, yaw_deg=0.0)
        rot = ranging.quat_matrix(state['imu']['quat'])
        no_imu = dict(state, imu={}, safety={'gravity': list(rot.T @ [0.0, 0.0, -1.0])})
        a = ranging.ground_point(self.cam, state, (100, 450, 120, 470), 0.01)
        b = ranging.ground_point(self.cam, no_imu, (100, 450, 120, 470), 0.01)
        self.assertAlmostEqual(a[0], b[0], places=6)
        self.assertAlmostEqual(a[1], b[1], places=6)

    def test_ray_above_the_horizon_has_no_ground_point(self):
        self.assertIsNone(ranging.ground_point(self.cam, self.state(pitch_deg=-30), (180, 0, 180, 0),
                                               0.01))


class FakeRobot:
    """robot.state at 50 Hz: stand until the ground_pick request, GroundPick for 0.7 of a 4 s
    cycle from the next tick, then stand. The mouth joint reads `held_deg` once closed."""

    def __init__(self, held_deg=3.0, start_delay=0.06):
        self.t, self.started, self.start_delay = 10.0, None, start_delay
        self.mouth_log, self.held_deg = [], held_deg
        self.closed = False

    def policy(self):
        if self.started is not None and self.started <= self.t < self.started + 0.7 * 4.0:
            return 'ground_pick'
        return 'stand'

    def state(self):
        mouth = math.radians(self.held_deg if self.closed else 30.0)
        joints = [0.0] * 15
        joints[pick_lib.MOUTH_INDEX] = mouth
        return {'t': round(self.t, 4), 'policy': self.policy(), 'joints': joints,
                'safety': {'fallen': False}}

    def mouth(self, fraction):
        self.mouth_log.append((round(self.t, 4), fraction))
        self.closed = fraction < 0.5

    def request(self, method, params):
        self.started = self.t + self.start_delay
        return {'ok': True}

    def wait(self, predicate, timeout, what):
        end = self.t + timeout
        while self.t <= end:
            self.t += 0.02
            state = self.state()
            if predicate(state):
                return state
        raise TimeoutError(what)

    def snapshot(self):
        self.t += 1.0                           # confirm_s went by
        return self.state()


class PickTests(unittest.TestCase):
    def run_pick(self, robot, phi=0.33):
        with mock.patch.object(pick_lib.time, 'sleep'):
            return pick_lib.ground_pick(robot, phi, period_s=4.0, held_threshold_deg=-1.5)

    def test_closes_at_the_phase_and_reads_held(self):
        robot = FakeRobot(held_deg=3.0)
        out = self.run_pick(robot)
        self.assertTrue(out['ran'])
        close_t = [t for t, f in robot.mouth_log if f == 0.0][0]
        self.assertAlmostEqual(close_t, out['start_t'] + 0.33 * 4.0, delta=0.021)
        self.assertAlmostEqual(out['close_phase'], 0.33, delta=0.006)
        self.assertAlmostEqual(out['handback_phase'], 0.70, delta=0.006)
        self.assertTrue(out['held'])
        self.assertEqual(robot.mouth_log[0][1], 1.0)          # opened first

    def test_empty_beak_reads_not_held(self):
        self.assertFalse(self.run_pick(FakeRobot(held_deg=-5.0))['held'])

    def test_no_ground_pick_closes_the_mouth_and_says_so(self):
        robot = FakeRobot(start_delay=99.0)
        out = self.run_pick(robot)
        self.assertFalse(out['ran'])
        self.assertIn('ground_pick start', out['error'])
        self.assertEqual(robot.mouth_log[-1][1], 0.0)


class TruthTests(unittest.TestCase):
    def test_read_truth_types_and_at_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'truth.csv'
            with path.open('w', newline='') as handle:
                w = csv.writer(handle)
                w.writerow(['t', 'mono', 'x', 'lfoot', 'mark', 'holding', 'bus_x'])
                w.writerow([0.0, 1.00, 0.10, 1, '', '', 0.5])
                w.writerow([0.02, 1.02, 0.20, 1, 'a', 'bus', 0.6])
                w.writerow([0.04, 1.04, 0.00, 0, '', '', 0.7])      # the next reset's teleport
            rows = metrics.read_truth(path)
        self.assertEqual(rows[1]['holding'], 'bus')
        self.assertEqual(rows[0]['mark'], '')
        self.assertIsInstance(rows[0]['lfoot'], int)
        self.assertEqual(rows[1]['bus_x'], 0.6)
        self.assertEqual(metrics.at_mark(rows, 1.03)['x'], 0.20)     # not the teleported row
        self.assertEqual(metrics.at(rows, 1.03)['x'], 0.00)
        self.assertEqual(metrics.at_mark(rows, 0.5)['x'], 0.10)      # before the first row


if __name__ == '__main__':
    unittest.main()
