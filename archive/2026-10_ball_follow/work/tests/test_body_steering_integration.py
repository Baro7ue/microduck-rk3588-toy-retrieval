"""Body-referenced steering with real local UNIX/TCP peers and fixture inference only."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
import test_follow_ball_integration as fixtures
from test_camera_motion_integration import PoseBody, PoseCamera
from test_follow_ball_npu import FixtureNPUDetector, follow_npu


FOCAL_PX = 434.55844122715706
DEPRESSION = math.radians(25)


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def head_angle(index):
    return math.radians(12 if index % 2 else -12)


def camera_quat(index):
    # CV right/down/forward -> body -y/-z/+x, then depress25deg and pan±12deg.
    h = head_angle(index)
    return multiply(multiply([math.cos(h/2), 0, 0, math.sin(h/2)],
                             [math.cos(DEPRESSION/2), 0, math.sin(DEPRESSION/2), 0]),
                    [.5, -.5, .5, -.5])


def projected_center(index, target_bearing):
    # Independent dot products onto the camera's right/down/forward basis.
    # The world target is [cos(beta),sin(beta),-tan(depression)].
    delta = head_angle(index)-target_bearing
    denominator = (math.cos(DEPRESSION)*math.cos(delta) +
                   math.sin(DEPRESSION)*math.tan(DEPRESSION))
    x = math.sin(delta)
    y = math.sin(DEPRESSION)*(1-math.cos(delta))
    return 180+FOCAL_PX*x/denominator, 320+FOCAL_PX*y/denominator


class SteeringBody(PoseBody):
    def publish_camera_pose(self, index, mode):
        with self.state_lock:
            self.state['t_ns'] = time.monotonic_ns()
            self.state['frames'] = {'camera': {'quat': camera_quat(index)}}
            if mode == 'unavailable':
                self.state['imu'] = {}
            else:
                self.state['imu'] = {'quat': [0.0]*4 if mode == 'invalid' else [1., 0., 0., 0.]}
            if self.subscription is not None:
                self.send_state(copy.deepcopy(self.state))


class SteeringDetector(FixtureNPUDetector):
    def __init__(self, target_bearing=0):
        super().__init__()
        self.target_bearing = target_bearing

    def process(self, payload):
        index = payload[1]
        self.processed.append(index)
        width, height = ((40, 40) if index <= 6 else (50, 50) if index == 7 else
                         (60, 70) if index == 8 else (70, 100) if index == 9 else
                         (80, 140) if index == 10 else (80, 180))
        u, v = projected_center(index, self.target_bearing)
        return {'detections': [{
            'class_id': 32, 'class_name': 'sports ball', 'confidence': .8,
            'xyxy': [u-width/2, v-height/2, u+width/2, v+height/2],
        }], 'inference_call_ms': .01, 'complete_processing_ms': .01}, np.zeros(
            (640, 360, 3), dtype=np.uint8)


class BodySteeringIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='body-steer-e2e-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = SteeringBody(self.root/'body.sock')
        self.addCleanup(self.body.close)

    def trial(self, *, target_bearing=0, mode='valid'):
        camera = PoseCamera(self.body, mode)
        self.addCleanup(camera.close)
        detector = SteeringDetector(target_bearing)
        directory = self.root/'trial'
        with patch.object(follow_npu, 'NPUCameraDetector', return_value=detector), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = follow_npu.run_trial(self.root/'unused.rknn', camera.address,
                self.root/'body.sock', directory, duration=3, speed=.3, stop_area=.06,
                orange_continuation=True, camera_motion_compensation=True, body_steering=True)
        return status, json.loads((directory/'summary.json').read_text()), detector, directory

    def assert_retained_stop(self, report, detector):
        self.assertTrue(report['body_steering'])
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertEqual(detector.release_count, 1)

    def test_straight_body_target_stays_zero_yaw_under_alternating_head_pan_then_stops_at_area(self):
        status, report, detector, directory = self.trial()
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assert_retained_stop(report, detector)
        predictions = fixtures.rows(directory/'detections.jsonl')
        following = [row for row in predictions if row['control']['mode'] == 'following']
        self.assertTrue(following)
        self.assertTrue(any(row['control']['image_error'] > .3 for row in following))
        self.assertTrue(any(row['control']['image_error'] < -.3 for row in following))
        for row in following:
            self.assertEqual(row['control']['vx'], .3)
            self.assertEqual(row['control']['vyaw'], 0)
            self.assertAlmostEqual(row['control']['e'], 0)
            self.assertAlmostEqual(row['control']['body_bearing_rad'], 0)
            self.assertEqual(row['camera_pose']['trunk_quat'], [1., 0., 0., 0.])
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(any(row['vx'] == .3 for row in commands))
        self.assertTrue(all(row['vyaw'] == 0 for row in commands))
        reached = next(row for row in predictions if row['control']['reason'] == 'target_area_reached')
        self.assertGreaterEqual(reached['receiver_index'], 11)
        self.assertAlmostEqual(reached['control']['area'], .0625)

    def test_left_body_target_requests_positive_yaw_regardless_of_head_pan(self):
        bearing = math.radians(4)
        status, report, detector, directory = self.trial(target_bearing=bearing)
        self.assertEqual(status, 0)
        self.assert_retained_stop(report, detector)
        predictions = fixtures.rows(directory/'detections.jsonl')
        following = [row for row in predictions if row['control']['mode'] == 'following']
        self.assertTrue(following)
        for row in following:
            self.assertGreater(row['control']['vyaw'], 0)
            self.assertLessEqual(row['control']['vyaw'], .2)
            self.assertAlmostEqual(row['control']['body_bearing_rad'], bearing)
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(any(row['vx'] == .3 and row['vyaw'] > 0 for row in commands))
        self.assertTrue(all(row['vyaw'] >= 0 for row in commands))

    def test_missing_paired_pose_holds_all_zero_then_retains_confirmed_stand(self):
        status, report, detector, directory = self.trial(mode='unavailable')
        self.assertEqual(status, 2)
        self.assertEqual(report['stop_reason'], 'camera_pose_unavailable')
        self.assert_retained_stop(report, detector)
        predictions = fixtures.rows(directory/'detections.jsonl')
        self.assertTrue(predictions)
        self.assertEqual(predictions[0]['control']['mode'], 'holding')
        self.assertTrue(all(row['camera_pose'] is None for row in predictions))
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(all([row[key] for key in ('vx', 'vy', 'vyaw')] == [0, 0, 0] for row in commands))

    def test_invalid_paired_pose_stops_all_zero_without_exception_disable(self):
        status, report, detector, directory = self.trial(mode='invalid')
        self.assertEqual(status, 2)
        self.assertEqual(report['stop_reason'], 'invalid_camera_pose')
        self.assert_retained_stop(report, detector)
        predictions = fixtures.rows(directory/'detections.jsonl')
        self.assertTrue(any(row['camera_pose'].get('error') for row in predictions))
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(all([row[key] for key in ('vx', 'vy', 'vyaw')] == [0, 0, 0] for row in commands))
        self.assertNotIn('error', report)


if __name__ == '__main__':
    unittest.main()
