"""Association-only camera compensation through actual local body/camera transports."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
import test_follow_ball_integration as fixtures
from test_follow_ball_npu import FixtureNPUDetector, follow_npu


FOCAL_PX = 434.55844122715706
PAN_RAD = -math.atan(120 / FOCAL_PX)
TURNED_QUAT = [math.cos(PAN_RAD / 2), 0.0, math.sin(PAN_RAD / 2), 0.0]


class PoseBody(fixtures.AdvancingBody):
    """Advancing real notifications; serialize fixture ticks to prevent artificial reversal."""
    def advance(self):
        while not self.stopped.wait(.025):
            with self.state_lock:
                self.state['t_ns'] = time.monotonic_ns()
                if self.subscription is not None:
                    try:
                        self.send_state(copy.deepcopy(self.state))
                    except OSError:
                        return

    def publish_camera_pose(self, index, mode):
        with self.state_lock:
            self.state['t_ns'] = time.monotonic_ns()
            if mode == 'unavailable':
                self.state.pop('imu', None)
                self.state.pop('frames', None)
            else:
                self.state['imu'] = {'quat': [1.0, 0.0, 0.0, 0.0]}
                quat = ([0.0] * 4 if mode == 'invalid' else
                        [1.0, 0.0, 0.0, 0.0] if index <= 5 else TURNED_QUAT)
                self.state['frames'] = {'camera': {'quat': quat.copy()}}
            if self.subscription is not None:
                self.send_state(copy.deepcopy(self.state))


class PoseCamera(fixtures.LoopbackCamera):
    def __init__(self, body, mode):
        self.body, self.mode = body, mode
        super().__init__()

    def run(self):
        try:
            while not self.stopped.is_set():
                try:
                    self.client, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if self.client is None:
                return
            with self.client:
                while not self.stopped.is_set():
                    index = self.sent + 1
                    self.body.publish_camera_pose(index, self.mode)
                    # The pose genuinely arrives before its complete camera payload.
                    # This is not an inference stub supplying pose metadata to the controller.
                    if self.stopped.wait(.02):
                        return
                    payload = bytes((128, index % 250, 128, index % 250)) * 115_200
                    self.client.sendall(struct.pack('<I', len(payload)) + payload)
                    self.sent = index
                    self.stopped.wait(.08)
        except OSError:
            pass


class PanDetector(FixtureNPUDetector):
    """Substitute neural output only, with a hand-derived compensated target center."""
    def process(self, payload):
        index = payload[1]
        self.processed.append(index)
        if index <= 6:
            width, height = 40, 40
        elif index == 7:
            width, height = 50, 50
        elif index == 8:
            width, height = 60, 70
        elif index == 9:
            width, height = 70, 100
        elif index == 10:
            width, height = 80, 140
        else:
            width, height = 80, 180
        center = 180 if index <= 5 else 300
        class_id = 32 if index <= 5 else 49
        prediction = {'detections': [{
            'class_id': class_id, 'class_name': 'sports ball' if class_id == 32 else 'orange',
            'confidence': .8,
            'xyxy': [center-width/2, 320-height/2, center+width/2, 320+height/2],
        }], 'inference_call_ms': .01, 'complete_processing_ms': .01}
        return prediction, np.zeros((640, 360, 3), dtype=np.uint8)


class CameraMotionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='camera-pan-e2e-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = PoseBody(self.root/'body.sock')
        self.addCleanup(self.body.close)

    def trial(self, mode='valid'):
        camera = PoseCamera(self.body, mode)
        self.addCleanup(camera.close)
        detector = PanDetector()
        directory = self.root/'trial'
        with patch.object(follow_npu, 'NPUCameraDetector', return_value=detector), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = follow_npu.run_trial(self.root/'unused.rknn', camera.address,
                self.root/'body.sock', directory, duration=3, speed=.3, stop_area=.06,
                orange_continuation=True, camera_motion_compensation=True)
        report = json.loads((directory/'summary.json').read_text())
        return status, report, detector, directory

    def assert_retained_stop(self, report, detector):
        self.assertTrue(report['camera_motion_compensation'])
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertEqual(detector.release_count, 1)

    def test_world_pose_compensates_orange_association_but_steers_from_raw_image(self):
        # q_current^-1 rotates the old forward ray toward x=120/f: u=180+120=300.
        # The raw displacement120px exceeds1.5*40=60px, so ordinary matching cannot pass.
        self.assertAlmostEqual(180 - FOCAL_PX * math.tan(PAN_RAD), 300)
        status, report, detector, directory = self.trial()
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assert_retained_stop(report, detector)
        predictions = fixtures.rows(directory/'detections.jsonl')
        turned = next(row for row in predictions if row['receiver_index'] == 6)
        self.assertEqual(turned['detections'][0]['class_id'], 49)
        self.assertEqual(turned['detections'][0]['xyxy'], [280.0, 300.0, 320.0, 340.0])
        self.assertEqual(turned['control']['association'], 'camera_motion_continuation')
        self.assertEqual(turned['control']['source_class_id'], 49)
        self.assertEqual(turned['control']['vx'], .3)
        self.assertEqual(turned['control']['vyaw'], -.2)
        for actual, expected in zip(turned['camera_pose']['quat'], TURNED_QUAT):
            self.assertAlmostEqual(actual, expected)
        self.assertLessEqual(abs(turned['camera_pose']['sensor_to_frame_s']), .1)
        self.assertLessEqual(abs(turned['camera_pose']['receipt_to_frame_s']), .1)
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(any(row['receiver_index'] == 6 and row['vx'] == .3 and row['vyaw'] == -.2
                            for row in commands))
        reached = next(row for row in predictions if row['control']['reason'] == 'target_area_reached')
        self.assertGreaterEqual(reached['receiver_index'], 11)
        self.assertAlmostEqual(reached['control']['area'], .0625)

    def test_unavailable_pose_never_moves_and_is_a_controlled_standing_stop(self):
        status, report, detector, directory = self.trial('unavailable')
        self.assertEqual(status, 2)
        self.assertEqual(report['stop_reason'], 'camera_pose_unavailable')
        self.assert_retained_stop(report, detector)
        predictions = fixtures.rows(directory/'detections.jsonl')
        self.assertTrue(predictions)
        self.assertTrue(all(row['camera_pose'] is None for row in predictions))
        self.assertTrue(all(row['control']['vx'] == row['control']['vyaw'] == 0 for row in predictions))
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(all([row[key] for key in ('vx', 'vy', 'vyaw')] == [0, 0, 0] for row in commands))

    def test_invalid_pose_never_moves_and_does_not_trigger_exception_disable(self):
        status, report, detector, directory = self.trial('invalid')
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
