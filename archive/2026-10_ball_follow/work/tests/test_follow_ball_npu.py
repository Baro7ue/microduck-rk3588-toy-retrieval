"""NPU follower integration uses local peers and substitutes neural inference only."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
import follow_ball_cpu
import test_follow_ball_integration as fixtures

SOURCE = WORK / 'follow_ball_npu.py'
follow_npu = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('board_follow_ball_npu', SOURCE)
    follow_npu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(follow_npu)

RKNN_SHA = '72b202f6b6cebb12f60bd1ae2dceffa706ba19cc66c93955e483645ef8c26a02'


class FixtureNPUDetector(fixtures.FixtureDetector):
    """Reuse the CPU fixture's box schedule; model identity and runtime are NPU-specific."""
    def __init__(self, mode='triple_area', wrong_model=False, on_close=lambda: None):
        super().__init__(mode)
        self.model = {'sha256': fixtures.MODEL_SHA if wrong_model else RKNN_SHA}
        self.runtime_info = {'backend': 'npu', 'runtime': 'RKNNLite',
                             'version': '2.3.2', 'sdk_version': 'fixture-only'}
        self.release_count = 0
        self.on_close = on_close
        self.close_snapshot = None

    def process(self, payload):
        if self.mode == 'inference_error' and payload[1] >= 8:
            raise RuntimeError('fixture NPU inference failed after movement')
        prediction, image = super().process(payload)
        prediction['inference_call_ms'] = prediction.pop('session_run_ms')
        return prediction, image

    def close(self):
        self.release_count += 1
        self.close_snapshot = self.on_close()


class FollowBallNPUTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(follow_npu, 'the NPU follower entry point is missing')
        self.temp = tempfile.TemporaryDirectory(prefix='ball-npu-e2e-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = fixtures.AdvancingBody(self.root / 'body.sock')
        self.addCleanup(self.body.close)
        self.camera = fixtures.LoopbackCamera()
        self.addCleanup(self.camera.close)
        self.links = []
        self.real_link = follow_ball_cpu.RobotLink

    def robot_factory(self, *args, **kwargs):
        link = self.real_link(*args, **kwargs)
        self.links.append(link)
        return link

    def closed_transport_snapshot(self):
        if not self.links:
            return None
        link = self.links[-1]
        return {'sender_alive': any(thread.name == 'ball-drive' and thread.is_alive()
                                    for thread in threading.enumerate()),
                'state_socket_closed': link._state_socket is None or link._state_socket.fileno() == -1,
                'move_socket_closed': link._move_socket is None,
                'state_reader_alive': link._thread is not None and link._thread.is_alive()}

    def trial(self, detector=None, factory_error=None, stop_area=.012):
        directory = self.root / 'trial'
        constructor = patch.object(follow_npu, 'NPUCameraDetector',
                                   side_effect=factory_error, return_value=detector)
        with constructor, patch.object(follow_ball_cpu, 'RobotLink', self.robot_factory), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = follow_npu.run_trial(self.root / 'unused.rknn', self.camera.address,
                                         self.root / 'body.sock', directory, duration=3,
                                         speed=.3, stop_area=stop_area)
        report = json.loads((directory / 'summary.json').read_text())
        return status, report, directory

    def assert_release_after_control_transport_closed(self, detector):
        self.assertEqual(detector.release_count, 1)
        self.assertEqual(detector.close_snapshot,
                         {'sender_alive': False, 'state_socket_closed': True,
                          'move_socket_closed': True, 'state_reader_alive': False})

    def test_fixed_area_at_point_three_stops_confirms_stand_then_releases(self):
        # CPU hash validation or premature release breaks the NPU path despite real safe control.
        detector = FixtureNPUDetector(on_close=self.closed_transport_snapshot)
        status, report, directory = self.trial(detector, stop_area=1/48)
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertIn('NPU', report['scope'])
        self.assertEqual(report['runtime']['runtime'], 'RKNNLite')
        self.assertEqual(report['model']['sha256'], RKNN_SHA)
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertTrue(report['walk_observed'])
        self.assertEqual(report['stop_area_fraction'], 1/48)
        self.assertNotIn('area_stop_ratio', report)
        self.assertEqual(report['speed_command_m_s'], .3)
        predictions = fixtures.rows(directory / 'detections.jsonl')
        below_target = [row for row in predictions if 11 <= row['receiver_index'] < 21]
        self.assertTrue(below_target)
        self.assertTrue(all(row['control']['vx'] == .3 and row['control']['reason'] is None
                            for row in below_target))
        reached = next(row for row in predictions if row['control']['reason'] == 'target_area_reached')
        self.assertGreaterEqual(reached['receiver_index'], 21)
        self.assertAlmostEqual(reached['control']['area'], 1 / 48)
        self.assertTrue(all('inference_call_ms' in row and 'session_run_ms' not in row
                            for row in predictions))
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertTrue(any(event['message']['method'] == 'robot.move' and
                            event['message']['params']['vx'] == .3 for event in self.body.events))
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['requested'], [0, 0, 0])
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))
        self.assert_release_after_control_transport_closed(detector)

    def test_first_frame_already_at_fixed_area_never_walks_and_confirms_stand(self):
        # A nearby starting ball is already a valid success; startup must never send forward motion.
        detector = FixtureNPUDetector(mode='already_near', on_close=self.closed_transport_snapshot)
        status, report, directory = self.trial(detector, stop_area=.012)
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_already_at_target_returned_stand')
        self.assertEqual(report['stop_area_fraction'], .012)
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertFalse(report['walk_observed'])
        predictions = fixtures.rows(directory / 'detections.jsonl')
        self.assertTrue(predictions)
        self.assertGreaterEqual(predictions[0]['control']['area'], .012)
        commands = fixtures.rows(directory / 'commands.jsonl')
        self.assertTrue(commands)
        self.assertTrue(all([row[key] for key in ('vx', 'vy', 'vyaw')] == [0, 0, 0]
                            for row in commands))
        moves = [event['message']['params'] for event in self.body.events
                 if event['message']['method'] == 'robot.move']
        self.assertTrue(moves)
        self.assertTrue(all(move == {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0} for move in moves))
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['requested'], [0, 0, 0])
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))
        self.assert_release_after_control_transport_closed(detector)

    def test_npu_inference_failure_orders_zero_stop_disable_then_releases(self):
        # An exception after walking starts must revoke success and retain the control cleanup.
        detector = FixtureNPUDetector(mode='inference_error', on_close=self.closed_transport_snapshot)
        status, report, directory = self.trial(detector)
        self.assertEqual(status, 1)
        self.assertEqual(report['outcome'], 'failed')
        self.assertFalse(report['left_enabled_standing'])
        commands = fixtures.rows(directory / 'commands.jsonl')
        self.assertTrue(any(row['vx'] == .3 for row in commands))
        self.assertEqual([commands[-1][key] for key in ('vx', 'vy', 'vyaw')], [0, 0, 0])
        stop = fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        disable = next(event for event in self.body.events
                       if event['message']['method'] == 'robot.enable')
        self.assertFalse(disable['message']['params']['on'])
        self.assertGreater(disable['received_at'], stop['received_at'])
        self.assertTrue(report['robot.enable_cleanup']['accepted'])
        self.assertIn('RuntimeError: fixture NPU inference failed after movement',
                      (directory / 'error.txt').read_text())
        self.assert_release_after_control_transport_closed(detector)

    def test_lost_ball_latches_zero_and_keeps_confirmed_standing_enabled(self):
        # A controlled incomplete trial must keep the confirmed balancing policy without resuming.
        detector = FixtureNPUDetector(mode='loss', on_close=self.closed_transport_snapshot)
        status, report, directory = self.trial(detector)
        self.assertEqual(status, 2)
        self.assertEqual(report['outcome'], 'stopped_missing_ball')
        self.assertEqual(report['stop_reason'], 'missing_ball')
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertIn(8, detector.processed)
        self.assertGreaterEqual(self.camera.sent, 11)
        commands = fixtures.rows(directory / 'commands.jsonl')
        self.assertTrue(any(row['vx'] == .3 for row in commands))
        lost = next(row for row in fixtures.rows(directory / 'detections.jsonl')
                    if not row['detections'])
        self.assertTrue(all([row[key] for key in ('vx', 'vy', 'vyaw')] == [0, 0, 0]
                            for row in commands
                            if row['sent_monotonic_s'] >= lost['processed_monotonic_s']))
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['requested'], [0, 0, 0])
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertNotIn('robot.enable_cleanup', report)
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))
        self.assert_release_after_control_transport_closed(detector)

    def test_standing_confirmation_failure_still_disables_and_releases(self):
        # Preserving balance is allowed only after a real confirmed stop, never on confirmation failure.
        detector = FixtureNPUDetector(mode='success', on_close=self.closed_transport_snapshot)
        with patch.object(follow_ball_cpu, 'confirm_stand',
                          side_effect=TimeoutError('fixture standing confirmation failed')):
            status, report, directory = self.trial(detector)
        self.assertEqual(status, 1)
        self.assertEqual(report['outcome'], 'failed')
        self.assertFalse(report['stop_confirmed'])
        self.assertFalse(report['left_enabled_standing'])
        stop = fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        disable = next(event for event in self.body.events
                       if event['message']['method'] == 'robot.enable')
        self.assertFalse(disable['message']['params']['on'])
        self.assertGreater(disable['received_at'], stop['received_at'])
        self.assertTrue(report['robot.enable_cleanup']['accepted'])
        self.assertIn('TimeoutError: fixture standing confirmation failed',
                      (directory / 'error.txt').read_text())
        self.assert_release_after_control_transport_closed(detector)

    def test_final_output_failure_revokes_saved_success_and_disables(self):
        # A broken final output pipe must not leave a saved success after cleanup disables balance.
        detector = FixtureNPUDetector(mode='success', on_close=self.closed_transport_snapshot)
        original_print = print
        failed_once = False

        def fail_first_result(*args, **kwargs):
            nonlocal failed_once
            if args and isinstance(args[0], str) and args[0].startswith('结果：') and not failed_once:
                failed_once = True
                raise BrokenPipeError('fixture final output pipe closed')
            return original_print(*args, **kwargs)

        with patch('builtins.print', side_effect=fail_first_result):
            status, report, _ = self.trial(detector)
        self.assertTrue(failed_once)
        self.assertEqual(status, 1)
        stop = fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        disable = next(event for event in self.body.events
                       if event['message']['method'] == 'robot.enable')
        self.assertFalse(disable['message']['params']['on'])
        self.assertGreater(disable['received_at'], stop['received_at'])
        self.assertEqual(report['outcome'], 'failed')
        self.assertFalse(report['left_enabled_standing'])
        self.assertTrue(report['robot.enable_cleanup']['accepted'])
        self.assertIn('BrokenPipeError: fixture final output pipe closed', report['final_output_error'])
        self.assert_release_after_control_transport_closed(detector)

    def test_wrong_rknn_hash_is_rejected_before_any_robot_or_camera_connection(self):
        # An ONNX CPU hash must not authorize the NPU follower, even with valid ball detections.
        detector = FixtureNPUDetector(wrong_model=True, on_close=self.closed_transport_snapshot)
        status, report, _ = self.trial(detector)
        self.assertEqual(status, 1)
        self.assertEqual(report['outcome'], 'failed')
        self.assertEqual(self.links, [])
        self.assertEqual(self.body.clients, [])
        self.assertEqual(self.body.events, [])
        self.assertEqual(self.camera.sent, 0)
        self.assertEqual(detector.release_count, 1)

    def test_npu_initialization_failure_never_connects_to_robot_or_camera(self):
        # RKNN load/init failures must occur before any RPC endpoint is contacted.
        status, report, directory = self.trial(factory_error=RuntimeError('fixture RKNN init failed'))
        self.assertEqual(status, 1)
        self.assertEqual(report['outcome'], 'failed')
        self.assertEqual(self.links, [])
        self.assertEqual(self.body.clients, [])
        self.assertEqual(self.body.events, [])
        self.assertEqual(self.camera.sent, 0)
        self.assertIn('RuntimeError: fixture RKNN init failed', (directory / 'error.txt').read_text())


if __name__ == '__main__':
    unittest.main()
