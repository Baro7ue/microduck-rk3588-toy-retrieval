"""Close-follow opt-in exercises real camera transport, RPC and stop confirmation."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
import follow_ball_cpu
import test_follow_ball_integration as fixtures
from test_follow_ball_npu import FixtureNPUDetector, follow_npu


def growing_predictions(payload):
    """Five sports-ball acquisitions, orange continuation, one empty frame and growth."""
    index = payload[1]
    width = (80 if index <= 5 else 90 if index <= 8 else
             100 if index <= 11 else 110 if index <= 14 else 120)
    class_id = 32 if index <= 5 else 49
    targets = [] if index == 7 else [{
        'class_id': class_id,
        'class_name': 'sports ball' if class_id == 32 else 'orange',
        'confidence': .8,
        'xyxy': [180-width/2, 320-width/2, 180+width/2, 320+width/2],
    }]
    return {'detections': targets, 'complete_processing_ms': .01}, np.zeros(
        (640, 360, 3), dtype=np.uint8)


class GrowingCPUDetector(fixtures.FixtureDetector):
    def process(self, payload):
        self.processed.append(payload[1])
        prediction, image = growing_predictions(payload)
        prediction['session_run_ms'] = .01
        return prediction, image


class GrowingNPUDetector(FixtureNPUDetector):
    def process(self, payload):
        self.processed.append(payload[1])
        prediction, image = growing_predictions(payload)
        prediction['inference_call_ms'] = .01
        return prediction, image


class ClippedNPUDetector(GrowingNPUDetector):
    def process(self, payload):
        prediction, image = super().process(payload)
        if payload[1] == 6:
            # This partial edge box exceeds the fixed stop area and the scale ratio limit.
            prediction['detections'] = [{
                'class_id': 32, 'class_name': 'sports ball', 'confidence': .8,
                'xyxy': [0, 240, 160, 400],
            }]
        elif payload[1] == 7:
            prediction['detections'] = [{
                'class_id': 49, 'class_name': 'orange', 'confidence': .8,
                'xyxy': [135, 275, 225, 365],
            }]
        return prediction, image


class CloseBallFollowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='close-ball-e2e-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = fixtures.AdvancingBody(self.root/'body.sock')
        self.addCleanup(self.body.close)
        self.camera = fixtures.LoopbackCamera()
        self.addCleanup(self.camera.close)

    def trial(self, *, backend='cpu', orange_continuation=None, detector=None):
        directory = self.root/'trial'
        arguments = dict(duration=3, speed=.3, stop_area=.06)
        if orange_continuation is not None:
            arguments['orange_continuation'] = orange_continuation
        detector = detector or (GrowingCPUDetector() if backend == 'cpu' else GrowingNPUDetector())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if backend == 'cpu':
                status = follow_ball_cpu.run_trial(self.root/'unused.onnx', self.camera.address,
                    self.root/'body.sock', directory, detector=detector, **arguments)
            else:
                with patch.object(follow_npu, 'NPUCameraDetector', return_value=detector) as factory:
                    status = follow_npu.run_trial(self.root/'unused.rknn', self.camera.address,
                        self.root/'body.sock', directory, **arguments)
                self.assertTrue(factory.call_args.kwargs['include_orange'])
        report = json.loads((directory/'summary.json').read_text())
        return status, report, detector, directory

    def assert_close_approach(self, status, report, directory):
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertTrue(report['orange_continuation'])
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertTrue(report['walk_observed'])
        self.assertEqual(report['stop_reason'], 'target_area_reached')
        self.assertEqual(report['stop_area_fraction'], .06)
        predictions = fixtures.rows(directory/'detections.jsonl')
        oranges = [row for row in predictions if any(
            target['class_id'] == 49 for target in row['detections'])]
        self.assertTrue(oranges, 'raw class 49 evidence must be retained')
        self.assertTrue(all(row['control']['source_class_id'] == 49 for row in oranges))
        self.assertTrue(any(row['control']['vx'] == .3 for row in oranges))
        empty = next(row for row in predictions if not row['detections'])
        self.assertEqual(empty['receiver_index'], 7)
        self.assertEqual(empty['control']['mode'], 'holding')
        self.assertEqual(empty['control']['reason'], None)
        self.assertEqual([empty['control'][key] for key in ('vx', 'vy', 'vyaw')], [0, 0, 0])
        resumed = next(row for row in predictions if row['receiver_index'] > 7 and
                       row['control']['vx'] == .3)
        self.assertEqual(resumed['control']['source_class_id'], 49)
        reached = next(row for row in predictions if row['control']['reason'] == 'target_area_reached')
        self.assertGreaterEqual(reached['receiver_index'], 15)
        self.assertAlmostEqual(reached['control']['area'], .0625)
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(any(row['receiver_index'] == 7 and row['vx'] == 0 for row in commands),
                        'temporary loss must send zero to the actual RPC peer')
        self.assertTrue(any(row['receiver_index'] is not None and row['receiver_index'] > 7 and
                            row['vx'] == .3 for row in commands))
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))

    def test_cpu_opt_in_continues_orange_after_zero_hold_then_reaches_fixed_close_area(self):
        status, report, _, directory = self.trial(orange_continuation=True)
        self.assert_close_approach(status, report, directory)

    def test_npu_opt_in_continues_orange_after_zero_hold_then_reaches_fixed_close_area(self):
        self.assertIsNotNone(follow_npu)
        status, report, detector, directory = self.trial(backend='npu', orange_continuation=True)
        self.assert_close_approach(status, report, directory)
        self.assertEqual(detector.release_count, 1)
        self.assertEqual(report['runtime']['backend'], 'npu')

    def test_npu_clipped_large_box_holds_zero_without_false_stop_then_full_orange_resumes(self):
        status, report, detector, directory = self.trial(
            backend='npu', orange_continuation=True, detector=ClippedNPUDetector())
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        predictions = fixtures.rows(directory/'detections.jsonl')
        partial = next(row for row in predictions if row['receiver_index'] == 6)
        self.assertEqual(partial['detections'][0]['xyxy'], [0, 240, 160, 400])
        self.assertEqual(partial['detections'][0]['class_id'], 32)
        self.assertGreaterEqual(160*160/(360*640), report['stop_area_fraction'])
        self.assertEqual(partial['control']['mode'], 'holding')
        self.assertEqual(partial['control']['association'], 'clipped_hold')
        self.assertIsNone(partial['control']['area'])
        self.assertIsNone(partial['control']['reason'])
        self.assertEqual([partial['control'][key] for key in ('vx', 'vy', 'vyaw')], [0, 0, 0])
        restored = next(row for row in predictions if row['receiver_index'] == 7)
        self.assertEqual(restored['detections'][0]['class_id'], 49)
        self.assertEqual(restored['control']['source_class_id'], 49)
        self.assertEqual(restored['control']['vx'], .3)
        self.assertEqual(restored['control']['association'], 'spatial_continuation')
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertTrue(any(row['receiver_index'] == 6 and row['vx'] == 0 for row in commands),
                        'partial detections must send zero to the actual RPC peer')
        self.assertTrue(any(row['receiver_index'] is not None and row['receiver_index'] >= 7 and
                            row['vx'] == .3 for row in commands))
        reached = next(row for row in predictions if row['control']['reason'] == 'target_area_reached')
        self.assertGreaterEqual(reached['receiver_index'], 15)
        self.assertAlmostEqual(reached['control']['area'], .0625)
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))
        self.assertEqual(detector.release_count, 1)

    def test_default_mode_stops_on_orange_and_retains_legacy_no_resume(self):
        status, report, _, directory = self.trial()
        self.assertEqual(status, 2, json.dumps(report, ensure_ascii=False, indent=2))
        self.assertFalse(report['orange_continuation'])
        self.assertEqual(report['outcome'], 'stopped_missing_ball')
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        predictions = fixtures.rows(directory/'detections.jsonl')
        orange = next(row for row in predictions if any(
            target['class_id'] == 49 for target in row['detections']))
        self.assertEqual(orange['control']['reason'], 'missing_ball')
        self.assertEqual(orange['control']['vx'], 0)
        commands = fixtures.rows(directory/'commands.jsonl')
        self.assertFalse(any(row['vx'] > 0 for row in commands if
                             row['sent_monotonic_s'] >= orange['processed_monotonic_s']))
        fixtures.FollowBallIntegrationTests.assert_ordered_stop(self, self.body)
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in self.body.events))


if __name__ == '__main__':
    unittest.main()
