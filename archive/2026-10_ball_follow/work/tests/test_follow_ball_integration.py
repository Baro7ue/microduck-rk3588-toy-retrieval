"""Real loopback camera and temporary UNIX body exercise the complete follower."""
import copy
import json
import math
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from follow_ball_cpu import run_trial
from test_ball_follow_rpc import MODELS, STATE, UnixPeer

MODEL_SHA = '121662c804b9aa906563235ed2909815f2c23bc6e4052e651e01468817551826'


class AdvancingBody(UnixPeer):
    """20 Hz sensor clock; received intents change the emitted body state."""
    def __init__(self, path, held=False):
        self.state_lock = threading.Lock()
        self.events = []
        self.started_at = time.monotonic()
        state = copy.deepcopy(STATE)
        state['t_ns'] = time.monotonic_ns()
        if held:
            state['policy'] = 'held'
        super().__init__(path, state=state, models=MODELS)
        self.clock = threading.Thread(target=self.advance, daemon=True)
        self.clock.start()

    def advance(self):
        while not self.stopped.wait(.05):
            with self.state_lock:
                self.state['t_ns'] = time.monotonic_ns()
                state = copy.deepcopy(self.state)
            if self.subscription is not None:
                try:
                    self.send_state(state)
                except OSError:
                    return

    def handle(self, client):
        try:
            with client.makefile('rb') as reader:
                while line := reader.readline():
                    message = json.loads(line)
                    method, params = message['method'], message['params']
                    if method == 'robot.subscribe':
                        with self.state_lock:
                            state = copy.deepcopy(self.state)
                        with self.write_lock:
                            client.sendall(self.packet({'jsonrpc': '2.0', 'id': message['id'],
                                                        'result': self.models}))
                            client.sendall(self.packet({'jsonrpc': '2.0', 'method': 'robot.state',
                                                        'params': state}))
                            self.subscription = client
                        self.subscription_request = message
                        continue
                    with self.state_lock:
                        event = {'received_at': time.monotonic(), 'connection': id(client),
                                 'message': message, 't_ns': self.state['t_ns']}
                        self.events.append(event)
                        if method == 'robot.move':
                            velocity = [params[key] for key in ('vx', 'vy', 'vyaw')]
                            self.state['move'] = {'requested': velocity, 'applied': velocity.copy()}
                            if self.state['policy'] != 'held':
                                self.state['policy'] = ('walk' if math.sqrt(sum(v*v for v in velocity))
                                                        > .05 else 'stand')
                            self.moves.append(message)
                        elif method == 'robot.stop':
                            self.state['move'] = {'requested': [0.0]*3, 'applied': [0.0]*3}
                            if self.state['policy'] != 'held':
                                self.state['policy'] = 'stand'
                        elif method == 'robot.enable' and not params['on']:
                            self.state['policy'] = 'held'
                        if 'id' not in message:
                            continue
                        self.requests.append((client, message))
                        reply = {'accepted': True}
                        if method == 'robot.health':
                            reply['loop'] = {'ticks': int((time.monotonic()-self.started_at)*50),
                                             'missed': 0, 'hz': 50.0}
                    # A wrong-ID stop reply cannot authorize successful completion.
                    if method == 'robot.stop':
                        client.sendall(self.packet({'jsonrpc': '2.0', 'id': message['id']+500,
                                                    'result': {'accepted': False}}))
                    client.sendall(self.packet({'jsonrpc': '2.0', 'id': message['id'],
                                                'result': reply}))
        except (OSError, ValueError):
            pass

    def close(self):
        super().close()
        self.clock.join(.5)


class LoopbackCamera:
    def __init__(self, close_after=None):
        self.stopped = threading.Event()
        self.sent = 0
        self.client = None
        self.close_after = close_after
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(('127.0.0.1', 0))
        self.address = self.listener.getsockname()
        assert self.address[0] == '127.0.0.1' and self.address[1] not in (7801, 7901)
        self.listener.listen()
        self.listener.settimeout(.05)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

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
                    index = self.sent+1
                    # Y encodes the source fixture index; every complete UYVY frame is distinct.
                    payload = bytes((128, index % 250, 128, index % 250)) * 115_200
                    self.client.sendall(struct.pack('<I', 460_800) + payload)
                    self.sent = index
                    if self.close_after is not None and index >= self.close_after:
                        return
                    self.stopped.wait(.05)
        except OSError:
            pass  # A closed follower connection is an intentional trial boundary.

    def close(self):
        self.stopped.set()
        self.listener.close()
        if self.client is not None:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(.5)


class FixtureDetector:
    """Only expensive neural inference is substituted; all transport/control is real."""
    def __init__(self, mode='success', wrong_model=False, start_width=40):
        self.model = {'sha256': 'wrong' if wrong_model else MODEL_SHA}
        self.mode = mode
        self.start_width = start_width
        self.blocked_at = self.unblocked_at = None
        self.processed = []

    def process(self, payload):
        index = payload[1]
        self.processed.append(index)
        began = time.monotonic()
        if self.mode == 'blocked' and index == 8:
            self.blocked_at = time.monotonic()
            time.sleep(.65)
            self.unblocked_at = time.monotonic()
        box = [216-self.start_width/2, 300, 216+self.start_width/2, 340]
        if self.mode == 'already_near' or (self.mode == 'success' and index >= 11):
            box = [176, 280, 256, 320]  # 80*40 / (640*360) = 1/72, above default .012.
        if self.mode == 'triple_area' and index >= 11:
            if index < 16:
                box = [176, 300, 256, 340]  # Normalized area 1/72.
            elif index < 21:
                box = [166, 300, 266, 340]  # Normalized area 5/288.
            else:
                box = [156, 300, 276, 340]  # Normalized area exactly 1/48.
        targets = [{'class_id': 32, 'class_name': 'sports ball', 'confidence': .8, 'xyxy': box}]
        if self.mode == 'loss' and index == 8:
            targets = []
        elapsed = 1000*(time.monotonic()-began)
        return {'detections': targets, 'session_run_ms': elapsed,
                'complete_processing_ms': elapsed}, np.zeros((640, 360, 3), dtype=np.uint8)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class FollowBallIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ball-e2e-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def trial(self, mode='success', wrong_model=False, held=False, close_after=None, speed=.1,
              stop_area=.012, start_width=40, trial_name='trial'):
        body = AdvancingBody(self.root/f'{trial_name}-body.sock', held=held)
        self.addCleanup(body.close)
        camera = LoopbackCamera(close_after)
        self.addCleanup(camera.close)
        detector = FixtureDetector(mode, wrong_model, start_width=start_width)
        directory = self.root/trial_name
        status = run_trial(self.root/'unused.onnx', camera.address,
                           self.root/f'{trial_name}-body.sock', directory, duration=3,
                           detector=detector, speed=speed, stop_area=stop_area)
        report = json.loads((directory/'summary.json').read_text())
        print(f'fixture {self._testMethodName}: status={status}, outcome={report["outcome"]}, '
              f'camera_sent={camera.sent}, processed={report["processed_frame_count"]}', flush=True)
        return status, report, body, camera, detector, directory

    def assert_ordered_stop(self, body):
        stops = [event for event in body.events if event['message']['method'] == 'robot.stop']
        self.assertTrue(stops, 'the trial must issue a real stop RPC')
        first = stops[0]
        index = body.events.index(first)
        previous = body.events[index-1]
        self.assertEqual(previous['message']['method'], 'robot.move')
        self.assertEqual(previous['message']['params'], {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0})
        self.assertEqual(first['connection'], previous['connection'])
        self.assertFalse(any(event['message']['method'] == 'robot.move' and
                             event['message']['params']['vx'] > 0 for event in body.events[index:]))
        return first

    def test_reaching_area_saves_real_evidence_then_confirms_new_stand_for_one_second(self):
        status, report, body, camera, _, directory = self.trial()
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertTrue(report['walk_observed'])
        self.assertEqual(report['stop_reason'], 'target_area_reached')
        self.assertEqual(report['stop_reply'], {'accepted': True})
        self.assertEqual(body.subscription_request['params'], {'hz': 20})
        self.assertGreater(report['health_end']['loop']['ticks'], report['health_start']['loop']['ticks'])
        for file in ('frames.jsonl', 'states.jsonl', 'detections.jsonl', 'commands.jsonl',
                     'first_annotated.jpg', 'last_annotated.jpg', 'camera_first.uyvy', 'camera_last.uyvy'):
            self.assertGreater((directory/file).stat().st_size, 0)
        frames = {row['index']: row for row in rows(directory/'frames.jsonl')}
        predictions = rows(directory/'detections.jsonl')
        self.assertEqual(len(predictions), report['processed_frame_count'])
        for prediction in predictions:
            self.assertEqual(prediction['payload_sha256'], frames[prediction['receiver_index']]['payload_sha256'])
        commands = rows(directory/'commands.jsonl')
        self.assertTrue(any(row['vx'] == .1 and row['vyaw'] < 0 for row in commands))
        self.assertEqual([commands[-1][key] for key in ('vx', 'vy', 'vyaw')], [0, 0, 0])
        stop = self.assert_ordered_stop(body)
        states = [row['params'] for row in rows(directory/'states.jsonl') if row.get('method') == 'robot.state']
        self.assertTrue(any(state['policy'] == 'walk' and state['move']['applied'][0] == .1 for state in states))
        final_tick = report['final_state']['t_ns']
        settled = [state for state in states if stop['t_ns'] < state['t_ns'] <= final_tick]
        # At 20 Hz, the last raw sample can precede the end of a one-second observation by one tick.
        health_end = [event for event in body.events
                      if event['message']['method'] == 'robot.health'][-1]
        observed_s = health_end['received_at']-settled[0]['t_ns']/1_000_000_000
        self.assertGreaterEqual(observed_s, 1.0)
        self.assertGreaterEqual((settled[-1]['t_ns']-settled[0]['t_ns'])/1_000_000_000, .9)
        self.assertTrue(all(0 < (b['t_ns']-a['t_ns'])/1_000_000_000 < .1
                            for a, b in zip(settled, settled[1:])))
        print(f'new standing observation={observed_s:.3f}s; raw samples={len(settled)}', flush=True)
        self.assertTrue(all(state['policy'] == 'stand' and state['move']['requested'] == [0]*3
                            and state['move']['applied'] == [0]*3 for state in settled))
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in body.events))
        self.assertGreaterEqual(camera.sent, 11)

    def test_speed_point_three_reaches_body_and_summary_then_stops_and_confirms_stand(self):
        status, report, body, _, _, directory = self.trial(speed=.3)
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertEqual(report['speed_command_m_s'], .3)
        self.assertEqual(report['duration_limit_s'], 3)
        self.assertEqual(report['freshness_limit_s'], .3)
        self.assertEqual(report['stop_area_fraction'], .012)
        self.assertNotIn('area_stop_ratio', report)
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertTrue(report['walk_observed'])
        commands = rows(directory/'commands.jsonl')
        moving = [row for row in commands if row['vx'] > 0]
        self.assertTrue(moving)
        self.assertTrue(all(row['vx'] == .3 and row['vy'] == 0 and row['vyaw'] == -.12
                            for row in moving))
        stop = self.assert_ordered_stop(body)
        states = [row['params'] for row in rows(directory/'states.jsonl') if row.get('method') == 'robot.state']
        self.assertTrue(any(state['policy'] == 'walk' and state['move']['requested'][0] == .3
                            and state['move']['applied'][0] == .3 for state in states))
        settled = [state for state in states
                   if stop['t_ns'] < state['t_ns'] <= report['final_state']['t_ns']]
        health_end = [event for event in body.events
                      if event['message']['method'] == 'robot.health'][-1]
        observed_s = health_end['received_at']-settled[0]['t_ns']/1_000_000_000
        self.assertGreaterEqual(observed_s, 1.0)
        self.assertGreaterEqual((settled[-1]['t_ns']-settled[0]['t_ns'])/1_000_000_000, .9)
        self.assertTrue(all(0 < (b['t_ns']-a['t_ns'])/1_000_000_000 < .1
                            for a, b in zip(settled, settled[1:])))
        self.assertTrue(all(state['policy'] == 'stand' and state['move']['requested'] == [0]*3
                            and state['move']['applied'] == [0]*3 for state in settled))
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in body.events))
        print(f'.3 m/s standing observation={observed_s:.3f}s; raw samples={len(settled)}', flush=True)

    def test_fixed_area_reaches_real_commands_and_report_before_confirmed_stop(self):
        # Ignoring the configured fixed threshold would stop at frame 11 instead of frame 21.
        status, report, body, _, _, directory = self.trial(mode='triple_area', speed=.3,
                                                        stop_area=1/48)
        self.assertEqual(status, 0)
        self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertEqual(report['stop_area_fraction'], 1/48)
        self.assertNotIn('area_stop_ratio', report)
        self.assertAlmostEqual(report['baseline_area'], 1/144)
        self.assertAlmostEqual(report['target_area'], 1/48)
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertTrue(report['walk_observed'])
        predictions = rows(directory/'detections.jsonl')
        below_target = [row for row in predictions if 11 <= row['receiver_index'] < 21]
        self.assertTrue(below_target)
        self.assertTrue(all(row['control']['vx'] == .3 and row['control']['reason'] is None
                            for row in below_target))
        reached = next(row for row in predictions
                       if row['control']['reason'] == 'target_area_reached')
        self.assertGreaterEqual(reached['receiver_index'], 21)
        self.assertAlmostEqual(reached['control']['area'], 1/48)
        commands = rows(directory/'commands.jsonl')
        self.assertTrue(any(row['vx'] == .3 and 11 <= row['receiver_index'] < 21
                            for row in commands))
        self.assertTrue(all(row['vx'] == 0 for row in commands
                            if row['sent_monotonic_s'] >= reached['processed_monotonic_s']))
        stop = self.assert_ordered_stop(body)
        states = [row['params'] for row in rows(directory/'states.jsonl')
                  if row.get('method') == 'robot.state']
        settled = [state for state in states
                   if stop['t_ns'] < state['t_ns'] <= report['final_state']['t_ns']]
        health_end = [event for event in body.events
                      if event['message']['method'] == 'robot.health'][-1]
        observed_s = health_end['received_at']-settled[0]['t_ns']/1_000_000_000
        self.assertGreaterEqual(observed_s, 1)
        # Thread scheduling may deliver 19 samples over one wall-clock second.
        self.assertGreaterEqual((settled[-1]['t_ns']-settled[0]['t_ns'])/1_000_000_000, .9)
        self.assertTrue(all(0 < (b['t_ns']-a['t_ns'])/1_000_000_000 < .1
                            for a, b in zip(settled, settled[1:])))
        self.assertTrue(all(state['policy'] == 'stand' and state['move']['requested'] == [0]*3
                            and state['move']['applied'] == [0]*3 for state in settled))
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in body.events))

    def test_different_initial_box_sizes_reach_the_same_configured_stop_area(self):
        # Multiplying the initial baseline would move the target when only the start size changes.
        for width, expected_baseline in ((40, 1/144), (60, 1/96)):
            with self.subTest(start_width=width):
                status, report, body, _, _, directory = self.trial(
                    start_width=width, stop_area=.012, trial_name=f'start-{width}')
                self.assertEqual(status, 0)
                self.assertEqual(report['outcome'], 'completed_target_area_returned_stand')
                self.assertAlmostEqual(report['baseline_area'], expected_baseline)
                self.assertEqual(report['stop_area_fraction'], .012)
                self.assertEqual(report['target_area'], .012)
                predictions = rows(directory/'detections.jsonl')
                reached = next(row for row in predictions
                               if row['control']['reason'] == 'target_area_reached')
                self.assertGreaterEqual(reached['receiver_index'], 11)
                self.assertAlmostEqual(reached['control']['area'], 1/72)
                self.assertTrue(report['stop_confirmed'])
                self.assertTrue(report['left_enabled_standing'])
                self.assert_ordered_stop(body)

    def test_lost_ball_latches_zero_even_as_recovered_camera_frames_continue(self):
        status, report, body, camera, detector, directory = self.trial(mode='loss')
        self.assertEqual(status, 2)
        self.assertEqual(report['outcome'], 'stopped_missing_ball')
        self.assertEqual(report['stop_reason'], 'missing_ball')
        self.assertTrue(report['stop_confirmed'])
        self.assertTrue(report['left_enabled_standing'])
        self.assertIn(8, detector.processed)
        self.assertGreaterEqual(camera.sent, 11, 'the source must continue recovered-ball frames')
        commands = rows(directory/'commands.jsonl')
        self.assertTrue(any(row['vx'] > 0 for row in commands))
        lost = next(row for row in rows(directory/'detections.jsonl') if not row['detections'])
        self.assertTrue(all([row[key] for key in ('vx', 'vy', 'vyaw')] == [0, 0, 0]
                            for row in commands
                            if row['sent_monotonic_s'] >= lost['processed_monotonic_s']))
        self.assert_ordered_stop(body)
        self.assertEqual(report['final_state']['policy'], 'stand')
        self.assertEqual(report['final_state']['move']['requested'], [0, 0, 0])
        self.assertEqual(report['final_state']['move']['applied'], [0, 0, 0])
        self.assertNotIn('robot.enable_cleanup', report)
        self.assertFalse(any(event['message']['method'] == 'robot.enable' for event in body.events))

    def test_blocked_inference_cannot_delay_sender_zero_until_it_returns(self):
        status, report, body, _, detector, directory = self.trial(mode='blocked')
        self.assertNotEqual(status, 0)
        self.assertEqual(report['stop_reason'], 'frame_timeout')
        self.assertIsNotNone(detector.blocked_at)
        commands = rows(directory/'commands.jsonl')
        self.assertTrue(any(row['vx'] > 0 for row in commands))
        zeros = [row for row in commands if row['vx'] == 0 and
                 row['sent_monotonic_s'] >= detector.blocked_at]
        self.assertTrue(zeros)
        self.assertLessEqual(zeros[0]['sent_monotonic_s']-detector.blocked_at, .3)
        self.assertLess(zeros[0]['sent_monotonic_s'], detector.unblocked_at)
        print(f'blocked-inference first zero after '
              f'{zeros[0]["sent_monotonic_s"]-detector.blocked_at:.3f}s', flush=True)
        self.assertTrue(all(row['vx'] == 0 for row in commands
                            if row['sent_monotonic_s'] >= zeros[0]['sent_monotonic_s']))
        self.assert_ordered_stop(body)

    def test_wrong_model_hash_never_connects_or_sends_a_nonzero_intent(self):
        status, report, body, camera, _, _ = self.trial(wrong_model=True)
        self.assertNotEqual(status, 0)
        self.assertEqual(report['outcome'], 'failed')
        self.assertFalse(report['left_enabled_standing'])
        self.assertEqual(body.events, [])
        self.assertEqual(camera.sent, 0)

    def test_initial_held_state_never_sends_a_nonzero_intent(self):
        status, report, body, camera, _, _ = self.trial(held=True)
        self.assertNotEqual(status, 0)
        self.assertEqual(report['outcome'], 'failed')
        self.assertFalse(report['left_enabled_standing'])
        self.assertFalse(any(event['message']['method'] == 'robot.move' and
                             event['message']['params']['vx'] > 0 for event in body.events))
        self.assertEqual(camera.sent, 0)

    def test_camera_eof_stops_in_order_and_disables_after_a_walking_state(self):
        status, report, body, _, _, directory = self.trial(mode='blocked', close_after=8)
        self.assertNotEqual(status, 0)
        self.assertEqual(report['outcome'], 'failed')
        self.assertFalse(report['left_enabled_standing'])
        self.assertTrue(any(row['vx'] > 0 for row in rows(directory/'commands.jsonl')))
        self.assertIn('ConnectionError', report['transport']['error'])
        self.assert_ordered_stop(body)
        self.assertTrue(report['robot.enable_cleanup']['accepted'])

    def test_final_camera_write_failure_revokes_completed_and_attempts_disable(self):
        from probe_runtime_visual_load import FrameWorker
        original = FrameWorker.stop

        def fail_final_write(worker):
            original(worker)
            worker.error = 'fixture final frame write failed'

        with patch.object(FrameWorker, 'stop', fail_final_write):
            status, report, body, _, _, _ = self.trial()
        self.assertNotEqual(status, 0)
        self.assertNotEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertFalse(report['left_enabled_standing'])
        self.assertTrue(report['robot.enable_cleanup']['accepted'])
        self.assert_ordered_stop(body)

    def test_last_annotated_image_write_failure_still_saves_summary_and_disables(self):
        import probe_board_yolo11_camera
        original = probe_board_yolo11_camera.save_image

        def fail_last_image(path, image):
            if Path(path).name == 'last_annotated.jpg':
                raise OSError('fixture final annotated image write failed')
            original(path, image)

        with patch.object(probe_board_yolo11_camera, 'save_image', fail_last_image):
            status, report, body, _, _, _ = self.trial()
        self.assertNotEqual(status, 0)
        self.assertNotEqual(report['outcome'], 'completed_target_area_returned_stand')
        self.assertFalse(report['left_enabled_standing'])
        detail = '\n'.join(value for key, value in report.items()
                           if (key == 'error' or key.endswith('_error')) and isinstance(value, str))
        self.assertIn('Traceback (most recent call last)', detail)
        self.assertIn('OSError', detail)
        self.assertIn('fixture final annotated image write failed', detail)
        self.assertTrue(report['robot.enable_cleanup']['accepted'])
        self.assert_ordered_stop(body)


if __name__ == '__main__':
    unittest.main()
