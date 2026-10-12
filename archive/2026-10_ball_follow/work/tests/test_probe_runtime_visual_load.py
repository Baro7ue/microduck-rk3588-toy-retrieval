"""Offline protocol and evidence checks. Servers bind ephemeral loopback ports only."""
from __future__ import annotations

import contextlib
import _thread
import importlib.util
import io
import json
from pathlib import Path
import socketserver
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'probe_runtime_visual_load.py'
probe = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('visual_load_probe', SOURCE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)


class LocalServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextlib.contextmanager
def servers(camera_mode='frames', body_mode='normal'):
    origin = time.monotonic()
    requests = []
    gray_frame = bytes([128, 100, 128, 100]) * (640 * 360 // 2)

    class Body(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                for line in self.rfile:
                    request = json.loads(line)
                    requests.append(request)
                    if request == {'op': 'hello', 'protocol': 1}:
                        response = {'protocol': 1}
                    elif request == {'op': 'read'}:
                        if body_mode == 'close' and sum(r['op'] == 'read' for r in requests) > 1:
                            return
                        response = {'sim_time': time.monotonic() - origin,
                                    'positions': [0.0] * 14, 'imu': {'gravity': [0, 0, -1]}}
                    else:
                        response = {'error': 'unexpected request'}
                    encoded = (json.dumps(response) + '\n').encode()
                    for start in range(0, len(encoded), 7):
                        self.request.sendall(encoded[start:start + 7])
            except (BrokenPipeError, ConnectionResetError):
                pass

    class Camera(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                if camera_mode == 'oversize':
                    self.request.sendall(struct.pack('<I', 0xffffffff))
                    return
                if camera_mode == 'silent':
                    time.sleep(2.5)
                    return
                packet = struct.pack('<I', len(gray_frame)) + gray_frame
                while True:
                    for start in range(0, len(packet), 16381):
                        self.request.sendall(packet[start:start + 16381])
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

    body = LocalServer(('127.0.0.1', 0), Body)
    camera = LocalServer(('127.0.0.1', 0), Camera)
    for server in (body, camera):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield body.server_address, camera.server_address, requests
    finally:
        for server in (body, camera):
            server.shutdown()
            server.server_close()


class VisualLoadTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(probe, 'the read-only visual-load helper is missing')

    def run_probe(self, camera_mode='frames', body_mode='normal', camera=True, interrupt_after=None):
        with tempfile.TemporaryDirectory(prefix='visual-load-test-') as temporary:
            root = Path(temporary)
            with servers(camera_mode, body_mode) as (body, frame, requests):
                with patch.object(probe, 'BODY_ADDRESS', body), \
                        patch.object(probe, 'CAMERA_ADDRESS', frame), \
                        patch.object(probe, 'EVIDENCE_ROOT', root), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    before = time.monotonic()
                    args = ['--seconds', '1', '--samples', '1', '--label', 'offline fixture']
                    timer = None
                    if interrupt_after is not None:
                        timer = threading.Timer(interrupt_after, _thread.interrupt_main)
                        timer.start()
                    try:
                        result = probe.main(args + (['--camera'] if camera else []))
                    finally:
                        if timer is not None:
                            timer.cancel()
                    duration = time.monotonic() - before
                directory, = root.iterdir()
                report = json.loads((directory / 'summary.json').read_text())
                states = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
                if result == 0 and camera and camera_mode == 'frames':
                    from PIL import Image
                    with Image.open(directory / 'camera_first_rotated.png') as image:
                        self.assertEqual(image.size, (360, 640))
                    self.assertEqual((directory / 'camera_last.uyvy').stat().st_size, 460800)
                return result, duration, report, states, requests

    def test_fragmented_streams_produce_midpoint_rtf_and_static_frame_evidence(self):
        result, _, report, states, requests = self.run_probe()
        self.assertEqual(result, 0)
        self.assertEqual(report['outcome'], 'completed')
        self.assertEqual(report['label'], 'offline fixture')
        self.assertEqual(len(report['windows']), 1)
        window = report['windows'][0]
        self.assertAlmostEqual(window['realtime_factor'], 1.0, delta=0.12)
        self.assertGreaterEqual(window['wall_delta_s'], 1.0)
        for sample in states:
            self.assertAlmostEqual(sample['midpoint_monotonic_s'],
                                   (sample['sent_monotonic_s'] + sample['received_monotonic_s']) / 2)
        self.assertGreaterEqual(report['camera']['frame_count'], 3)
        self.assertGreaterEqual(report['camera']['repeated_consecutive_hash_count'], 2)
        self.assertGreater(report['camera']['transport_fps'], 0)
        self.assertEqual({r['op'] for r in requests}, {'hello', 'read'})
        self.assertEqual(len(states), 2)

    def test_oversized_camera_header_fails_quickly_and_saves_partial_evidence(self):
        result, duration, report, _, _ = self.run_probe(camera_mode='oversize')
        self.assertEqual(result, 1)
        self.assertLess(duration, 1)
        self.assertEqual(report['outcome'], 'error')
        self.assertIn('FrameProtocolError', report['camera']['error'])
        self.assertIn('Traceback', report['error'])

    def test_body_disconnect_retains_first_sample(self):
        result, _, report, states, _ = self.run_probe(body_mode='close', camera=False)
        self.assertEqual(result, 1)
        self.assertEqual(report['outcome'], 'error')
        self.assertEqual(len(states), 1)
        self.assertEqual(report['windows'][0]['outcome'], 'incomplete')

    def test_requested_camera_without_frames_is_an_error(self):
        result, _, report, _, _ = self.run_probe(camera_mode='silent')
        self.assertEqual(result, 1)
        self.assertEqual(report['outcome'], 'error')
        self.assertEqual(report['camera']['frame_count'], 0)

    def test_keyboard_interrupt_saves_partial_evidence(self):
        result, duration, report, states, requests = self.run_probe(interrupt_after=0.15)
        self.assertEqual(result, 130)
        self.assertLess(duration, 0.8)
        self.assertEqual(report['outcome'], 'interrupted')
        self.assertEqual(report['completed_windows'], 0)
        self.assertGreaterEqual(len(states), 1)
        self.assertGreaterEqual(report['camera']['frame_count'], 1)
        self.assertEqual({r['op'] for r in requests}, {'hello', 'read'})

    def test_cli_rejects_nonfinite_and_out_of_range_durations_and_windows(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for value in ['nan', 'inf', '0', '61']:
                with self.subTest(seconds=value), self.assertRaises(SystemExit) as error:
                    probe.parse_args(['--seconds', value])
                self.assertEqual(error.exception.code, 2)
            for value in ['0', '11']:
                with self.subTest(samples=value), self.assertRaises(SystemExit) as error:
                    probe.parse_args(['--samples', value])
                self.assertEqual(error.exception.code, 2)

    def test_uyvy_full_range_conversion_and_clockwise_rotation(self):
        from PIL import Image
        frame = bytearray(bytes([128, 0, 128, 0]) * (640 * 360 // 2))
        frame[0:4] = bytes([128, 255, 128, 255])
        frame[1276:1280] = bytes([85, 76, 255, 76])
        with tempfile.TemporaryDirectory(prefix='uyvy-test-') as temporary:
            path = Path(temporary) / 'rotated.png'
            probe.save_rotated_png(bytes(frame), path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (360, 640))
                self.assertEqual(image.getpixel((359, 0)), (255, 255, 255))
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
                red = image.getpixel((359, 639))
            self.assertGreaterEqual(red[0], 253)
            self.assertLessEqual(red[1], 1)
            self.assertLessEqual(red[2], 1)

    def test_camera_stop_interrupts_a_silent_peer_promptly(self):
        with tempfile.TemporaryDirectory(prefix='silent-camera-test-') as temporary:
            with servers('silent') as (_, address, _):
                worker = probe.FrameWorker(address, Path(temporary))
                with contextlib.redirect_stdout(io.StringIO()):
                    worker.start()
                    time.sleep(0.05)
                    before = time.monotonic()
                    worker.stop()
                self.assertLess(time.monotonic() - before, 0.5)
                self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
