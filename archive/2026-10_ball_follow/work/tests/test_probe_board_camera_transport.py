"""Board receiver fixtures bind temporary loopback TCP ports only."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
SOURCE = WORK / 'probe_board_camera_transport.py'
probe = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('board_camera_transport', SOURCE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

PAYLOAD = bytes((128, 100, 128, 100)) * 115200


@contextlib.contextmanager
def camera_server(bad_length=False):
    requests = []

    class Camera(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                if bad_length:
                    self.request.sendall(struct.pack('<I', 460799))
                else:
                    while True:
                        header = struct.pack('<I', 460800)
                        for part in (header[:1], header[1:3], header[3:],
                                     PAYLOAD[:3], PAYLOAD[3:70001], PAYLOAD[70001:]):
                            self.request.sendall(part)
                            time.sleep(0.001)
                        time.sleep(0.015)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.request.settimeout(0.5)
                try:
                    requests.append(self.request.recv(1024))
                except (TimeoutError, ConnectionResetError):
                    requests.append(b'')

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    with Server(('127.0.0.1', 0), Camera) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address, requests
        finally:
            server.shutdown()
            thread.join(1)


class BoardCameraTransportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(probe, 'the board camera receiver has not been implemented')

    def test_fragmented_frames_save_exact_raw_payloads_without_sending_requests(self):
        with tempfile.TemporaryDirectory(prefix='board-camera-fixture-') as temporary:
            root = Path(temporary)
            with camera_server() as (address, requests), contextlib.redirect_stdout(io.StringIO()):
                result = probe.main(duration=0.3, address=address, root=root)
            directory, = root.iterdir()
            report = json.loads((directory / 'summary.json').read_text())
            self.assertEqual(result, 0)
            self.assertEqual(report['outcome'], 'completed')
            self.assertGreaterEqual(report['camera']['frame_count'], 3)
            self.assertGreater(report['camera']['transport_fps'], 0)
            self.assertGreater(report['camera']['interarrival_fps'], 0)
            self.assertEqual((directory / 'camera_first.uyvy').read_bytes(), PAYLOAD)
            self.assertEqual((directory / 'camera_last.uyvy').read_bytes(), PAYLOAD)
            frames = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
            self.assertTrue(all(frame['payload_bytes'] == 460800 for frame in frames))
            self.assertFalse(report['camera']['png_generated'])
            self.assertFalse(list(directory.glob('*.png')))
            self.assertFalse((directory / 'error.txt').exists())
            self.assertEqual(requests, [b''])

    def test_wrong_length_preserves_full_exception_and_returns_failure(self):
        with tempfile.TemporaryDirectory(prefix='board-camera-bad-fixture-') as temporary:
            root = Path(temporary)
            with camera_server(bad_length=True) as (address, requests), contextlib.redirect_stdout(io.StringIO()):
                result = probe.main(duration=0.3, address=address, root=root)
            directory, = root.iterdir()
            report = json.loads((directory / 'summary.json').read_text())
            error = (directory / 'error.txt').read_text()
            self.assertEqual(result, 1)
            self.assertEqual(report['outcome'], 'error')
            self.assertEqual(report['camera']['frame_count'], 0)
            self.assertIn('Traceback', error)
            self.assertIn('FrameProtocolError', error)
            self.assertIn('460799', error)
            self.assertIn('460800', error)
            self.assertEqual(requests, [b''])

    def test_sigterm_retains_partial_frames_and_interrupted_summary(self):
        with tempfile.TemporaryDirectory(prefix='board-camera-cancel-fixture-') as temporary:
            root = Path(temporary)
            with camera_server() as (address, _):
                code = ('import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                        'from probe_board_camera_transport import main; '
                        'sys.exit(main(address=(sys.argv[2],int(sys.argv[3])),root=Path(sys.argv[4])))')
                child = subprocess.Popen([sys.executable, '-c', code, str(WORK),
                                          address[0], str(address[1]), str(root)],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 3
                    while not list(root.glob('*/camera_first.uyvy')) and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(list(root.glob('*/camera_first.uyvy')))
                    child.send_signal(signal.SIGTERM)
                    stdout, stderr = child.communicate(timeout=3)
                    self.assertEqual(child.returncode, 130, stdout + stderr)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.communicate(timeout=3)
            directory, = root.iterdir()
            report = json.loads((directory / 'summary.json').read_text())
            self.assertEqual(report['outcome'], 'interrupted')
            self.assertGreaterEqual(report['camera']['frame_count'], 1)
            self.assertEqual((directory / 'camera_last.uyvy').read_bytes(), PAYLOAD)
            self.assertIn('KeyboardInterrupt', (directory / 'error.txt').read_text())


if __name__ == '__main__':
    unittest.main()
