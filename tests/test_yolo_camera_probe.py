# Moved 2026-10-11 from work/tests/test_probe_board_yolo11_camera.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Camera-only fixtures: temporary loopback sockets and the actual exported ONNX."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))
SOURCE = SRC / 'microduck_toy/vision/yolo_camera_probe.py'
MODEL = REPO / 'models/yolo/yolo11n_coco/yolo11n_coco_fp32_320.onnx'
probe = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('board_yolo_camera', SOURCE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)


class LocalServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextlib.contextmanager
def camera(mode='frames'):
    payload = bytes([128, 100, 128, 100]) * (640 * 360 // 2)

    class Camera(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                packet = struct.pack('<I', len(payload)) + payload
                while True:
                    # Split the length header and payload across actual socket writes.
                    for chunk in (packet[:1], packet[1:3], packet[3:10007]):
                        self.request.sendall(chunk)
                    for start in range(10007, len(packet), 16381):
                        self.request.sendall(packet[start:start + 16381])
                    if mode != 'frames':
                        time.sleep(0.1)
                        if mode == 'bad_length':
                            self.request.sendall(struct.pack('<I', 17))
                        elif mode == 'close':
                            self.request.sendall(struct.pack('<I', len(payload)) + payload[:31])
                        return
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = LocalServer(('127.0.0.1', 0), Camera)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address, payload
    finally:
        server.shutdown()
        server.server_close()


class CameraDetectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(probe, 'the camera-only detector is missing')

    def test_full_range_gray_pixels_rotate_clockwise(self):
        # A studio-range decoder would turn Y=16 into black; wrong rotation swaps sides.
        payload = bytes([128, 16, 128, 235, 128, 0, 128, 255,
                         128, 32, 128, 100, 128, 50, 128, 200])
        image = probe.decode_frame(payload, width=4, height=2)
        expected = np.array([[[32]*3, [16]*3], [[100]*3, [235]*3],
                             [[50]*3, [0]*3], [[200]*3, [255]*3]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_opt_in_orange_runs_the_shared_cpu_postprocess_without_relabeling(self):
        # A CPU/NPU class filter mismatch would make the same continuity rule differ by backend.
        detector = probe.CameraDetector(MODEL, include_orange=True)
        raw = np.zeros((1, 84, 2100), dtype=np.float32)
        raw[0, :4, 0] = [160, 160, 40, 20]
        raw[0, 4 + 49, 0] = .8
        payload = bytes([128, 100, 128, 100]) * (640 * 360 // 2)
        with mock.patch.object(detector.session, 'run', return_value=[raw]):
            result, _ = detector.process(payload)
        target, = result['detections']
        self.assertEqual((target['class_id'], target['class_name']), (49, 'orange'))
        np.testing.assert_allclose(target['xyxy'], [140, 300, 220, 340])

    def test_full_range_bt601_chroma_and_bgr_order(self):
        # Hand-calculated BT.601 full-range values; catches RGB/BGR and UV reversals.
        image = probe.decode_frame(bytes([90, 81, 240, 145]), width=2, height=1)
        np.testing.assert_array_equal(image,
                                      np.array([[[14, 14, 238]], [[78, 78, 255]]], dtype=np.uint8))

    def test_latest_slot_keeps_pairs_and_skips_without_repeating(self):
        # An unbounded queue would return frame 1; split record/payload publication can mismatch.
        slot = probe.LatestFrameSlot()
        slot.publish({'index': 1, 'payload_sha256': 'a'}, b'a')
        slot.publish({'index': 2, 'payload_sha256': 'b'}, b'b')
        record = {'index': 3, 'payload_sha256': 'c'}
        slot.publish(record, b'c')
        record['index'] = 999
        latest = slot.take_after(0, 0)
        self.assertEqual(latest, ({'index': 3, 'payload_sha256': 'c'}, b'c'))
        self.assertIsNone(slot.take_after(3, 0.01))
        slot.publish({'index': 4, 'payload_sha256': 'd'}, b'd')
        self.assertEqual(slot.take_after(3, 0), ({'index': 4, 'payload_sha256': 'd'}, b'd'))

    def test_frame_worker_callback_receives_only_complete_persisted_frames(self):
        # Calling back after a partial recv or before recording breaks the consumer's evidence pair.
        with tempfile.TemporaryDirectory(prefix='camera-callback-test-') as temp, \
                camera() as (address, payload), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            received = []
            ready = threading.Event()

            def callback(record, data):
                persisted = [json.loads(x) for x in (directory/'frames.jsonl').read_text().splitlines()]
                received.append((record, data, persisted[-1]))
                ready.set()

            worker = probe.FrameWorker(address, directory, on_frame=callback)
            worker.start()
            try:
                self.assertTrue(ready.wait(3))
            finally:
                worker.stop()
            worker.raise_if_failed()
            record, data, persisted = received[0]
            self.assertEqual(data, payload)
            self.assertEqual(record, persisted)
            self.assertEqual(record['payload_sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual((directory/'camera_first.uyvy').read_bytes(), payload)

    def run_fixture(self, mode='frames'):
        with tempfile.TemporaryDirectory(prefix='camera-yolo-test-') as temp, \
                camera(mode) as (address, payload), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = probe.main(duration=0.35, address=address, root=Path(temp), model_path=MODEL)
            directory, = Path(temp).iterdir()
            summary = json.loads((directory/'summary.json').read_text())
            frames = [json.loads(x) for x in (directory/'frames.jsonl').read_text().splitlines()]
            detections = [json.loads(x) for x in (directory/'detections.jsonl').read_text().splitlines()]
            error = (directory/'error.txt').read_text() if (directory/'error.txt').exists() else None
            first = (directory/'camera_first.uyvy').read_bytes()
            last = (directory/'camera_last.uyvy').read_bytes()
            images = [x.exists() for x in (directory/'first_annotated.jpg', directory/'last_annotated.jpg')]
            return result, summary, frames, detections, error, first, last, images, payload

    def test_fragmented_camera_runs_actual_onnx_and_records_matching_detections(self):
        # Fake inference or repeated latest-frame consumption would not produce this coherent evidence.
        result, summary, frames, detections, error, first, last, images, payload = self.run_fixture()
        self.assertEqual(result, 0)
        self.assertEqual(summary['outcome'], 'completed')
        self.assertGreaterEqual(len(detections), 2)
        self.assertEqual(summary['received_frame_count'], len(frames))
        self.assertEqual(summary['processed_frame_count'], len(detections))
        self.assertEqual(summary['providers'], ['CPUExecutionProvider'])
        self.assertEqual(summary['model']['sha256'], hashlib.sha256(MODEL.read_bytes()).hexdigest())
        indices = [d['receiver_index'] for d in detections]
        self.assertEqual(indices, sorted(set(indices)))
        recorded = {r['index']: r for r in frames}
        for d in detections:
            self.assertEqual(d['payload_sha256'], recorded[d['receiver_index']]['payload_sha256'])
            self.assertGreaterEqual(d['receive_to_process_wait_ms'], 0)
            self.assertGreater(d['session_run_ms'], 0)
            self.assertGreaterEqual(d['complete_processing_ms'], d['session_run_ms'])
        self.assertEqual(first, payload)
        self.assertEqual(last, payload)
        self.assertEqual(images, [True, True])
        self.assertIsNone(error)
        self.assertGreaterEqual(summary['elapsed_from_connection_s'], 0.35)
        self.assertLess(summary['elapsed_from_connection_s'], 1.0)
        self.assertGreater(summary['cpu_window']['process_time_s'], 0)
        self.assertGreater(summary['cpu_window']['cpu_percent_one_core'], 0)
        for field in ('session_run_ms', 'complete_processing_ms'):
            samples = [d[field] for d in detections]
            self.assertAlmostEqual(summary['timing_ms'][field]['p50'], float(np.median(samples)))
            self.assertAlmostEqual(summary['timing_ms'][field]['p95'], float(np.percentile(samples, 95)))
            self.assertEqual(summary['timing_ms'][field]['max'], max(samples))

    def test_bad_length_preserves_full_protocol_error_and_completed_frame(self):
        # Silently ignoring a malformed packet would report completed instead of retaining an error.
        result, summary, frames, _, error, first, last, _, payload = self.run_fixture('bad_length')
        self.assertEqual(result, 1)
        self.assertEqual(summary['outcome'], 'failed')
        self.assertEqual(len(frames), 1)
        self.assertIn('FrameProtocolError', error)
        self.assertIn('17', error)
        self.assertEqual(first, payload)
        self.assertEqual(last, payload)

    def test_mid_payload_close_preserves_error_and_partial_evidence(self):
        # Closing while reading a payload must not publish the partial bytes as a second frame.
        result, summary, frames, _, error, first, last, _, payload = self.run_fixture('close')
        self.assertEqual(result, 1)
        self.assertEqual(summary['outcome'], 'failed')
        self.assertEqual(len(frames), 1)
        self.assertIn('ConnectionError', error)
        self.assertIn('31/460800', error)
        self.assertEqual(first, payload)
        self.assertEqual(last, payload)

    def test_sigterm_saves_partial_camera_evidence_and_returns_130(self):
        # A raw SIGTERM exits -15 and skips finally, losing summary and the final image.
        with tempfile.TemporaryDirectory(prefix='camera-sigterm-test-') as temp, \
                camera() as (address, payload):
            directory = Path(temp) / 'partial'
            code = (f'import sys; sys.path.insert(0, {str(SRC)!r}); '
                    'from microduck_toy.vision import yolo_camera_probe as probe; '
                    f'sys.exit(probe.main(duration=30, address={address!r}, '
                    f'model_path=probe.Path({str(MODEL)!r}), '
                    f'directory=probe.Path({str(directory)!r})))')
            child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                first_image = directory / 'first_annotated.jpg'
                while not first_image.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(first_image.exists(), 'child did not process the first real frame')
                child.terminate()
                stdout, stderr = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 130, stdout + stderr)
                summary = json.loads((directory/'summary.json').read_text())
                self.assertEqual(summary['outcome'], 'interrupted')
                self.assertGreaterEqual(summary['received_frame_count'], 1)
                self.assertGreaterEqual(summary['processed_frame_count'], 1)
                self.assertIn('KeyboardInterrupt', (directory/'error.txt').read_text())
                self.assertEqual((directory/'camera_first.uyvy').read_bytes(), payload)
                self.assertEqual((directory/'camera_last.uyvy').read_bytes(), payload)
                self.assertTrue((directory/'last_annotated.jpg').exists())
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=5)

    def test_first_jpeg_save_failure_does_not_count_processed_frame_as_tail(self):
        # Saving an illustration can fail after inference and its JSON record have completed.
        with tempfile.TemporaryDirectory(prefix='camera-jpeg-error-test-') as temp, \
                camera() as (address, _), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(probe, 'save_image', side_effect=OSError('fixture JPEG save failed')):
            status = probe.main(duration=0.35, address=address, root=Path(temp), model_path=MODEL)
            directory, = Path(temp).iterdir()
            summary = json.loads((directory/'summary.json').read_text())
            self.assertEqual(status, 1)
            self.assertEqual(summary['outcome'], 'failed')
            self.assertGreaterEqual(summary['processed_frame_count'], 1)
            self.assertEqual(summary['received_frame_count'],
                             summary['processed_frame_count'] + summary['skipped_receiver_frames'] +
                             summary['unprocessed_tail_receiver_frames'])
            self.assertIn('OSError: fixture JPEG save failed', (directory/'error.txt').read_text())


if __name__ == '__main__':
    unittest.main()
