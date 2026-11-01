"""Stage 2 2F live detector: window decode, camera receiver, timed detect, robotd tick summary,
NPU load parsing, the motion schedule and the deployment table."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest

import numpy as np

from microduck_toy.stage2 import deploy
from microduck_toy.vision import detect, uyvy, views

from test_stage2_detect import split_outputs

NAMES = ['bus', 'henry', 'ketchup', 'croc', 'thomas', 'tray']


def payload(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, deploy.FRAME_BYTES, dtype=np.uint8).tobytes()


def full_frame(data: bytes) -> np.ndarray:
    pairs = np.frombuffer(data, np.uint8).reshape(deploy.RAW_H, deploy.RAW_W // 2, 4)
    return np.ascontiguousarray(np.rot90(uyvy.decode(pairs), 3))


class FrameInputTests(unittest.TestCase):
    def test_window_from_its_columns_equals_the_whole_frame_path(self):
        data = payload(1)
        frame = full_frame(data)
        for y0 in (0, 1, 137, 200, 279, 280):
            m = views.Mapping('window', 0.0, float(y0), views.INPUT / views.WINDOW, 0.0, 0.0)
            np.testing.assert_array_equal(deploy.frame_input(data, m), views.to_input(frame, m))

    def test_letterbox_equals_the_whole_frame_path(self):
        data = payload(2)
        m = views.letterbox_mapping()
        np.testing.assert_array_equal(deploy.frame_input(data, m), views.to_input(full_frame(data), m))

    def test_mapping_follows_view_and_falls_back_to_letterbox(self):
        self.assertEqual(deploy.mapping_for(None, 'search', 30).kind, 'letterbox')
        level = {'frames': {'camera': {'pos': [0.07, 0.0, 0.11],
                                       # camera looking along the trunk's +x, level (OpenCV axes)
                                       'quat': [0.5, -0.5, 0.5, -0.5]}},
                 'imu': {'quat': [1.0, 0.0, 0.0, 0.0]}, 'odom': {'position': [0.0, 0.0, 0.12]}}
        m = deploy.mapping_for(level, 'search', 30)
        # horizon at the middle row (320): 30 rows above is 290, kept inside the frame at 640 - 360
        self.assertEqual((m.kind, m.y0), ('window', 280.0))
        self.assertEqual(deploy.mapping_for(level, 'approach', 30).kind, 'letterbox')


class Server:
    """A one-shot camera: sends the given (length, payload) records, then waits to be closed."""

    def __init__(self, records):
        self.sock = socket.socket()
        self.sock.bind(('127.0.0.1', 0))
        self.sock.listen(1)
        self.address = self.sock.getsockname()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._serve, args=(records,), daemon=True)
        self.thread.start()

    def _serve(self, records):
        conn, _ = self.sock.accept()
        with conn:
            for size, data in records:
                conn.sendall(struct.pack('<I', size) + data)
                time.sleep(0.02)
            self.done.wait(3.0)

    def close(self):
        self.done.set()
        self.sock.close()


class ReceiverTests(unittest.TestCase):
    def test_keeps_the_newest_frame_with_arrival_times(self):
        frames = [payload(i) for i in range(3)]
        server = Server([(deploy.FRAME_BYTES, f) for f in frames])
        rx = deploy.FrameReceiver(server.address)
        rx.start()
        try:
            first = rx.take_after(0, 2.0)
            self.assertIsNotNone(first)
            deadline = time.monotonic() + 2.0
            while rx.count < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            last = rx.take_after(first.seq, 1.0)
            self.assertEqual(last.seq, 3)
            self.assertEqual(last.payload, frames[2])
            self.assertLessEqual(last.first, last.done)
            self.assertIsNone(rx.take_after(3, 0.05))
        finally:
            rx.stop()
            server.close()

    def test_a_frame_of_the_wrong_size_is_an_error(self):
        server = Server([(100, b'x' * 100)])
        rx = deploy.FrameReceiver(server.address)
        rx.start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'expected 460800'):
                rx.take_after(0, 2.0)
        finally:
            rx.stop()
            server.close()


class FakeNet:
    names = NAMES

    def __init__(self):
        self.seen = None

    prepare = staticmethod(detect.RknnYolo.prepare)

    def run(self, x):
        self.seen = x
        return split_outputs(np.random.default_rng(3), hot=[(1, 4, 7, 1, 0.9)])


class DetectTests(unittest.TestCase):
    def test_boxes_come_back_in_frame_pixels_with_all_times(self):
        net = FakeNet()
        det = deploy.LiveDetector('int8', Path('unused'), NAMES, 0.25, 0.45, net=net)
        frame = deploy.Frame(1, payload(4), first=time.perf_counter() - 0.004,
                             done=time.perf_counter())
        dets, m, ms = det.detect(frame, None, 'search', 30)
        self.assertEqual(m.kind, 'letterbox')                    # no state -> whole frame
        self.assertEqual(net.seen.shape, (1, 320, 320, 3))
        self.assertEqual(net.seen.dtype, np.uint8)
        self.assertEqual([d['cls'] for d in dets], ['henry'])
        np.testing.assert_allclose(dets[0]['xyxy'], views.to_frame_box(dets[0]['xyxy_input'], m),
                                   atol=0.01)
        self.assertEqual(set(ms), set(deploy.SEGMENTS))
        self.assertGreaterEqual(ms['receive'], 3.9)
        parts = ms['receive'] + ms['wait'] + ms['preprocess'] + ms['inference'] + ms['postprocess']
        self.assertAlmostEqual(ms['e2e'], parts, delta=0.01)

    def test_class_names_must_match_the_config(self):
        with self.assertRaises(ValueError):
            deploy.LiveDetector('cpu', Path('unused'), NAMES[::-1], 0.25, 0.45, net=FakeNet())
        with self.assertRaises(ValueError):
            deploy.LiveDetector('gpu', Path('unused'), NAMES, 0.25, 0.45, net=FakeNet())


class TickTests(unittest.TestCase):
    STAGE0 = {'missed': 6, 'ticks': 90000}

    def feed(self, tap, n, missed_at=(), gap_at=()):
        t = 100.0
        missed = 3
        for i in range(n):
            t += 0.04 if i in gap_at else 0.02
            missed += i in missed_at
            tap({'t': t, 'policy': 'walk', 'loop': {'missed': missed, 'hz': 50.0}},
                time.monotonic())

    def test_counts_missed_ticks_and_gaps(self):
        tap = deploy.TickTap()
        self.feed(tap, 3500, missed_at={100}, gap_at={200, 201})
        s = tap.summary(self.STAGE0)
        self.assertEqual(s['missed'], 1)
        self.assertEqual(s['stream_gaps'], 2)
        self.assertAlmostEqual(s['period_ms']['p50'], 20.0, places=3)
        self.assertTrue(s['pass'])

    def test_more_missed_than_stage0_fails(self):
        tap = deploy.TickTap()
        self.feed(tap, 3500, missed_at={10, 20})
        s = tap.summary(self.STAGE0)
        self.assertEqual(s['missed'], 2)
        self.assertFalse(s['pass'])

    def test_no_stream_fails(self):
        self.assertFalse(deploy.TickTap().summary(self.STAGE0)['pass'])

    def test_only_ticks_inside_the_window_count(self):
        tap = deploy.TickTap()
        self.feed(tap, 50, missed_at={5})          # before start(): homing, takeover
        tap.start()
        self.feed(tap, 3500)
        tap.stop()
        time.sleep(0.01)
        self.feed(tap, 50, missed_at={1, 2, 3})    # after stop(): head back, close
        s = tap.summary(self.STAGE0)
        self.assertEqual((s['ticks_seen'], s['missed']), (3500, 0))


class SmallTests(unittest.TestCase):
    def test_npu_load_parsing(self):
        self.assertEqual(deploy.parse_npu_load('NPU load:  Core0: 12%, Core1:  0%, Core2:  7%,\n'),
                         [12, 0, 7])
        self.assertEqual(deploy.parse_npu_load('permission denied'), [])

    def test_motion_schedule_repeats(self):
        motion = [['stand', 4.0], ['left', 8.0], ['stand', 4.0], ['right', 8.0]]
        self.assertEqual([deploy.motion_phase(motion, t) for t in (0, 3.9, 4, 12.5, 16.1, 24.0, 28.0)],
                         ['stand', 'stand', 'left', 'stand', 'right', 'stand', 'left'])

    def test_percentiles(self):
        p = deploy.percentiles(range(1, 101))
        self.assertEqual((p['n'], p['max']), (100, 100.0))
        self.assertAlmostEqual(p['p50'], 50.5)
        self.assertIsNone(deploy.percentiles([]))

    def test_config_has_the_deploy_section(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/vision/stage2.json').read_text())
        d = cfg['deploy']
        self.assertEqual(d['e2e_p99_ms_max'], 66.0)
        self.assertEqual({p for p, _ in d['motion']}, {'stand', 'left', 'right'})


class TableTests(unittest.TestCase):
    def test_cpus_text(self):
        self.assertEqual(deploy.cpus_text([4, 5, 6, 7]), '4-7')
        self.assertEqual(deploy.cpus_text([0, 2, 3, 7]), '0,2-3,7')
        self.assertEqual(deploy.cpus_text(None), '–')

    def test_table_lists_every_backend_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            seg = {'n': 1000, 'p50': 10.0, 'p90': 12.0, 'p99': 20.0, 'max': 25.0, 'mean': 10.5}
            for b, ok in (('cpu', False), ('int8', True)):
                (run / 'board' / b).mkdir(parents=True)
                (run / 'board' / b / 'summary.json').write_text(json.dumps({
                    'frames': 1000, 'ms': {s: seg for s in deploy.SEGMENTS}, 'processed_fps': 15.0,
                    'resources': {'self_cpu_pct_core_mean': 80.0, 'system_cpu_pct_mean': 20.0,
                                  'npu_load_pct_mean': [30.0, 0.0, 0.0]},
                    'robotd': {'missed': 0, 'ticks_expected': 3500}, 'pass': ok, 'outcome': 'completed',
                    'versions': {'python': '3.10'}, **({'cpus': [4, 5, 6, 7]} if ok else {})}))
            out = deploy.table(run)
            md = (run / 'deploy_table.md').read_text().splitlines()
        self.assertEqual(sorted(out['backends']), ['cpu', 'int8'])
        self.assertEqual(len(md), 4)
        self.assertIn('| int8 | 4-7 | 1000 |', md[3])
        self.assertIn('| cpu | – | 1000 |', md[2])
        self.assertEqual(out['backends']['int8']['cpus'], [4, 5, 6, 7])
        self.assertEqual(md[0].count('|'), md[1].count('|'))
        self.assertIn('30.0/0.0/0.0 %', md[3])
        self.assertTrue(md[3].endswith('| 是 |'))


if __name__ == '__main__':
    unittest.main()
