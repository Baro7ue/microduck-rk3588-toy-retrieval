"""Check runtime PI sampling, feedback loss, and actual outgoing command sign."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heading_hold import HeadingConfig
from probe_runtime_walk import Drive


def state(timestamp, gyro_z=0.0):
    return {'t_ns': timestamp, 'imu': {'gyro': [0.0, 0.0, gyro_z]},
            'safety': {'gravity': [0.0, 0.0, -1.0]}}


class RuntimeHeadingTests(unittest.TestCase):
    def make_drive(self, path='unused'):
        return Drive(path, 0.1, HeadingConfig(kp=1.0, ki=0.05, period_s=0.05))

    def test_network_sample_period_keeps_reference_and_corrects_left_drift(self):
        drive = self.make_drive()
        drive.update_feedback(state(1_000_000_000))
        drive.update_feedback(state(1_040_000_000, 0.1))
        drive.update_feedback(state(1_100_000_000, 0.1))
        self.assertLess(drive.vyaw, 0.0)
        self.assertAlmostEqual(drive.heading.heading, 0.01)
        self.assertAlmostEqual(drive.heading.reference, 0.0)
        self.assertLess(drive.heading.integral, 0.0)
        drive.stop()
        self.assertFalse(drive.heading.active)

    def test_duplicate_sample_does_not_keep_feedback_alive_and_gap_fails(self):
        drive = self.make_drive()
        with patch('probe_runtime_walk.time.monotonic', return_value=1.0):
            drive.update_feedback(state(1_000_000_000))
        with patch('probe_runtime_walk.time.monotonic', return_value=2.0):
            drive.update_feedback(state(1_000_000_000))
        self.assertEqual(drive.feedback_at, 1.0)
        with self.assertRaises(RuntimeError):
            drive.update_feedback(state(1_100_000_000, 0.1))

    def test_full_rate_feedback_with_delayed_tick_preserves_heading_reference(self):
        drive = self.make_drive()
        drive.update_feedback(state(1_000_000_000, 0.1))
        reference = drive.heading.reference
        # Actual sensor intervals, including a delayed tick; none exceeds
        # the unchanged 75 ms limit. The first-sample offset cancels out.
        for stamp in (1_020_000_000, 1_055_000_000, 1_115_000_000):
            drive.update_feedback(state(stamp, 0.1))
        self.assertAlmostEqual(drive.heading.reference, reference)
        self.assertAlmostEqual(drive.heading.heading - reference, 0.0115)
        self.assertAlmostEqual(drive.heading.error, -0.0115)
        self.assertLess(drive.vyaw, 0.0)
        with self.assertRaises(RuntimeError):
            drive.update_feedback(state(1_215_000_000, 0.1))

    def test_real_sender_emits_correction_then_zero_on_feedback_loss(self):
        with tempfile.TemporaryDirectory(prefix='md-pi-') as directory:
            path = str(Path(directory) / 'fixture.sock')
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(path)
            listener.listen(1)
            listener.settimeout(2)
            packets = []

            def receive():
                with listener.accept()[0] as client:
                    with client.makefile('rb') as stream:
                        for line in stream:
                            packets.append(json.loads(line))

            receiver = threading.Thread(target=receive)
            receiver.start()
            drive = self.make_drive(path)
            drive.update_feedback(state(1_000_000_000))
            drive.update_feedback(state(1_050_000_000, 0.1))
            drive.thread.start()
            drive.thread.join(timeout=2)
            drive.stop()
            receiver.join(timeout=2)
            listener.close()
            self.assertFalse(receiver.is_alive())
            self.assertIsNotNone(drive.error)
            self.assertGreaterEqual(len(packets), 2)
            self.assertLess(packets[0]['params']['vyaw'], 0.0)
            self.assertEqual(packets[0]['params']['vx'], 0.1)
            self.assertEqual(packets[-1]['params'], {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0})


if __name__ == '__main__':
    unittest.main()
