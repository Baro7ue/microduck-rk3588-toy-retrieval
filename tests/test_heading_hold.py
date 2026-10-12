# Moved 2026-10-11 from work/tests/test_heading_hold.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Check feedback sign, frame handling, and command ownership transitions."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from microduck_toy.control.heading_hold import HeadingConfig, HeadingHold, wrap_angle, yaw_rate_from_imu


class HeadingTests(unittest.TestCase):
    def test_imu_yaw_with_roll_and_pitch(self):
        roll, pitch, yaw_rate = 0.4, 0.3, 0.7
        gravity = (math.sin(pitch), -math.sin(roll)*math.cos(pitch),
                   -math.cos(roll)*math.cos(pitch))
        # Pure world-z angular rotation, expressed in the tilted body frame.
        gyro = tuple(-yaw_rate*g for g in gravity)
        self.assertAlmostEqual(yaw_rate_from_imu(gyro, gravity), yaw_rate)

    def test_right_correction_after_left_drift(self):
        controller = HeadingHold()
        controller.update((0, 0, 0), (0, 0, -1), .02, (.3, 0, 0))
        output = controller.update((0, 0, .1), (0, 0, -1), .02, (.3, 0, 0))
        self.assertLess(output, 0)
        self.assertTrue(controller.active)

    def test_manual_turn_stop_and_restart(self):
        controller = HeadingHold(HeadingConfig(ki=.05))
        controller.update((0, 0, 0), (0, 0, -1), .02, (.3, 0, 0))
        self.assertEqual(controller.update((0, 0, .2), (0, 0, -1), .02,
                                          (.3, 0, 1.5)), 1.5)
        self.assertFalse(controller.active)
        self.assertEqual(controller.update((0, 0, .2), (0, 0, -1), .02,
                                          (0, 0, 0)), 0)
        output = controller.update((0, 0, .2), (0, 0, -1), .02, (.3, 0, 0))
        self.assertAlmostEqual(output, 0)
        self.assertTrue(controller.active)

    def test_pause_gap_and_reverse_relock(self):
        controller = HeadingHold()
        controller.update((0, 0, 0), (0, 0, -1), .02, (.3, 0, 0))
        controller.update((0, 0, .3), (0, 0, -1), .02, (.3, 0, 0))
        self.assertAlmostEqual(controller.update((0, 0, .3), (0, 0, -1), .4,
                                                (.3, 0, 0)), 0)
        self.assertAlmostEqual(controller.update((0, 0, .3), (0, 0, -1), .02,
                                                (-.3, 0, 0)), 0)

    def test_angle_wrap_and_saturation(self):
        self.assertAlmostEqual(wrap_angle(math.pi + .1), -math.pi + .1)
        controller = HeadingHold(HeadingConfig(kp=100, ki=.1))
        controller.update((0, 0, 0), (0, 0, -1), .02, (.3, 0, 0))
        output = controller.update((0, 0, 1), (0, 0, -1), .02, (.3, 0, 0))
        self.assertEqual(output, -.3)
        self.assertEqual(controller.integral, 0)

    def test_invalid_configuration_and_measurement(self):
        for values in ({'kp':-1}, {'ki':float('nan')}, {'period_s':0}):
            with self.assertRaises(ValueError):
                HeadingConfig(**values)
        controller = HeadingHold()
        controller.update((0, 0, 0), (0, 0, -1), .02, (.3, 0, 0))
        self.assertEqual(controller.update((0, 0, float('nan')), (0, 0, -1),
                                          .02, (.3, 0, 0)), 0)
        self.assertFalse(controller.active)


if __name__ == '__main__':
    unittest.main()
