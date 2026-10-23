"""Stage 2 detector input: window / letterbox mappings, the horizon row, the UYVY round trip."""
from __future__ import annotations

import math
import unittest

import numpy as np

from microduck_toy.vision import uyvy, views
from microduck_toy.vision.yolo_postprocess import letterbox


def camera_pitched(pitch_down: float) -> np.ndarray:
    """OpenCV camera axes -> level frame for a camera looking along +x, tilted down by pitch."""
    c, s = math.cos(pitch_down), math.sin(pitch_down)
    forward = np.array([c, 0.0, -s])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


class HorizonTests(unittest.TestCase):
    def test_level_camera_puts_the_horizon_in_the_middle(self):
        self.assertAlmostEqual(views.horizon_row(camera_pitched(0.0)), 320.0, places=6)

    def test_pitch_down_moves_the_horizon_up_by_f_tan(self):
        for pitch in (0.1, 0.3, 0.6):
            self.assertAlmostEqual(views.horizon_row(camera_pitched(pitch)),
                                   320 - views.FOCAL_PX * math.tan(pitch), places=6)

    def test_window_top_is_margin_above_the_horizon_and_stays_in_the_frame(self):
        self.assertEqual(views.window_top(185.4), 155)
        self.assertEqual(views.window_top(320.0), 280)       # clamped: 290 would run off the bottom
        self.assertEqual(views.window_top(10.0), 0)


class MappingTests(unittest.TestCase):
    def test_box_round_trip(self):
        for m in (views.window_mapping(185.4), views.letterbox_mapping()):
            box = [100.0, 300.0, 140.0, 330.0]
            back = views.to_frame_box(views.to_input_box(box, m, clip=False), m)
            np.testing.assert_allclose(back, box, atol=1e-9)

    def test_window_gain_is_1_78_times_the_letterbox_gain(self):
        self.assertAlmostEqual(views.window_mapping(300).gain / views.letterbox_mapping().gain,
                               320 / 360 / 0.5)

    def test_letterbox_matches_the_stage0_detector(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (640, 360, 3), dtype=np.uint8)
        mine = views.tensor(views.to_input(frame, views.letterbox_mapping()))
        old, gain, pad = letterbox(frame)
        np.testing.assert_array_equal(mine, old)
        self.assertEqual((gain, pad), (0.5, (70, 0)))

    def test_window_input_is_the_scaled_crop(self):
        frame = np.zeros((640, 360, 3), np.uint8)
        frame[155:515] = 200
        img = views.to_input(frame, views.window_mapping(185.4))
        self.assertEqual(img.shape, (320, 320, 3))
        self.assertTrue((img == 200).all())

    def test_boxes_outside_the_window_are_dropped_or_clipped(self):
        m = views.window_mapping(185.4)             # rows 155-515
        self.assertIsNone(views.to_input_box([10, 100, 20, 150], m))
        clipped = views.to_input_box([10, 500, 20, 530], m)
        self.assertAlmostEqual(clipped[3], 320.0)


class UyvyTests(unittest.TestCase):
    def test_grey_survives_the_round_trip(self):
        # Within 1: upstream's encoder casts with truncation (127.99... -> 127), as the sim does.
        img = np.full((640, 360, 3), 128, np.uint8)
        diff = uyvy.round_trip_upright(img).astype(int) - img
        self.assertLessEqual(np.abs(diff).max(), 1)

    def test_chroma_is_shared_by_upright_row_pairs(self):
        # Raw pixel pairs are neighbouring raw columns = neighbouring upright rows (2k, 2k+1), so
        # a red/blue edge between rows 10 and 11 (one pair) bleeds colour into row 10, and an edge
        # between rows 11 and 12 (two pairs) does not.
        red, blue = (200, 30, 30), (30, 30, 200)
        inside = np.zeros((640, 360, 3), np.uint8)
        inside[:, :] = red
        inside[11:, :] = blue
        self.assertLess(int(uyvy.round_trip_upright(inside)[10, 100, 0]), 160)
        between = np.zeros((640, 360, 3), np.uint8)
        between[:, :] = red
        between[12:, :] = blue
        out = uyvy.round_trip_upright(between)
        self.assertLess(np.abs(out[11, 100].astype(int) - red).max(), 3)


if __name__ == '__main__':
    unittest.main()
