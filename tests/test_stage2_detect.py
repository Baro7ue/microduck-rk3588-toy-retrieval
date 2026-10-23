"""Stage 2 detectors: single-output decoding + NMS, and the colour-threshold baseline."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from microduck_toy.vision import detect


def raw_from(anchors, nc=6, n=2100):
    """[1, 4+nc, n] with the given (cx, cy, w, h, cls, score) anchors and zeros elsewhere."""
    raw = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(anchors):
        raw[0, :4, i] = (cx, cy, w, h)
        raw[0, 4 + c, i] = s
    return raw


class DecodeTests(unittest.TestCase):
    def test_overlapping_boxes_of_one_class_keep_the_best(self):
        boxes, scores, classes = detect.decode_single(
            raw_from([(100, 100, 20, 10, 0, 0.9), (101, 100, 20, 10, 0, 0.8)]), 0.25, 0.45)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [90, 95, 110, 105])
        self.assertAlmostEqual(scores[0], 0.9, places=6)

    def test_overlapping_boxes_of_two_classes_both_stay(self):
        _b, _s, classes = detect.decode_single(
            raw_from([(100, 100, 20, 10, 0, 0.9), (101, 100, 20, 10, 3, 0.8)]), 0.25, 0.45)
        self.assertEqual(sorted(classes.tolist()), [0, 3])

    def test_threshold_and_clipping(self):
        boxes, scores, _c = detect.decode_single(
            raw_from([(5, 100, 20, 10, 1, 0.3), (200, 200, 10, 10, 2, 0.2)]), 0.25, 0.45)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][0], 0.0)


class ColourTests(unittest.TestCase):
    CFG = {'score_half_px': 20.0,
           'classes': {'bus': {'ranges_hsv': [[[17, 150, 90], [31, 255, 255]]], 'min_area_px': 4},
                       'ketchup': {'ranges_hsv': [[[0, 140, 60], [6, 255, 255]],
                                                  [[172, 140, 60], [180, 255, 255]]],
                                   'min_area_px': 4}}}

    def test_one_box_per_blob_with_exact_bounds(self):
        img = np.full((320, 320, 3), 128, np.uint8)
        img[100:110, 50:80] = (0, 220, 240)      # yellow (BGR)
        img[200:204, 10:13] = (0, 220, 240)      # second, smaller yellow blob
        img[150:160, 150:170] = (20, 20, 200)    # red
        dets = detect.ColourBoxes(self.CFG)(img, conf=0.0)
        got = sorted((d['cls'], d['xyxy']) for d in dets)
        self.assertEqual(got, [('bus', [10.0, 200.0, 13.0, 204.0]), ('bus', [50.0, 100.0, 80.0, 110.0]),
                               ('ketchup', [150.0, 150.0, 170.0, 160.0])])
        big = [d for d in dets if d['xyxy'][0] == 50.0][0]
        self.assertAlmostEqual(big['conf'], 300 / 320)

    def test_blobs_below_the_minimum_are_dropped(self):
        img = np.full((320, 320, 3), 128, np.uint8)
        img[10:11, 10:13] = (0, 220, 240)        # 3 px
        self.assertEqual(detect.ColourBoxes(self.CFG)(img, conf=0.0), [])

    def test_hsv_matches_opencv(self):
        img = np.zeros((1, 1, 3), np.uint8)
        img[0, 0] = (0, 220, 240)
        h, s, v = detect.ColourBoxes.hsv(img)[0, 0]
        self.assertEqual((int(v), int(s)), (240, 255))
        self.assertTrue(17 <= int(h) <= 31)
        self.assertEqual(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[0, 0].tolist(), [h, s, v])


if __name__ == '__main__':
    unittest.main()
