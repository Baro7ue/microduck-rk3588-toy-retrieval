"""Stage 2 detectors: single- and split-output decoding + NMS, RKNN input, the colour baseline."""
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


def split_outputs(rng, nc=6, hot=()):
    """Random split outputs (box logits, sigmoid scores, clamped sums) with mostly low scores;
    hot = (stride index, y, x, cls, score) cells to raise above the threshold."""
    outs = []
    for k, stride in enumerate(detect.STRIDES):
        g = 320 // stride
        box = rng.normal(0, 2, (1, 64, g, g)).astype(np.float32)
        cls = (rng.random((1, nc, g, g)) * 0.02).astype(np.float32)
        for kk, y, x, c, score in hot:
            if kk == k:
                cls[0, c, y, x] = score
        outs += [box, cls, np.clip(cls.sum(1, keepdims=True), 0, 1)]
    return outs


def single_from_split(outs):
    """What Ultralytics' own head computes from the same tensors: DFL, anchors, xywh * stride."""
    cols = []
    for k, stride in enumerate(detect.STRIDES):
        box, cls = outs[3 * k].astype(np.float64), outs[3 * k + 1]
        g = box.shape[2]
        logits = box[0].reshape(4, 16, g * g)
        prob = np.exp(logits - logits.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        dist = (prob * np.arange(16)[None, :, None]).sum(1)
        ys, xs = np.divmod(np.arange(g * g), g)
        ax, ay = xs + 0.5, ys + 0.5
        x0, y0, x1, y1 = ax - dist[0], ay - dist[1], ax + dist[2], ay + dist[3]
        xywh = np.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)) * stride
        cols.append(np.concatenate((xywh, cls[0].reshape(cls.shape[1], -1)), 0))
    return np.concatenate(cols, 1)[None].astype(np.float32)


class SplitDecodeTests(unittest.TestCase):
    def test_one_cell_with_sharp_bins_gives_the_expected_box(self):
        outs = split_outputs(np.random.default_rng(0), hot=[(1, 4, 7, 2, 0.8)])
        box = np.full((64,), -30.0, np.float32)
        for side, bin_ in enumerate((2, 1, 3, 5)):   # left top right bottom, in grid cells
            box[side * 16 + bin_] = 30.0
        outs[3][0, :, 4, 7] = box
        boxes, scores, classes = detect.decode_split(outs, 0.25, 0.45)
        self.assertEqual(classes.tolist(), [2])
        self.assertAlmostEqual(scores[0], 0.8, places=6)
        ax, ay = 7.5, 4.5   # stride 16
        np.testing.assert_allclose(boxes[0], [(ax - 2) * 16, (ay - 1) * 16, (ax + 3) * 16,
                                              (ay + 5) * 16], atol=1e-6)

    def test_same_boxes_as_the_single_output_decoder(self):
        rng = np.random.default_rng(1)
        for trial in range(20):
            hot = [(int(rng.integers(3)), int(rng.integers(10)), int(rng.integers(10)),
                    int(rng.integers(6)), float(rng.uniform(0.2, 0.99))) for _ in range(8)]
            outs = split_outputs(rng, hot=hot)
            for conf in (0.25, 0.001):
                a = detect.decode_split(outs, conf, 0.45)
                b = detect.decode_single(single_from_split(outs), conf, 0.45)
                self.assertEqual(len(a[0]), len(b[0]), (trial, conf))
                np.testing.assert_array_equal(a[2], b[2])
                np.testing.assert_allclose(a[1], b[1], atol=1e-6)
                np.testing.assert_allclose(a[0], b[0], atol=1e-3)

    def test_empty_cells_are_skipped_and_nothing_found_is_empty(self):
        outs = split_outputs(np.random.default_rng(2))
        boxes, scores, classes = detect.decode_split(outs, 0.25, 0.45)
        self.assertEqual((boxes.shape, scores.shape, classes.shape), ((0, 4), (0,), (0,)))

    def test_a_low_sum_never_hides_a_cell_above_the_threshold(self):
        outs = split_outputs(np.random.default_rng(3), hot=[(0, 20, 20, 1, 0.26)])
        outs[2][0, 0, 20, 20] = 0.26 - detect.SUM_MARGIN / 2   # INT8-rounded sum below the max
        self.assertEqual(len(detect.decode_split(outs, 0.25, 0.45)[0]), 1)

    def test_wrong_output_count_or_shape_is_refused(self):
        outs = split_outputs(np.random.default_rng(4))
        with self.assertRaises(ValueError):
            detect.decode_split(outs[:6], 0.25, 0.45)
        outs[0] = outs[0][:, :32]
        with self.assertRaises(ValueError):
            detect.decode_split(outs, 0.25, 0.45)


class RknnInputTests(unittest.TestCase):
    def test_runtime_gets_rgb_uint8_nhwc(self):
        seen = {}

        class Runtime:
            def inference(self, inputs, data_format):
                seen['x'], seen['fmt'] = inputs[0], data_format
                return split_outputs(np.random.default_rng(5), hot=[(2, 3, 3, 4, 0.9)])

        img = np.zeros((320, 320, 3), np.uint8)
        img[..., 0] = 10   # blue in BGR
        dets = detect.RknnYolo(Runtime(), ['bus', 'henry', 'ketchup', 'croc', 'thomas', 'tray'])(img)
        self.assertEqual(seen['x'].shape, (1, 320, 320, 3))
        self.assertEqual(seen['x'].dtype, np.uint8)
        self.assertEqual(seen['x'][0, 0, 0].tolist(), [0, 0, 10])
        self.assertEqual(seen['fmt'], ['nhwc'])
        self.assertEqual([d['cls'] for d in dets], ['thomas'])


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
