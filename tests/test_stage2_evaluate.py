"""Stage 2 evaluator on hand-made frames: AP, recall vs distance, radius, wrong grab, H1 ranging."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from microduck_toy.stage2 import evaluate as ev

# A real approach-view camera state (trial render, head 0.6): the H1 ranging needs one.
STATE = {'frames': {'camera': {'pos': [0.082272, -9.1e-05, 0.086925],
                               'quat': [0.3239711, -0.6285242, 0.6285242, -0.3239711]}},
         'imu': {'quat': [0.9846604, -0.0058837, -0.0090195, 0.1741494]},
         'odom': {'position': [0.0, 0.0, 0.117215]}}
LETTERBOX = {'kind': 'letterbox', 'x0': 0.0, 'y0': 0.0, 'gain': 0.5, 'pad_x': 70.0, 'pad_y': 0.0}
WINDOW = {'kind': 'window', 'x0': 0.0, 'y0': 200.0, 'gain': 320 / 360, 'pad_x': 0.0, 'pad_y': 0.0}


def obj(name, box_in, dist, fwd=None, left=0.0, cut=False, frame=None):
    return {'name': name, 'xyxy_input': box_in, 'xyxy_frame': frame or box_in, 'cut_frame': cut,
            'dist_cam_m': dist, 'fwd_m': dist if fwd is None else fwd, 'left_m': left,
            'half_z_m': 0.0124}


def frame(i, view, objects, dropped=(), mapping=None):
    return {'id': f'f{i}', 'view': view, 'objects': list(objects), 'dropped': list(dropped),
            'mapping': mapping or (LETTERBOX if view == 'approach' else WINDOW), 'state': STATE}


def det(cls, box, conf=0.9):
    return {'cls': cls, 'conf': conf, 'xyxy': box}


class ApTests(unittest.TestCase):
    def test_perfect_ranking_is_1_and_a_miss_halves_it(self):
        self.assertAlmostEqual(ev.ap_101(np.array([1, 1]), np.array([0.9, 0.8]), 2), 1.0)
        self.assertAlmostEqual(ev.ap_101(np.array([1]), np.array([0.9]), 2), 51 / 101, places=6)

    def test_a_false_positive_above_the_hit_costs_precision(self):
        self.assertAlmostEqual(ev.ap_101(np.array([0, 1]), np.array([0.9, 0.8]), 1), 0.5)

    def test_boxes_on_too_small_objects_are_ignored_not_false(self):
        gts, ign = [[0, 0, 10, 10]], [[50, 50, 52, 53]]
        pairs = ev.match_class(gts, ign, [det('bus', [0, 0, 10, 10]), det('bus', [50, 50, 52, 53])],
                               0.5)
        self.assertEqual(pairs, [(0.9, 1)])

    def test_each_truth_box_matches_once(self):
        pairs = ev.match_class([[0, 0, 10, 10]], [], [det('bus', [0, 0, 10, 10], 0.9),
                                                       det('bus', [0, 0, 10, 9.5], 0.8)], 0.5)
        self.assertEqual(pairs, [(0.9, 1), (0.8, 0)])


class RadiusTests(unittest.TestCase):
    def test_run_stops_at_the_first_bad_bin_and_skips_thin_ones(self):
        curve = {'0.4': [9, 0, 0], '0.5': [10, 10, 0], '0.6': [3, 0, 0], '0.7': [10, 8, 0],
                 '0.8': [10, 7, 0], '0.9': [10, 10, 0]}
        r = ev.radius(curve)
        self.assertEqual(r['radius_m'], 0.8)
        self.assertEqual(r['stopped_at']['bin_m'], 0.8)

    def test_no_good_bin_gives_0(self):
        self.assertEqual(ev.radius({'0.5': [10, 1, 0]})['radius_m'], 0.0)


class EvaluateTests(unittest.TestCase):
    def run_eval(self, frames, preds, space='ours'):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / 'meta').mkdir()
            (data / 'meta' / 'test.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
            pred = data / 'pred.jsonl'
            pred.write_text(''.join(json.dumps({'id': f['id'], 'dets': d}) + '\n'
                                    for f, d in zip(frames, preds)))
            return ev.evaluate(data, 'test', pred, space)

    def scene(self):
        bus = obj('bus', [100, 100, 130, 110], 0.85)
        croc = obj('croc', [200, 100, 220, 110], 0.95)
        henry_far = obj('henry', [50, 90, 62, 94], 1.65)
        # H1 frame: the bus 0.25 m ahead, its frame box is the letterbox inverse of the input box.
        near_in = [160.0, 200.0, 200.0, 220.0]
        near = obj('bus', near_in, 0.2, fwd=0.25, frame=[180.0, 400.0, 260.0, 440.0])
        frames = [frame(0, 'search', [bus, croc, henry_far]), frame(1, 'approach', [near]),
                  frame(2, 'search', [])]
        perfect = [[det('bus', bus['xyxy_input']), det('croc', croc['xyxy_input']),
                    det('henry', henry_far['xyxy_input'])], [det('bus', near_in)], []]
        return frames, perfect

    def test_perfect_predictions(self):
        frames, preds = self.scene()
        s = self.run_eval(frames, preds)
        self.assertEqual(s['ap']['map50'], 1.0)
        self.assertEqual(s['wrong_grab']['all']['wrong'], 0)
        self.assertEqual(s['recall_by_distance']['search']['henry']['1.6'], [1, 1, 1])
        h = s['h1_ranging']
        self.assertEqual(h['n'], 1)
        self.assertLess(h['extra_rms_cm'], 1e-6)

    def test_croc_called_henry_is_a_wrong_grab_and_shows_in_the_confusion(self):
        frames, preds = self.scene()
        preds[0] = [det('bus', frames[0]['objects'][0]['xyxy_input']),
                    det('henry', frames[0]['objects'][1]['xyxy_input'], 0.95)]
        s = self.run_eval(frames, preds)
        self.assertEqual(s['wrong_grab']['by_target']['henry']['wrong'], 1)
        self.assertEqual(s['confusion']['mid']['croc'], {'henry': 1})
        self.assertEqual(s['confusion']['far']['henry'], {'missed': 1})
        self.assertEqual(s['recall_by_distance']['search']['henry']['1.6'], [1, 0, 0])

    def test_a_box_shifted_4_px_in_the_input_moves_the_ranged_point(self):
        frames, preds = self.scene()
        preds[1] = [det('bus', [164.0, 200.0, 204.0, 220.0])]   # 8 px right in the frame
        h = self.run_eval(frames, preds)['h1_ranging']
        self.assertGreater(h['extra_rms_cm'], 0.1)
        self.assertLess(h['extra_mean_cm'][1], 0)                # right of the oracle = less left

    def test_coco_space_scores_henry_and_thomas_as_train(self):
        frames, _ = self.scene()
        frames[0]['objects'].append(obj('thomas', [250, 100, 270, 110], 0.9))
        preds = [[det('train', frames[0]['objects'][3]['xyxy_input'])], [], []]
        s = self.run_eval(frames, preds, 'coco')
        self.assertEqual(set(s['ap']['per_class']), {'bus', 'train', 'bottle'})
        self.assertEqual(s['recall_by_distance']['search']['thomas']['0.9'], [1, 1, 1])
        self.assertEqual(s['wrong_grab']['by_target']['henry']['wrong'], 1)


if __name__ == '__main__':
    unittest.main()
