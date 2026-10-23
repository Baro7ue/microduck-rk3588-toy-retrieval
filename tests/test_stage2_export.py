"""Stage 2 export and test_walk helpers: box pairing, RKNN calibration frames, walk seeds."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from microduck_toy.stage2 import check_tools, datagen, walkrec

REPO = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / 'vision' / 'stage2' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(*rows):
    """(x0, y0, x1, y1, score, cls) rows -> decode_* style (boxes, scores, classes)."""
    a = np.array(rows, float).reshape(-1, 6)
    return a[:, :4], a[:, 4], a[:, 5].astype(int)


class PairTests(unittest.TestCase):
    def test_pairs_by_class_and_overlap(self):
        a = result((10, 10, 50, 50, 0.9, 0), (100, 100, 140, 130, 0.8, 2))
        b = result((100.5, 100, 140, 130, 0.79, 2), (11, 10, 50, 50, 0.88, 0), (200, 200, 210, 210, 0.3, 1))
        r = check_tools.pair_boxes(a, b)
        self.assertEqual(r['pairs'], 2)
        self.assertEqual(r['unpaired_a'], [])
        self.assertEqual(r['unpaired_b'], [2])
        self.assertAlmostEqual(r['box_px'], 1.0)
        self.assertAlmostEqual(r['conf'], 0.02)

    def test_other_class_or_low_overlap_does_not_pair(self):
        a = result((10, 10, 50, 50, 0.9, 0))
        self.assertEqual(check_tools.pair_boxes(a, result((10, 10, 50, 50, 0.9, 1)))['pairs'], 0)
        self.assertEqual(check_tools.pair_boxes(a, result((40, 40, 80, 80, 0.9, 0)))['pairs'], 0)

    def test_nms_tie_is_a_box_right_at_the_threshold(self):
        # Two w x 20 boxes overlapping by h rows: IoU = h / (40 - h) = 0.45 when h = 18 / 1.45.
        w = 40.0
        h_overlap = 0.45 * 40 / 1.45
        side = result((0, 0, w, 20, 0.9, 0), (0, 20 - h_overlap, w, 40 - h_overlap, 0.4, 0),
                      (0, 100, w, 120, 0.3, 0))
        self.assertTrue(check_tools.nms_tie(1, side, 0.45))
        self.assertFalse(check_tools.nms_tie(2, side, 0.45))
        self.assertFalse(check_tools.nms_tie(0, side, 0.45))   # nothing scores higher


class CalibrationTests(unittest.TestCase):
    def test_target_frames_only_in_proportion_and_repeatable(self):
        convert = load_script('convert_rknn')
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / 'meta').mkdir()
            rows = []
            for i in range(200):
                view = 'search' if i % 4 else 'approach'          # 150 search, 50 approach
                floor = 'grid' if i % 2 else 'wood'
                names = ['croc'] if i % 10 == 0 else ['bus', 'croc'] if i % 3 else ['tray']
                rows.append({'id': f'train_{i}', 'view': view, 'floor': floor,
                             'objects': [{'name': n} for n in names]})
            (data / 'meta' / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            ids, strata = convert.calibration(data, ['bus', 'henry', 'ketchup'], n=40)
            again, _ = convert.calibration(data, ['bus', 'henry', 'ketchup'], n=40)
        self.assertEqual(ids, again)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(v['chosen'] for v in strata.values()), 40)
        by_id = {r['id']: r for r in rows}
        self.assertTrue(all(any(o['name'] == 'bus' for o in by_id[i]['objects']) for i in ids))
        pool = sum(v['pool'] for v in strata.values())
        for v in strata.values():
            self.assertLessEqual(abs(v['chosen'] - 40 * v['pool'] / pool), 1.0)


class WalkSeedTests(unittest.TestCase):
    def test_seeds_follow_the_floor_and_never_overlap(self):
        cfg, stage1 = datagen.load_configs()
        grid = walkrec.episode_seeds(cfg, stage1, 'grid', 5)
        wood = walkrec.episode_seeds(cfg, stage1, 'wood', 5)
        self.assertEqual(grid, walkrec.episode_seeds(cfg, stage1, 'grid', 5))
        self.assertFalse(set(grid) & set(wood))
        for seed in grid:
            scene = datagen.sample(seed, walkrec.SPLIT, cfg, stage1)
            self.assertEqual(scene['floor'], 'grid')
            self.assertEqual(scene['view'], 'search')
            self.assertNotIn(scene['focus'], (None, 'tray'))


if __name__ == '__main__':
    unittest.main()
