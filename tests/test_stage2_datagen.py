"""Stage 2 data generator: scenes from seeds (no MuJoCo), and one rendered frame per view."""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from microduck_toy.stage2 import datagen

try:
    import mujoco  # noqa: F401
except ImportError:  # rendering needs the microduck_rl environment (MD_PY)
    mujoco = None

REPO = Path(__file__).resolve().parents[1]
CFG, STAGE1 = datagen.load_configs()


class SampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = [datagen.sample(s, 'train', CFG, STAGE1) for s in range(100000, 100400)]

    def test_same_seed_same_scene(self):
        self.assertEqual(datagen.sample(100123, 'train', CFG, STAGE1), self.scenes[123])

    def test_views_and_floors_follow_the_config(self):
        n = len(self.scenes)
        count = {v: sum(s['view'] == v for s in self.scenes) for v in ('search', 'empty', 'approach')}
        self.assertAlmostEqual(count['approach'] / n, 1 - CFG['datagen']['p_search'], delta=0.08)
        self.assertAlmostEqual(count['empty'] / n, 0.1, delta=0.05)
        self.assertEqual({s['floor'] for s in self.scenes}, {'grid', 'wood'})
        unseen = datagen.sample(400000, 'test_unseen', CFG, STAGE1)
        self.assertEqual(unseen['floor'], 'carpet')

    def test_toys_keep_the_stage1_layout_rules(self):
        lay, arena = STAGE1['layout'], STAGE1['arena']
        lim = arena['half_size_m'] - lay['wall_clear_m'] + 1e-6
        tray = arena['tray']['centre_m']
        for s in self.scenes:
            pts = [t[:2] for t in s['toys'].values()]
            for p in pts:
                self.assertLessEqual(max(abs(p[0]), abs(p[1])), lim)
                self.assertGreaterEqual(math.dist(p, tray), lay['tray_clear_m'] - 1e-6)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    self.assertGreaterEqual(math.dist(pts[i], pts[j]), lay['toy_gap_m'] - 1e-4)

    def test_search_focus_is_in_view_and_empty_views_have_no_toy_in_view(self):
        for s in self.scenes:
            if s['view'] == 'search' and s['focus'] != 'tray':
                self.assertTrue(datagen.in_view(s['robot'], s['toys'][s['focus']]))
            if s['view'] == 'empty':
                self.assertFalse(any(datagen.in_view(s['robot'], t, math.radians(30))
                                     for t in s['toys'].values()))

    def test_a_good_share_of_search_focus_toys_is_far(self):
        far = [math.dist(s['robot'][:2], s['toys'][s['focus']][:2])
               for s in self.scenes if s['view'] == 'search' and s['focus'] != 'tray']
        share = sum(d >= 1.2 for d in far) / len(far)
        self.assertGreater(share, 0.35)

    def test_crowding_puts_several_toys_in_view(self):
        for view, reach, need in (('search', 2.2, 2.5), ('approach', 0.95, 2.0)):
            frames = [s for s in self.scenes if s['view'] == view]
            seen = [sum(datagen.in_view(s['robot'], t, math.radians(21))
                        and math.dist(s['robot'][:2], t[:2]) <= reach for t in s['toys'].values())
                    for s in frames]
            self.assertGreater(sum(seen) / len(seen), need, view)

    def test_far_split_has_far_search_views_only(self):
        lo, hi = CFG['datagen']['search']['far_m']
        for seed in range(500000, 500060):
            s = datagen.sample(seed, 'test_far', CFG, STAGE1)
            self.assertEqual(s['view'], 'search')
            self.assertNotEqual(s['focus'], 'tray')
            d = math.dist(s['robot'][:2], s['toys'][s['focus']][:2])
            self.assertTrue(lo - 1e-6 <= d <= hi + 1e-6, d)

    def test_far_focus_distances_reach_the_far_end(self):
        far = [math.dist(s['robot'][:2], s['toys'][s['focus']][:2])
               for s in self.scenes if s['view'] == 'search' and s['focus'] != 'tray']
        self.assertGreater(sum(d >= 1.7 for d in far) / len(far), 0.08)

    def test_approach_toy_sits_where_the_config_says(self):
        a = CFG['datagen']['approach']
        for s in self.scenes:
            if s['view'] != 'approach':
                continue
            x, y, yaw = s['robot']
            tx, ty = s['toys'][s['focus']][:2]
            fwd = math.cos(yaw) * (tx - x) + math.sin(yaw) * (ty - y)
            left = -math.sin(yaw) * (tx - x) + math.cos(yaw) * (ty - y)
            self.assertTrue(a['forward_m'][0] - 1e-3 <= fwd <= a['forward_m'][1] + 1e-3)
            self.assertTrue(a['left_m'][0] - 1e-3 <= left <= a['left_m'][1] + 1e-3)


@unittest.skipIf(mujoco is None, 'mujoco not importable')
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = datagen.Renderer(REPO / 'third_party' / 'microduck_rl', CFG, STAGE1)

    def first(self, view):
        for seed in range(100000, 100100):
            s = datagen.sample(seed, 'train', CFG, STAGE1)
            if s['view'] == view and s['floor'] == 'grid':
                return s
        self.fail(view)

    def test_frames_are_deterministic_and_labels_are_inside(self):
        for view in ('search', 'approach'):
            scene = self.first(view)
            img, labels, meta = self.r.render(scene)
            img2, labels2, _ = self.r.render(scene)
            self.assertEqual(img.shape, (320, 320, 3))
            self.assertTrue((img == img2).all())
            self.assertEqual(labels, labels2)
            self.assertGreater(len(labels), 0)
            for _c, cx, cy, w, h in labels:
                self.assertTrue(0 <= cx - w / 2 and cx + w / 2 <= 1 and 0 <= cy - h / 2
                                and cy + h / 2 <= 1)
            self.assertEqual(meta['mapping']['kind'], 'letterbox' if view == 'approach' else 'window')

    def test_ranging_on_oracle_boxes_lands_on_the_toy(self):
        from microduck_toy.control import ranging
        cam = ranging.CameraModel.from_config(STAGE1['camera'])
        scene = self.first('approach')
        _img, _labels, meta = self.r.render(scene)
        checked = 0
        for o in meta['objects']:
            if o['name'] == 'tray' or o['cut_frame']:
                continue
            fwd, left = ranging.ground_point(cam, meta['state'], o['xyxy_frame'], o['half_z_m'])
            self.assertLess(math.hypot(fwd - o['fwd_m'], left - o['left_m']), 0.015)
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
