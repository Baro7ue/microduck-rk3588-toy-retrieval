"""Stage 1 field pieces that need MuJoCo but no robotd: toys, layouts, oracle boxes."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

try:
    import mujoco
    from microduck_toy.scene import layout as layout_lib
    from microduck_toy.scene import render as scene_render
    from microduck_toy.scene import toys as toy_lib
except ImportError:  # the test needs the microduck_rl environment (MD_PY)
    mujoco = None

REPO = Path(__file__).resolve().parents[1]
STAGE1 = json.loads((REPO / 'configs/control/stage1.json').read_text())
FACTS = json.loads((REPO / 'configs/control/stage0_facts.json').read_text())


@unittest.skipIf(mujoco is None, 'mujoco not importable')
class ToyTests(unittest.TestCase):
    def test_box_sizes_match_the_stage0_toy_table(self):
        rows = FACTS['toys']['targets'] + FACTS['toys']['distractors']
        self.assertEqual([r['name'] for r in rows], list(toy_lib.catalogue()))
        for row in rows:
            toy = toy_lib.catalogue()[row['name']]
            size_cm = [round(200 * h, 2) for h in toy.half]
            for got, want in zip(size_cm, row['size_cm']):
                self.assertAlmostEqual(got, want, delta=0.02, msg=row['name'])
            self.assertAlmostEqual(toy.mass_kg * 1000, row['mass_g'], places=3)
            self.assertLessEqual(max(size_cm), 8.0 + 1e-6)          # <= 8 cm, long axis on x
            self.assertEqual(max(size_cm), size_cm[0], row['name'])

    def test_short_names_round_trip(self):
        self.assertEqual(sorted(toy_lib.SHORT.values()), ['bus', 'croc', 'henry', 'ketchup', 'thomas'])
        for full, short in toy_lib.SHORT.items():
            self.assertEqual(toy_lib.resolve(short), full)
            self.assertEqual(toy_lib.resolve(full), full)
        self.assertEqual([toy_lib.SHORT[n] for n in toy_lib.targets()], ['bus', 'henry', 'ketchup'])


@unittest.skipIf(mujoco is None, 'mujoco not importable')
class LayoutTests(unittest.TestCase):
    cfg, arena = STAGE1['layout'], STAGE1['arena']

    def test_same_seed_same_layout(self):
        a = layout_lib.make_layout(1003, self.cfg, self.arena)
        b = layout_lib.make_layout(1003, self.cfg, self.arena)
        self.assertEqual(a, b)
        self.assertNotEqual(a, layout_lib.make_layout(1004, self.cfg, self.arena))

    def test_a_thousand_seeds_keep_every_clearance(self):
        for seed in range(1000):
            found = layout_lib.violations(layout_lib.make_layout(seed, self.cfg, self.arena),
                                          self.cfg, self.arena)
            self.assertEqual(found, [], seed)

    def test_committed_formal_layouts_are_the_seeded_ones(self):
        layouts = json.loads((REPO / 'configs/control/stage1_layouts.json').read_text())['layouts']
        self.assertEqual(len(layouts), int(self.cfg['formal_seeds']))
        self.assertEqual(layouts, layout_lib.formal_layouts(self.cfg, self.arena))

    def test_episode_fills_the_target_slot_only(self):
        lay = layout_lib.make_layout(1000, self.cfg, self.arena)
        for target in ('bus', 'henry', 'ketchup'):
            ep = layout_lib.episode(lay, target)
            self.assertEqual(sorted(ep['toys']), sorted([target, 'croc', 'thomas']))
            self.assertEqual(ep['robot'], lay['robot'])


@unittest.skipIf(mujoco is None, 'mujoco not importable')
class OracleBoxTests(unittest.TestCase):
    def seg(self):
        seg = np.full((360, 640, 2), -1, dtype=np.int32)
        seg[:, :, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
        return seg

    def test_raw_box_turns_into_the_upright_box(self):
        seg = self.seg()
        seg[100:120, 200:260, 0] = 7            # raw rows 100-119, columns 200-259
        (box,) = scene_render.boxes_from_seg(seg, {7: 'bus'})
        # upright = np.rot90(raw, 3): column = 359 - raw row, row = raw column
        self.assertEqual(box['xyxy'], [359 - 119, 200, 360 - 100, 260])
        self.assertEqual(box['area_px'], 20 * 60)
        self.assertFalse(box['cut'])
        upright = scene_render.upright(seg[:, :, 0])
        rows, cols = np.nonzero(upright == 7)
        self.assertEqual([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], box['xyxy'])

    def test_two_geoms_of_one_owner_merge_and_edges_mark_cut(self):
        seg = self.seg()
        seg[0:10, 0:10, 0] = 3
        seg[50:60, 70:80, 0] = 4
        (box,) = scene_render.boxes_from_seg(seg, {3: 'tray', 4: 'tray'})
        self.assertEqual(box['xyxy'], [359 - 59, 0, 360, 80])
        self.assertTrue(box['cut'])

    def test_non_geom_pixels_are_ignored(self):
        seg = self.seg()
        seg[10:20, 10:20, 0] = 5
        seg[10:20, 10:20, 1] = int(mujoco.mjtObj.mjOBJ_SITE)
        self.assertEqual(scene_render.boxes_from_seg(seg, {5: 'bus'}), [])


if __name__ == '__main__':
    unittest.main()
