"""Stage 2G exit table: the frozen models match their hashes and every acceptance line holds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[1]
FACTS = json.loads((REPO / 'configs/vision/stage2_facts.json').read_text())
CFG = json.loads((REPO / 'configs/vision/stage2.json').read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelFileTests(unittest.TestCase):
    def test_tracked_model_files_match_the_facts(self):
        model = FACTS['model']
        for key, f in model['files'].items():
            if key == 'pt':
                continue
            with self.subTest(key=key):
                self.assertEqual(sha256(REPO / model['dir'] / f['file']), f['sha256'])

    def test_closed_loop_model_is_the_int8_rknn_and_its_records_agree(self):
        model = FACTS['model']
        self.assertEqual(model['closed_loop'], 'int8')
        conv = json.loads((REPO / model['dir'] / 'conversion_int8.json').read_text())
        export = json.loads((REPO / model['dir'] / 'export.json').read_text())
        self.assertEqual(conv['rknn_sha256'], model['files']['rknn_int8']['sha256'])
        self.assertEqual(conv['onnx_sha256'], export['split']['sha256'])
        self.assertEqual(conv['source_pt_sha256'], model['files']['pt']['sha256'])
        self.assertEqual(export['names'], CFG['classes'])
        self.assertEqual(model['targets'], CFG['targets'])


class AcceptanceTests(unittest.TestCase):
    def test_search_radius_reaches_the_planned_1_6_m(self):
        r = FACTS['search_radius']
        self.assertGreaterEqual(r['radius_m'], r['need_m'])
        self.assertTrue(r['pass'])

    def test_confirm_distance_is_no_farther_than_the_2_percent_rule(self):
        c = FACTS['confirm_distance']
        self.assertLessEqual(c['zero_confusion_m'], c['rule_2pct_m'])
        for name, pair in c['named_pairs'].items():
            with self.subTest(pair=name):
                self.assertLessEqual(pair['rate'], 0.02)

    def test_h1_ranging_and_precision_choice(self):
        h = FACTS['h1_ranging']
        self.assertLessEqual(h['int8_board_extra_rms_cm'], h['need_extra_rms_cm_max'])
        self.assertLessEqual(FACTS['precision_choice']['drop_pp'], 2.0)

    def test_every_board_backend_keeps_up_with_the_camera_and_robotd(self):
        limit = CFG['deploy']['e2e_p99_ms_max']
        for name, b in FACTS['deploy_board']['backends'].items():
            with self.subTest(backend=name):
                self.assertLessEqual(b['e2e_ms_p50_p99'][1], limit)
                self.assertLessEqual(b['robotd_missed'], 1)
                self.assertTrue(b['pass'])

    def test_board_launchers_pin_to_the_cores_that_were_fast(self):
        c = FACTS['deploy_board']['preprocess_cores']
        self.assertLess(c['big_a76_cores_4_7_p50_ms'], c['little_a55_cores_0_3_p50_ms'])
        env_sh = (REPO / 'env.sh').read_text()
        self.assertIn('export MD_BOARD_CPUS="${MD_BOARD_CPUS-4-7}"', env_sh)

    def test_parity_and_board_agreement_pass(self):
        p = FACTS['parity']
        self.assertTrue(p['mac_same_frames']['pass'])
        self.assertTrue(p['mac_render']['pass'])
        for name, v in p['board_vs_cloud'].items():
            with self.subTest(pair=name):
                self.assertGreaterEqual(v['found_agree'], 0.95)


if __name__ == '__main__':
    unittest.main()
