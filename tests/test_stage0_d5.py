"""Stage 0 d5: the frame-pack detector check (no model is loaded) and its two launchers."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from microduck_toy.stage0 import d5_detect  # noqa: E402
import test_launchers  # noqa: E402
from test_launchers import ZSH, put  # noqa: E402

GT = [100, 300, 140, 320, 600]
BUS = {'cls': 'bus', 'conf': 0.6, 'xyxy': [101.0, 301.0, 139.0, 319.0]}
KITE = {'cls': 'kite', 'conf': 0.4, 'xyxy': [10.0, 10.0, 30.0, 30.0]}


def make_pack(root: Path, cloud_rows: list[dict]) -> Path:
    """Two frames, 0.5 m and 1.2 m, with the cloud's detections."""
    pack = root / 'pack'
    (pack / 'frames').mkdir(parents=True)
    (pack / 'cloud').mkdir()
    (pack / 'pack.json').write_text(json.dumps({'name': 'test', 'names': ['bus', 'kite']}))
    views = []
    for i, d in enumerate((0.5, 1.2)):
        frame = f'frames/{i:03d}.png'
        cv2.imwrite(str(pack / frame), np.zeros((640, 360, 3), np.uint8))
        views.append({'frame': frame, 'name': 'SCHOOL_BUS', 'coco': 'bus', 'head': 'h03', 'd': d,
                      'heading': 0, 'gt': GT, 'band': [250, 450]})
    (pack / 'views.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in views))
    (pack / 'cloud/views.jsonl').write_text(
        ''.join(json.dumps({'frame': v['frame'], **r}) + '\n' for v, r in zip(views, cloud_rows)))
    return pack


def write_result(root: Path, rows: list[dict], backend: str) -> Path:
    result = root / f'result_{backend}'
    result.mkdir()
    (result / 'summary.json').write_text(json.dumps({'backend': backend}))
    (result / 'views.jsonl').write_text(
        ''.join(json.dumps({'frame': f'frames/{i:03d}.png', **r}) + '\n' for i, r in enumerate(rows)))
    return result


class RuleTests(unittest.TestCase):
    def test_on_toy(self):
        self.assertTrue(d5_detect.on_toy(BUS, GT))
        self.assertFalse(d5_detect.on_toy(KITE, GT))
        self.assertFalse(d5_detect.on_toy(BUS, None))
        tiny = {'cls': 'kite', 'conf': 0.3, 'xyxy': [138.0, 318.0, 146.0, 324.0]}
        self.assertTrue(d5_detect.on_toy(tiny, GT))             # centre in the grown box
        huge = {'cls': 'kite', 'conf': 0.3, 'xyxy': [60.0, 260.0, 180.0, 360.0]}
        self.assertFalse(d5_detect.on_toy(huge, GT))            # centre inside, but > 4x area

    def test_match(self):
        moved = dict(BUS, conf=0.605, xyxy=[101.5, 301.0, 139.5, 319.0])
        result = d5_detect.match([BUS, KITE], [moved])
        self.assertEqual({k: result[k] for k in ('matched', 'only_expected', 'only_actual')},
                         {'matched': 1, 'only_expected': 1, 'only_actual': 0})
        self.assertAlmostEqual(result['max_dconf'], 0.005)
        wrong_class = dict(BUS, cls='car')
        self.assertEqual(d5_detect.match([BUS], [wrong_class])['matched'], 0)

    def test_band_tiles_go_back_to_frame_coordinates(self):
        image = np.zeros((640, 360, 3), np.uint8)
        seen = []

        def fake_detect(backend, tile, names):
            seen.append(tile.shape)
            return [{'cls': 'bus', 'conf': 0.5, 'xyxy': [1.0, 2.0, 11.0, 12.0]}]

        with mock.patch.object(d5_detect, 'detect', fake_detect):
            found = d5_detect.detect_band(None, image, (250, 450), ['bus'])
        self.assertEqual(seen, [(200, 180, 3)] * 3)
        self.assertEqual([d['xyxy'] for d in found],
                         [[1.0 + x, 252.0, 11.0 + x, 262.0] for x in (0, 90, 180)])


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cloud = [{'whole': [BUS, KITE], 'band': [BUS]}, {'whole': [], 'band': [KITE]}]
        self.pack = make_pack(self.root, self.cloud)

    def tearDown(self):
        self.tmp.cleanup()

    def compare(self, result: Path) -> tuple[int, dict]:
        with redirect_stdout(io.StringIO()):
            code = d5_detect.main(['compare', '--frames', str(self.pack), '--result', str(result)])
        return code, json.loads((result / 'compare.json').read_text())

    def test_cpu_identical_passes(self):
        code, report = self.compare(write_result(self.root, self.cloud, 'cpu'))
        self.assertEqual(code, 0)
        self.assertTrue(report['pass'])
        first = report['by_distance'][0]
        self.assertEqual(first, {'d': 0.5, 'mode': 'band', 'n': 1, 'cloud_any': 1, 'any': 1,
                                 'cloud_hit': 1, 'hit': 1})

    def test_cpu_lost_toy_fails(self):
        rows = [{'whole': [KITE], 'band': [BUS]}, self.cloud[1]]
        code, report = self.compare(write_result(self.root, rows, 'cpu'))
        self.assertEqual(code, 2)
        self.assertFalse(report['counts_ok'])

    def test_npu_allows_two_frames(self):
        rows = [{'whole': [KITE], 'band': [BUS]}, self.cloud[1]]
        code, report = self.compare(write_result(self.root, rows, 'npu'))
        self.assertEqual(code, 0)
        self.assertTrue(report['pass'])

    def test_subset_keeps_one_distance(self):
        out = self.root / 'small'
        with redirect_stdout(io.StringIO()):
            code = d5_detect.main(['subset', '--frames', str(self.pack), '--distances', '0.5',
                                   '--output-dir', str(out)])
        self.assertEqual(code, 0)
        self.assertEqual([v['d'] for v in d5_detect.read_jsonl(out / 'views.jsonl')], [0.5])
        self.assertEqual(len(d5_detect.read_jsonl(out / 'cloud/views.jsonl')), 1)
        self.assertEqual(sorted(p.name for p in (out / 'frames').iterdir()), ['000.png'])

    def test_run_rejects_an_unknown_onnx(self):
        model = put(self.root / 'other.onnx', 'not the model')
        with self.assertRaises(ValueError):
            d5_detect.main(['run', '--frames', str(self.pack), '--backend', 'cpu',
                            '--model', str(model), '--output-dir', str(self.root / 'out')])


@unittest.skipUnless(ZSH, 'zsh not installed')
class D5LauncherTests(unittest.TestCase):
    setUp_base = test_launchers.LauncherTests.setUp
    tearDown = test_launchers.LauncherTests.tearDown
    run_script = test_launchers.LauncherTests.run_script
    calls = test_launchers.LauncherTests.calls
    runs = test_launchers.LauncherTests.runs

    def setUp(self):
        self.setUp_base()
        self.pack = self.root / 'runs/20261013T120000Z_d5_cloud_pack'
        put(self.pack / 'pack.json', '{}')
        put(self.pack / 'cloud/views.jsonl', '')

    def test_mac_runs_then_compares(self):
        result = self.run_script('scripts/vision/stage0/run_d5_mac.zsh', str(self.pack))
        self.assertEqual(result.returncode, 0, result.stderr)
        (run_args, env), (compare_args, _) = self.calls('python')
        run = next(p for p in self.runs() if p.name.endswith('_d5_mac_cpu'))
        self.assertEqual(run_args[1:], [
            '-u', '-m', 'microduck_toy.stage0.d5_detect', 'run', '--frames', str(self.pack),
            '--backend', 'cpu', '--model',
            str(self.root / 'models/yolo/yolo11n_coco/yolo11n_coco_fp32_320.onnx'),
            '--output-dir', str(run)])
        self.assertEqual(compare_args[-4:], ['--frames', str(self.pack), '--result', str(run)])
        self.assertIn('offline/vision-export/site-packages', env['PYTHONPATH'])
        bad = self.run_script('scripts/vision/stage0/run_d5_mac.zsh', str(self.root))
        self.assertNotEqual(bad.returncode, 0)

    def test_board_sends_bundle_runs_npu_and_compares(self):
        result = self.run_script('scripts/board/stage0/run_d5_board.zsh', str(self.pack),
                                 '--distance', '1.2')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run = next(p for p in self.runs() if p.name.endswith('_d5_board_npu'))
        subset, compare = self.calls('python')
        self.assertEqual(subset[0][-6:], ['--frames', str(self.pack), '--distances', '1.2',
                                          '--output-dir', str(run / 'bundle/pack')])
        self.assertEqual(compare[0][-4:], ['--frames', str(run / 'bundle/pack'),
                                           '--result', str(run / 'board')])
        (board_args, _), = self.calls('board_python')
        self.assertEqual(board_args[1:], [
            '-u', '-m', 'microduck_toy.stage0.d5_detect', 'run', '--frames', 'pack',
            '--backend', 'npu', '--model', 'models/yolo11n_coco_fp16_320.rknn',
            '--output-dir', str(self.board / 'run' / run.name)])
        bundle = self.board / 'bundles' / run.name
        self.assertTrue((bundle / 'microduck_toy/stage0/d5_detect.py').is_file())
        self.assertTrue((bundle / 'models/yolo11n_coco_fp16_320.rknn').is_file())
        self.assertTrue((run / 'board/detections.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
