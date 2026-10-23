"""Stage 2 2H: the C++/Python comparison and the frame list format."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from microduck_toy.stage2 import cpp_bench
from microduck_toy.vision import views


def write_jsonl(path: Path, rows) -> Path:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


class CompareFrameTests(unittest.TestCase):
    REF = [[10, 10, 50, 50, 0.9, 0], [100, 100, 140, 130, 0.5, 2]]

    def test_identical_and_tiny_differences_pass(self):
        self.assertTrue(cpp_bench.compare_frame(self.REF, self.REF)['ok'])
        near = [[10.05, 10, 50, 50, 0.9005, 0], [100, 100, 140, 130.1, 0.5, 2]]
        r = cpp_bench.compare_frame(near[::-1], self.REF)    # order does not matter
        self.assertTrue(r['ok'])
        self.assertAlmostEqual(r['worst_conf'], 0.0005, places=6)

    def test_count_score_box_and_class_differences_fail(self):
        self.assertFalse(cpp_bench.compare_frame(self.REF[:1], self.REF)['ok'])
        self.assertFalse(cpp_bench.compare_frame([[10, 10, 50, 50, 0.902, 0], self.REF[1]], self.REF)['ok'])
        self.assertFalse(cpp_bench.compare_frame([[12, 10, 50, 50, 0.9, 0], self.REF[1]], self.REF)['ok'])
        r = cpp_bench.compare_frame([[10, 10, 50, 50, 0.9, 1], self.REF[1]], self.REF)
        self.assertEqual(r['unpaired_ref'], 1)
        self.assertFalse(r['ok'])


class CompareTests(unittest.TestCase):
    def test_report_names_the_first_bad_frame_and_input_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            box = [10, 10, 50, 50, 0.9, 0]
            ref = write_jsonl(t / 'ref.jsonl', [
                {'id': 'a', 'boxes': [box], 'ms': {'preprocess': 5, 'inference': 20, 'postprocess': 3}},
                {'id': 'b', 'boxes': [box], 'ms': {'preprocess': 5, 'inference': 20, 'postprocess': 3}}])
            cpp = write_jsonl(t / 'cpp.jsonl', [
                {'id': 'a', 'boxes': [box], 'ms': {'decode': 1, 'resize': 1, 'inference': 19, 'postprocess': 0.2}},
                {'id': 'b', 'boxes': [], 'ms': {'decode': 1, 'resize': 1, 'inference': 19, 'postprocess': 0.2}}])
            for d in ('ci', 'ri'):
                (t / d).mkdir()
            (t / 'ri' / 'a.rgb').write_bytes(bytes(12))
            (t / 'ci' / 'a.rgb').write_bytes(bytes(11) + b'\x02')
            out = cpp_bench.compare(cpp, ref, t / 'ci', t / 'ri')
        self.assertFalse(out['pass'])
        self.assertEqual((out['frames'], out['frames_ok']), (2, 1))
        self.assertEqual(out['first_bad_frame']['id'], 'b')
        self.assertEqual(out['inputs']['worst'], {'id': 'a', 'bytes_differ': 1, 'max': 2})
        self.assertEqual(out['p50_ms']['cpp']['postprocess'], 0.2)
        self.assertEqual(out['p50_ms']['python']['postprocess'], 3)

    def test_missing_frames_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            row = {'id': 'a', 'boxes': [], 'ms': {'postprocess': 1}}
            ref = write_jsonl(t / 'ref.jsonl', [row, {**row, 'id': 'b'}])
            cpp = write_jsonl(t / 'cpp.jsonl', [row])
            out = cpp_bench.compare(cpp, ref, None, None)
        self.assertFalse(out['pass'])
        self.assertEqual(out['missing_or_extra_ids'], ['b'])


class FramesTsvTests(unittest.TestCase):
    def test_rows_round_trip_through_the_reader(self):
        m = views.window_mapping(301.4, 30)
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            (pack / 'frames.tsv').write_text('id\tkind\tx0\ty0\tgain\tpad_x\tpad_y\n'
                                             + cpp_bench.mapping_row('test_1', m) + '\n'
                                             + cpp_bench.mapping_row('test_2', views.letterbox_mapping()) + '\n')
            rows = cpp_bench.read_frames(pack)
        self.assertEqual(rows, [('test_1', m), ('test_2', views.letterbox_mapping())])


if __name__ == '__main__':
    unittest.main()
