"""Tests for docs/reorg/reorg.py on a small throwaway folder."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'docs/reorg/reorg.py'
spec = importlib.util.spec_from_file_location('reorg', TOOL)
reorg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reorg)

RULES = [
    ('P2', 'move', 'old/evidence/G1', 'evidence/G1'),
    ('P2', 'move_rest', 'old/evidence', 'evidence/misc'),
    ('P2', 'rmdir', 'old/evidence', None),
    ('P3', 'move', 'old/repo', 'third_party/repo'),
    ('P3', 'move', 'third_party/repo/.venv', 'third_party/repo/.venv.pre_reorg'),
    ('P3', 'rmdir', 'old', None),
    ('P3', 'symlink', '../third_party/repo/assets', 'scene/assets'),
    ('P4', 'copy', 'work/model.onnx', 'models/m.onnx'),
    ('P4', 'move', 'work', 'archive/work'),
]
OPAQUE = ('old/repo', 'third_party/repo')
TOP = {'evidence', 'third_party', 'models', 'archive', 'scene', 'docs', 'README.md'}


def write(root: Path, rel: str, text: str = 'x') -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def tree(root: Path) -> dict:
    out = {}
    for base, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(base) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                out[rel] = 'L:' + os.readlink(path)
            elif path.is_file():
                out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                out[rel] = 'D'
    return out


class ReorgTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write(self.root, 'old/evidence/G1/a.log', 'a')
        write(self.root, 'old/evidence/b.csv', 'b')
        write(self.root, 'old/evidence/.DS_Store', 'ds')
        write(self.root, 'old/repo/assets/mesh.stl', 'mesh')
        write(self.root, 'old/repo/.venv/bin/python', 'py')
        write(self.root, 'old/.DS_Store', 'ds')
        write(self.root, 'work/model.onnx', 'onnx')
        write(self.root, 'work/run.py', 'run')
        write(self.root, 'work/__pycache__/run.cpython-312.pyc', 'pyc')
        write(self.root, 'docs/reorg/staged/README.md', 'readme')
        write(self.root, 'docs/reorg/staged/scene/scene.xml', '<mujoco/>')
        os.symlink('../old/repo/assets', self.root / 'work/assets')   # resolves now, dangles after P3/P4
        os.symlink('nowhere', self.root / 'work/broken')              # already dangling
        self.tool = reorg.Reorg(self.root, rules=RULES, opaque=OPAQUE, top_level=TOP)
        self.tool.busy_processes = lambda: []

    def tearDown(self):
        self.tmp.cleanup()

    def apply_all(self):
        for phase in ('P1', 'P2', 'P3', 'P4'):
            self.tool.apply(phase)

    def test_plan_has_no_errors_and_changes_nothing(self):
        before = tree(self.root)
        rows = self.tool.plan()
        self.assertEqual([r for r in rows if r.status.startswith('error')], [])
        self.assertIn(('P2', 'move', 'old/evidence/b.csv', 'evidence/misc/b.csv', 'ok'),
                      [(r.phase, r.op, r.src, r.dst, r.status) for r in rows])
        self.assertEqual(tree(self.root), {**before, 'docs/reorg/move_plan.tsv': tree(self.root)['docs/reorg/move_plan.tsv']})

    def test_plan_reports_unmapped_leftovers(self):
        write(self.root, 'stray/file.txt')
        rows = self.tool.plan()
        self.assertTrue(any(r.op == 'leftover' and r.src == 'stray' for r in rows))

    def test_plan_reports_destination_conflict(self):
        write(self.root, 'evidence/G1/existing.txt')
        rows = self.tool.plan()
        self.assertTrue(any(r.src == 'old/evidence/G1' and r.status.startswith('error') for r in rows))

    def test_apply_all_phases_reaches_target_layout(self):
        self.apply_all()
        after = tree(self.root)
        self.assertEqual(after['evidence/G1/a.log'], hashlib.sha256(b'a').hexdigest())
        self.assertIn('evidence/misc/b.csv', after)
        self.assertIn('third_party/repo/.venv.pre_reorg/bin/python', after)
        self.assertEqual(after['scene/assets'], 'L:../third_party/repo/assets')
        self.assertTrue((self.root / 'scene/assets/mesh.stl').is_file())
        self.assertIn('models/m.onnx', after)
        self.assertIn('archive/work/model.onnx', after)
        self.assertIn('README.md', after)
        self.assertIn('scene/scene.xml', after)
        self.assertNotIn('old', after)
        self.assertNotIn('work', after)
        self.assertNotIn('archive/work/__pycache__', after)
        self.assertEqual(self.tool.check(), [])

    def test_cleaning_keeps_caches_inside_opaque_upstream(self):
        write(self.root, 'old/repo/.DS_Store', 'ds')
        self.tool.apply('P1')
        self.assertTrue((self.root / 'old/repo/.DS_Store').exists())
        self.assertFalse((self.root / 'work/__pycache__').exists())
        self.assertFalse((self.root / 'old/.DS_Store').exists())

    def test_undo_each_phase_restores_previous_state(self):
        states = [tree(self.root)]
        for phase in ('P1', 'P2', 'P3', 'P4'):
            self.tool.apply(phase)
            states.append(tree(self.root))
        for index, phase in reversed(list(enumerate(('P1', 'P2', 'P3', 'P4')))):
            self.tool.undo(phase)
            expected = {k: v for k, v in states[index].items() if not k.startswith('docs/reorg/')}
            actual = {k: v for k, v in tree(self.root).items() if not k.startswith('docs/reorg/')}
            if phase == 'P1':
                # P1 cleaning removes caches for good; everything else comes back.
                expected = {k: v for k, v in expected.items()
                            if '.DS_Store' not in k and '__pycache__' not in k}
            self.assertEqual(actual, expected, phase)

    def test_phases_must_run_in_order(self):
        with self.assertRaises(reorg.ReorgError):
            self.tool.apply('P3')
        self.tool.apply('P1')
        self.tool.apply('P2')
        with self.assertRaises(reorg.ReorgError):
            self.tool.undo('P1')

    def test_apply_stops_before_overwriting(self):
        self.tool.apply('P1')
        write(self.root, 'evidence/misc/b.csv', 'other')
        with self.assertRaises(reorg.ReorgError):
            self.tool.apply('P2')
        self.assertEqual((self.root / 'old/evidence/b.csv').read_text(), 'b')
        self.assertEqual((self.root / 'evidence/misc/b.csv').read_text(), 'other')

    def test_rmdir_refuses_directory_with_content(self):
        self.tool.apply('P1')
        self.tool.apply('P2')
        write(self.root, 'old/new_file.txt')
        with self.assertRaises(reorg.ReorgError):
            self.tool.apply('P3')

    def test_check_detects_changed_file(self):
        self.apply_all()
        (self.root / 'evidence/G1/a.log').write_text('changed')
        problems = self.tool.check()
        self.assertEqual(len(problems), 1)
        self.assertIn('evidence/G1/a.log', problems[0])

    def test_failed_phase_can_resume_after_fix(self):
        self.tool.apply('P1')
        write(self.root, 'evidence/misc/b.csv', 'other')
        with self.assertRaises(reorg.ReorgError):
            self.tool.apply('P2')
        (self.root / 'evidence/misc/b.csv').unlink()
        self.tool.apply('P2')
        self.assertTrue((self.root / 'evidence/misc/b.csv').exists())
        self.assertFalse((self.root / 'old/evidence').exists())

    def test_manifest_lists_every_file_with_hash(self):
        self.apply_all()
        self.tool.write_manifests(['evidence'])
        lines = (self.root / 'evidence/MANIFEST.tsv').read_text().splitlines()
        self.assertEqual(lines[0], 'path\tbytes\tsha256')
        self.assertIn('G1/a.log\t1\t' + hashlib.sha256(b'a').hexdigest(), lines)
        self.assertNotIn('MANIFEST.tsv', ''.join(line.split('\t')[0] for line in lines[1:]))


    def without_tool_files(self) -> dict:
        return {k: v for k, v in tree(self.root).items() if not k.startswith('docs/reorg/')}

    def test_undo_stops_before_changing_anything_when_venv_was_rebuilt(self):
        self.apply_all()
        write(self.root, 'third_party/repo/.venv/bin/python', 'rebuilt')
        self.tool.undo('P4')
        before = self.without_tool_files()
        with self.assertRaises(reorg.ReorgError) as caught:
            self.tool.undo('P3')
        self.assertIn('third_party/repo/.venv', str(caught.exception))
        self.assertEqual(self.without_tool_files(), before)
        self.assertEqual(self.tool.phase_state('P3'), 'applied')
        os.rename(self.root / 'third_party/repo/.venv', self.root / 'third_party/repo/.venv.rebuilt')
        self.tool.undo('P3')
        self.assertTrue((self.root / 'old/repo/.venv/bin/python').is_file())
        self.assertTrue((self.root / 'old/repo/.venv.rebuilt/bin/python').is_file())

    def test_undo_removes_caches_and_tool_manifests_it_meets(self):
        original = tree(self.root)
        self.apply_all()
        write(self.root, 'scene/__pycache__/scene.cpython-39.pyc', 'pyc')
        write(self.root, 'evidence/misc/.DS_Store', 'ds')
        self.tool.write_manifests(['evidence', 'archive/work'])
        for phase in ('P4', 'P3', 'P2', 'P1'):
            self.tool.undo(phase)
        expected = {k: v for k, v in original.items()
                    if '.DS_Store' not in k and '__pycache__' not in k and not k.startswith('docs/reorg/')}
        self.assertEqual(self.without_tool_files(), expected)

    def test_undo_refuses_files_it_did_not_place(self):
        self.apply_all()
        write(self.root, 'evidence/misc/notes.txt', 'mine')
        self.tool.undo('P4')
        self.tool.undo('P3')
        before = self.without_tool_files()
        with self.assertRaises(reorg.ReorgError) as caught:
            self.tool.undo('P2')
        self.assertIn('evidence/misc/notes.txt', str(caught.exception))
        self.assertEqual(self.without_tool_files(), before)

    def test_undo_keeps_a_copy_that_was_changed(self):
        self.apply_all()
        write(self.root, 'models/m.onnx', 'retrained')
        before = self.without_tool_files()
        with self.assertRaises(reorg.ReorgError) as caught:
            self.tool.undo('P4')
        self.assertIn('models/m.onnx', str(caught.exception))
        self.assertEqual(self.without_tool_files(), before)

    def test_check_warns_about_links_that_no_longer_resolve(self):
        self.apply_all()
        self.assertEqual(self.tool.check(), [])
        self.assertEqual(self.tool.dangling_links(), ['work/assets -> archive/work/assets'])

    def test_running_simulator_blocks_moves_from_p2_on(self):
        self.tool.busy_processes = lambda: ['123 mjpython body_server']
        self.tool.apply('P1')
        with self.assertRaises(reorg.ReorgError):
            self.tool.apply('P2')
        self.tool.apply('P2', force=True)
        with self.assertRaises(reorg.ReorgError):
            self.tool.undo('P2')


if __name__ == '__main__':
    unittest.main()
