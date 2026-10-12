"""cpu_npu_compare.main() reads the CPU/NPU comparison bundle given with --root."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from microduck_toy.vision import cpu_npu_compare


class BundleLocationTests(unittest.TestCase):
    def run_main(self, *args):
        api = ModuleType('rknnlite.api')
        api.RKNNLite = object
        with mock.patch.dict(sys.modules, {'rknnlite': ModuleType('rknnlite'), 'rknnlite.api': api}), \
                mock.patch.object(sys, 'argv', ['cpu_npu_compare', *args]):
            cpu_npu_compare.main()

    def test_main_reads_models_from_the_given_bundle(self):
        with tempfile.TemporaryDirectory() as board:
            with self.assertRaises(FileNotFoundError) as caught:
                self.run_main('--root', f'{board}/bundle', '--output-dir', f'{board}/out')
        self.assertEqual(Path(caught.exception.filename), Path(board) / 'bundle/models/conversion.json')

    def test_bundle_root_is_required(self):
        with tempfile.TemporaryDirectory() as board, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.run_main('--output-dir', f'{board}/out')
            self.assertFalse(Path(board, 'out').exists())
        self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
