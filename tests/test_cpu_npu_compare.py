"""cpu_npu_compare.main() reads the board's CPU/NPU comparison bundle under MD_BOARD_ROOT."""
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
    def test_main_reads_models_from_the_board_bundle(self):
        api = ModuleType('rknnlite.api')
        api.RKNNLite = object
        with tempfile.TemporaryDirectory() as board, \
                mock.patch.dict(sys.modules, {'rknnlite': ModuleType('rknnlite'), 'rknnlite.api': api}), \
                mock.patch.dict('os.environ', {'MD_BOARD_ROOT': board}), \
                mock.patch.object(sys, 'argv', ['cpu_npu_compare', '--output-dir', f'{board}/out']):
            with self.assertRaises(FileNotFoundError) as caught:
                cpu_npu_compare.main()
        self.assertEqual(Path(caught.exception.filename),
                         Path(board) / 'yolo11n-cpu-npu/models/conversion.json')


if __name__ == '__main__':
    unittest.main()
