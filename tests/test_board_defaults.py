"""Board-side defaults after the 2026-10-12 board reorganisation (~/microduck/).

The board programs no longer fall back to GPT's old bundle folders under ~/microduck-board/:
the model and the output folder must be given (the launchers always pass both), and the
default output root is ~/microduck/run/.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microduck_toy.vision import rknn_runner, yolo_camera_probe
from test_yolo_camera_probe import MODEL, camera


def cli(module: str, *args: str) -> subprocess.CompletedProcess:
    path = os.pathsep.join(filter(None, (str(SRC), os.environ.get('PYTHONPATH'))))   # keeps cv2 on the Mac
    env = {**os.environ, 'PYTHONPATH': path}
    return subprocess.run([sys.executable, '-m', module, *args], env=env, capture_output=True,
                          text=True, timeout=60)


class BoardDefaultTests(unittest.TestCase):
    def test_camera_probes_need_an_explicit_model(self):
        for main in (yolo_camera_probe.main, rknn_runner.main):
            with tempfile.TemporaryDirectory() as board, \
                    mock.patch.dict('os.environ', {'MD_BOARD_ROOT': board}):
                with self.assertRaises(ValueError, msg=main.__module__):
                    main(duration=1.0)
                self.assertEqual(list(Path(board).iterdir()), [], main.__module__)

    def test_camera_probe_output_defaults_to_board_run_folder(self):
        with tempfile.TemporaryDirectory() as board, camera() as (address, _payload), \
                mock.patch.dict('os.environ', {'MD_BOARD_ROOT': board}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yolo_camera_probe.main(duration=0.35, address=address, model_path=MODEL)
            directory, = (Path(board) / 'run').iterdir()
        self.assertTrue(directory.name.startswith('camera_cpu_'))

    def test_command_lines_require_model_and_output_dir(self):
        for module in ('microduck_toy.vision.yolo_camera_probe', 'microduck_toy.vision.rknn_runner'):
            result = cli(module, '--duration', '1')
            self.assertEqual(result.returncode, 2, module + result.stderr)
            self.assertIn('--model', result.stderr)
            self.assertIn('--output-dir', result.stderr)

    def test_resource_sampler_defaults_to_the_new_board_root(self):
        spec = importlib.util.spec_from_file_location(
            'measure_board_resources', REPO / 'scripts/board/measure_board_resources.py')
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict('os.environ', {}, clear=True):
            spec.loader.exec_module(module)
        self.assertEqual(module.ROOT, Path('/home/linaro/microduck/run'))


if __name__ == '__main__':
    unittest.main()
