"""Path helpers: environment variables win, otherwise paths follow the checkout."""
from pathlib import Path
import sys
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from microduck_toy import paths


class PathTests(unittest.TestCase):
    def test_defaults_follow_the_checkout(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            self.assertEqual(paths.repo_root(), REPO)
            self.assertEqual(paths.runs_dir(), REPO / 'runs')
            self.assertEqual(paths.board_root(), Path.home() / 'microduck-board')

    def test_environment_overrides(self):
        env = {'MD_ROOT': '/r', 'MD_RUNS': '/r/x', 'MD_BOARD_ROOT': '/home/linaro/microduck'}
        with mock.patch.dict('os.environ', env, clear=True):
            self.assertEqual(paths.repo_root(), Path('/r'))
            self.assertEqual(paths.runs_dir(), Path('/r/x'))
            self.assertEqual(paths.board_root(), Path('/home/linaro/microduck'))


if __name__ == '__main__':
    unittest.main()
