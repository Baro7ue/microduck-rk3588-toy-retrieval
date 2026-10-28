"""Stage 4 zsh launchers (scripts/stage4/): the tables and the board load sampler, with the fakes from
test_launchers.py (ssh runs the remote command here, scp copies, taskset runs its command)."""
from __future__ import annotations

import json
import shutil
import unittest

import test_launchers
from test_launchers import RECORD, ZSH, put

# python3 on the "board": board_load.py's output files in --out.
BOARD_LOAD_PY = RECORD + '''out=''
while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
mkdir -p "$out" && printf 't_unix,cpu_all_pct\\n1700000000.0,5\\n' > "$out/load.csv" && echo '{}' > "$out/summary.json"
'''
FAKE_DATE = RECORD + 'echo 1700000000.250000\n'      # the board's `date +%s.%N` (BSD date has no %N)


@unittest.skipUnless(ZSH, 'zsh not installed')
class Stage4LauncherTests(unittest.TestCase):
    setUp = test_launchers.LauncherTests.setUp
    tearDown = test_launchers.LauncherTests.tearDown
    run_script = test_launchers.LauncherTests.run_script
    calls = test_launchers.LauncherTests.calls
    runs = test_launchers.LauncherTests.runs

    def test_run_tables_runs_the_tables_script_with_its_options(self):
        r = self.run_script('scripts/stage4/run_tables.zsh', '--dev', '--onef', 'runs/x')
        self.assertEqual(r.returncode, 0, r.stderr)
        args, env = self.calls('python')[-1]
        self.assertEqual(args[1:], ['-u', str(self.root / 'scripts/stage4/tables.py'), '--dev', '--onef', 'runs/x'])
        self.assertEqual(args[0], str(self.root))                      # from the repo root
        self.assertIn(str(self.root / 'src'), env['PYTHONPATH'])
        self.assertIn(str(self.root / 'offline/vision-export/site-packages'), env['PYTHONPATH'])   # cv2

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_sampler_runs_on_the_little_cores_and_copies_the_rows_back(self):
        put(self.stubs / 'python3', BOARD_LOAD_PY, 0o755)
        put(self.stubs / 'date', FAKE_DATE, 0o755)
        r = self.run_script('scripts/stage4/sample_board_load.zsh', '--period', '0.5')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        (run,) = [p for p in self.runs() if p.name.endswith('_board_load')]
        self.assertEqual((run / 'board/load.csv').read_text().splitlines()[0], 't_unix,cpu_all_pct')
        meta = json.loads((run / 'meta.json').read_text())
        self.assertEqual((meta['cpus'], meta['period_s'], meta['exit_status']), ('0-3', 0.5, 0))
        self.assertIsInstance(meta['offset_s'], float)
        remote = '\n'.join(a for args, _ in self.calls('ssh') for a in args)
        self.assertIn('sha256sum -c --quiet SHA256SUMS &&', remote)
        self.assertIn('taskset -c 0-3 nice -n 5 python3 -u board_load.py --out', remote)
        taskset = self.calls('taskset')[-1][0]
        self.assertEqual(taskset[1:4], ['-c', '0-3', 'nice'])

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_sampler_cores_can_be_changed_or_left_free(self):
        put(self.stubs / 'python3', BOARD_LOAD_PY, 0o755)
        put(self.stubs / 'date', FAKE_DATE, 0o755)
        r = self.run_script('scripts/stage4/sample_board_load.zsh', MD_BOARD_LOAD_CPUS='')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        remote = '\n'.join(a for args, _ in self.calls('ssh') for a in args)
        self.assertIn(' nice -n 5 python3 -u board_load.py', remote)
        self.assertNotIn('taskset', remote)

    def test_sampler_refuses_a_bad_period(self):
        r = self.run_script('scripts/stage4/sample_board_load.zsh', '--period', 'fast')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('--period', r.stderr)


if __name__ == '__main__':
    unittest.main()
