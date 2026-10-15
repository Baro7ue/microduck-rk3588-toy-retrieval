"""Stage 1 zsh launchers, with the fakes from test_launchers.py."""
from __future__ import annotations

import unittest

import test_launchers
from test_launchers import ZSH


@unittest.skipUnless(ZSH, 'zsh not installed')
class Stage1LauncherTests(unittest.TestCase):
    setUp = test_launchers.LauncherTests.setUp
    tearDown = test_launchers.LauncherTests.tearDown
    run_script = test_launchers.LauncherTests.run_script
    calls = test_launchers.LauncherTests.calls
    runs = test_launchers.LauncherTests.runs

    def test_arena_body_local_board_camera_and_floor(self):
        result = self.run_script('scripts/sim/stage1/run_arena_body.zsh')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, env), = self.calls('mjpython')
        root = str(self.root.resolve())
        self.assertEqual(args, [root + '/third_party/microduck_rl',
                                root + '/src/microduck_toy/sim/arena_body.py',
                                '--floor', 'grid',
                                '--stage1-config', root + '/configs/control/stage1.json',
                                '--probe-port', '7811', '--ducks', '1', '--host', '127.0.0.1',
                                '--port', '7801', '--keyframe', 'HOME', '--headless'])
        self.assertEqual(env['MD_STARTUP_SOCKET'], self.env['MD_MAC_STATE'] + '/robotd.sock')
        run, = self.runs()
        self.assertRegex(run.name, r'^\d{8}T\d{6}Z_s1_arena_body$')
        result = self.run_script('scripts/sim/stage1/run_arena_body.zsh', '--board', '--camera',
                                 '--floor', 'wood', '--oracle-fps', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        args, env = self.calls('mjpython')[-1]
        self.assertEqual(args[2:4], ['--floor', 'wood'])
        self.assertEqual(args[args.index('--host') + 1], '192.168.50.1')
        self.assertEqual(args[args.index('--cameras'):args.index('--cameras') + 6],
                         ['--cameras', 'a', '--frame-port', '7901', '--oracle-fps', '0'])
        self.assertEqual(env['MD_STARTUP_SOCKET'], self.env['MD_STARTUP_SOCKET'])
        for bad in (('--floor', 'carpet'), ('--scene', 'x.xml')):
            self.assertNotEqual(self.run_script('scripts/sim/stage1/run_arena_body.zsh',
                                                *bad).returncode, 0, bad)
        self.assertNotEqual(self.run_script('scripts/sim/stage1/run_arena_body.zsh',
                                            FAKE_LSOF_STATUS='0').returncode, 0)   # port busy

    def test_run_test_picks_the_module(self):
        self.assertNotEqual(self.run_script('scripts/sim/stage1/run_test.zsh', '1b',
                                            '--config', 's2').returncode, 0)   # no test port
        for test, module in (('1a', 's1a_scene'), ('1b', 's1b_grasp'), ('1c', 's1c_back'),
                             ('1d', 's1d_range')):
            result = self.run_script('scripts/sim/stage1/run_test.zsh', test, '--config', 's2',
                                     '--quick', FAKE_LSOF_STATUS='0')
            self.assertEqual(result.returncode, 0, result.stderr)
            args, env = self.calls('python')[-1]
            self.assertEqual(args, [str(self.root.resolve()), '-u', '-m',
                                    f'microduck_toy.stage1.{module}', '--config', 's2', '--quick'])
        result = self.run_script('scripts/sim/stage1/run_test.zsh', '1c', '--rescore', 'runs/x')
        self.assertEqual(result.returncode, 0, result.stderr)            # no test port needed
        self.assertEqual(self.calls('python')[-1][0][-2:], ['--rescore', 'runs/x'])
        result = self.run_script('scripts/sim/stage1/run_test.zsh', 'parity', '--floor', 'wood')
        self.assertEqual(result.returncode, 0, result.stderr)            # no body needed
        self.assertEqual(self.calls('python')[-1][0][1:],
                         ['-u', '-m', 'microduck_toy.stage1.parity', '--floor', 'wood'])
        self.assertNotEqual(self.run_script('scripts/sim/stage1/run_test.zsh', '1e').returncode, 0)


if __name__ == '__main__':
    unittest.main()
