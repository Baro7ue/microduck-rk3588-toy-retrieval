"""Stage 0 robotd templates and zsh launchers, with the fakes from test_launchers.py."""
from __future__ import annotations

import shutil
import tomllib
import unittest

import test_launchers
from test_launchers import REPO, ZSH, put

TEMPLATES = REPO / 'configs/robotd/stage0'
OURS = {'action_scale': 1.0, 'standing_gain_ratio': 1.0, 'head_lowpass': 1.0,
        'legs_lowpass': 1.0}
UPSTREAM = {'action_scale': 0.9, 'standing_gain_ratio': 0.8, 'head_lowpass': 0.5,
            'legs_lowpass': 0.7}
STAND = {'s1': ('alpha_stand.onnx', True), 's2': ('velstand.onnx', False)}


class TemplateTests(unittest.TestCase):
    def test_four_candidates(self):
        self.assertEqual(sorted(p.name for p in TEMPLATES.glob('*.toml.in')),
                         ['s1.toml.in', 's1u.toml.in', 's2.toml.in', 's2u.toml.in'])
        for path in TEMPLATES.glob('*.toml.in'):
            name = path.name.split('.')[0]
            config = tomllib.loads(path.read_text())
            policy, safety = config['policy'], config['safety']
            stand, limp = STAND[name[:2]]
            self.assertEqual(policy['walk'], '@POLICY_DIR@/walk_17500.onnx', name)
            self.assertEqual(policy['stand'], f'@POLICY_DIR@/{stand}', name)
            self.assertEqual(policy['ground_pick'], '@POLICY_DIR@/alpha_ground_pick.onnx', name)
            self.assertEqual(safety['limp_fall'], limp, name)
            tuning = UPSTREAM if name.endswith('u') else OURS
            self.assertEqual({k: policy[k] for k in tuning}, tuning, name)
            self.assertFalse(policy['voltage_adapt'], name)
            comments = [line for line in path.read_text().splitlines() if line.startswith('#')]
            self.assertFalse(any('@POLICY_DIR@' in line for line in comments), name)


@unittest.skipUnless(ZSH, 'zsh not installed')
class Stage0LauncherTests(unittest.TestCase):
    setUp_base = test_launchers.LauncherTests.setUp
    tearDown = test_launchers.LauncherTests.tearDown
    run_script = test_launchers.LauncherTests.run_script
    calls = test_launchers.LauncherTests.calls
    runs = test_launchers.LauncherTests.runs

    def setUp(self):
        self.setUp_base()
        for name in ('alpha_stand.onnx', 'alpha_ground_pick.onnx'):
            put(self.root / 'models/policies' / name, name)

    def test_probe_body_local_and_board(self):
        result = self.run_script('scripts/sim/stage0/run_probe_body.zsh')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, env), = self.calls('mjpython')
        root = str(self.root.resolve())
        self.assertEqual(args, [root + '/third_party/microduck_rl',
                                root + '/src/microduck_toy/sim/probe_body.py',
                                '--probe-port', '7811', '--ducks', '1', '--host', '127.0.0.1',
                                '--port', '7801', '--keyframe', 'HOME', '--headless'])
        self.assertEqual(env['MD_STARTUP_SOCKET'], self.env['MD_MAC_STATE'] + '/robotd.sock')
        run, = self.runs()
        self.assertRegex(run.name, r'^\d{8}T\d{6}Z_s0_probe_body$')
        result = self.run_script('scripts/sim/stage0/run_probe_body.zsh', '--board', '--camera')
        self.assertEqual(result.returncode, 0, result.stderr)
        args, env = self.calls('mjpython')[-1]
        self.assertEqual(args[6:8], ['--host', '192.168.50.1'])
        self.assertIn('--cameras', args)
        self.assertEqual(env['MD_STARTUP_SOCKET'], self.env['MD_STARTUP_SOCKET'])
        self.assertNotEqual(self.run_script('scripts/sim/stage0/run_probe_body.zsh',
                                            FAKE_LSOF_STATUS='0').returncode, 0)

    def test_mac_robotd_renders_the_chosen_config(self):
        result = self.run_script('scripts/sim/stage0/run_robotd.zsh', 's1', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, env), = self.calls('robotd')
        run, = self.runs()
        self.assertRegex(run.name, r'^\d{8}T\d{6}Z_s0_robotd_s1$')
        self.assertEqual(args[1:], ['--sim', '127.0.0.1:7801', '--params', str(run / 'robotd.toml'),
                                    '--socket', self.env['MD_MAC_STATE'] + '/robotd.sock'])
        config = tomllib.loads((run / 'robotd.toml').read_text())
        policies = str(self.root.resolve() / 'models/policies')
        self.assertEqual(config['policy']['stand'], f'{policies}/alpha_stand.onnx')
        self.assertTrue(config['safety']['limp_fall'])

    def test_mac_robotd_refuses_unknown_config_and_missing_policy(self):
        self.assertNotEqual(self.run_script('scripts/sim/stage0/run_robotd.zsh', 's9',
                                            FAKE_LSOF_STATUS='0').returncode, 0)
        (self.root / 'models/policies/alpha_ground_pick.onnx').unlink()
        result = self.run_script('scripts/sim/stage0/run_robotd.zsh', 's2', FAKE_LSOF_STATUS='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('alpha_ground_pick.onnx', result.stderr)
        self.assertEqual(self.calls('robotd'), [])

    def test_run_test_picks_the_module(self):
        self.assertNotEqual(self.run_script('scripts/sim/stage0/run_test.zsh', '0c',
                                            '--config', 's2').returncode, 0)   # no probe port
        for test, module in (('0a', 's0a_walk'), ('0b', 's0b_pick'), ('0c', 's0c_getup')):
            result = self.run_script('scripts/sim/stage0/run_test.zsh', test, '--config', 's2',
                                     '--quick', FAKE_LSOF_STATUS='0')
            self.assertEqual(result.returncode, 0, result.stderr)
            args, env = self.calls('python')[-1]
            self.assertEqual(args, [str(self.root.resolve()), '-u', '-m',
                                    f'microduck_toy.stage0.{module}', '--config', 's2', '--quick'])
            self.assertTrue(env['PYTHONPATH'].startswith(str(self.root.resolve() / 'src')))
        result = self.run_script('scripts/sim/stage0/run_test.zsh', '0a', '--rescore', 'runs/x')
        self.assertEqual(result.returncode, 0, result.stderr)            # no probe needed
        self.assertEqual(self.calls('python')[-1][0][-2:], ['--rescore', 'runs/x'])

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_board_robotd_bundles_the_config_policies(self):
        result = self.run_script('scripts/board/stage0/run_board_robotd.zsh', 's1')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run, = self.runs()
        self.assertRegex(run.name, r'^\d{8}T\d{6}Z_s0_board_robotd_s1$')
        remote = self.board / 'bundles' / run.name
        self.assertEqual(sorted(p.name for p in (remote / 'policies').iterdir()),
                         ['alpha_ground_pick.onnx', 'alpha_stand.onnx', 'walk_17500.onnx'])
        config = tomllib.loads((remote / 'robotd.toml').read_text())
        self.assertEqual(config['policy']['stand'], f'{remote}/policies/alpha_stand.onnx')
        (args, _), = self.calls('robotd')
        self.assertEqual(args[1:5], ['--sim', '192.168.50.1:7801', '--params',
                                     f'{remote}/robotd.toml'])
        (pin, _), = self.calls('taskset')                 # robotd on a big core of its own
        self.assertEqual(pin[1:4], ['-c', '7', str(self.board / 'bin/robotd')])
        self.assertNotEqual(self.run_script('scripts/board/stage0/run_board_robotd.zsh').returncode, 0)

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_board_robotd_unpinned_when_asked(self):
        result = self.run_script('scripts/board/stage0/run_board_robotd.zsh', 's2', MD_BOARD_ROBOTD_CPUS='')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.calls('taskset'), [])
        self.assertEqual(len(self.calls('robotd')), 1)


if __name__ == '__main__':
    unittest.main()
