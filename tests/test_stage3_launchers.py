"""Stage 3 zsh launchers (Mac tests, board brain), with the fakes from test_launchers.py."""
from __future__ import annotations

import socket
import unittest

import test_launchers
from test_launchers import ZSH, put

TOY_MODELS = ('yolo11n_toys_split_320.onnx', 'yolo11n_toys_fp16.rknn', 'yolo11n_toys_int8.rknn')


@unittest.skipUnless(ZSH, 'zsh not installed')
class Stage3LauncherTests(unittest.TestCase):
    setUp_base = test_launchers.LauncherTests.setUp
    tearDown = test_launchers.LauncherTests.tearDown
    run_script = test_launchers.LauncherTests.run_script
    calls = test_launchers.LauncherTests.calls
    runs = test_launchers.LauncherTests.runs

    def setUp(self):
        self.setUp_base()
        for name in TOY_MODELS:
            put(self.root / 'models/yolo/yolo11n_toys' / name, name)

    def unix_socket(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        self.addCleanup(sock.close)

    def modules(self):
        return [(args[args.index('-m') + 1], args[args.index('-m') + 2:]) for args, _ in self.calls('python')]

    # ── Mac ─────────────────────────────────────────────────────────────

    def test_mac_runs_the_episode_module_from_the_repo(self):
        self.unix_socket(self.root.parent / 'state/robotd.sock')
        result = self.run_script('scripts/sim/stage3/run_test.zsh', '3e', '--quick', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        (args, env), = self.calls('python')
        self.assertEqual(args[0], str(self.root.resolve()))
        self.assertEqual(args[args.index('-m') + 1:], ['microduck_toy.stage3.episode', '3e', '--source', 'stream',
                                                        '--quick'])
        self.assertIn('offline/vision-export/site-packages', env['PYTHONPATH'])

    def test_mac_stops_without_the_body_or_robotd(self):
        self.assertNotEqual(self.run_script('scripts/sim/stage3/run_test.zsh', '3ab').returncode, 0)
        result = self.run_script('scripts/sim/stage3/run_test.zsh', '3ab', FAKE_LSOF_STATUS='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('run_robotd.zsh s2', result.stderr)
        self.assertEqual(self.calls('python'), [])

    def test_mac_layouts_and_rescore_need_no_robot(self):
        self.assertEqual(self.run_script('scripts/sim/stage3/run_test.zsh', 'layouts').returncode, 0)
        self.assertEqual(self.run_script('scripts/sim/stage3/run_test.zsh', '3e', '--rescore',
                                         'runs/x').returncode, 0)
        self.assertEqual([m for m, _ in self.modules()], ['microduck_toy.stage3.episode'] * 2)
        self.assertEqual(self.modules()[1][1], ['3e', '--rescore', 'runs/x'])

    def test_mac_refuses_unknown_tests_and_board_targets(self):
        for args in (('3x',), ('3e', '--target', 'board')):
            self.assertNotEqual(self.run_script('scripts/sim/stage3/run_test.zsh', *args,
                                                FAKE_LSOF_STATUS='0').returncode, 0)
        self.assertEqual(self.calls('python'), [])

    # ── board ───────────────────────────────────────────────────────────

    def test_board_sends_a_checked_bundle_then_runs_the_first_9(self):
        self.unix_socket(self.board / 'run/robotd.sock')
        result = self.run_script('scripts/board/stage3/run_board_brain.zsh', '3e', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run = next(p for p in self.runs() if p.name.endswith('_s3_board_bundle'))
        bundle = self.board / 'bundles' / run.name
        for rel in ('microduck_toy/stage3/brain.py', 'configs/control/stage3.json',
                    'configs/control/stage1_facts.json', 'configs/vision/stage2.json',
                    'models/yolo/yolo11n_toys/yolo11n_toys_int8.rknn', 'SHA256SUMS'):
            self.assertTrue((bundle / rel).is_file(), rel)
        self.assertTrue((self.board / 'run' / run.name).is_dir())
        (module, args), = self.modules()
        self.assertEqual(module, 'microduck_toy.stage3.episode')
        self.assertEqual(args[:4], ['3e', '--target', 'board', '--source'])
        self.assertEqual(args[args.index('--board-bundle') + 1], str(bundle))
        self.assertEqual(args[args.index('--board-out') + 1], str(self.board / 'run' / run.name))
        eps = args[args.index('--episodes') + 1].split(',')
        self.assertEqual(len(eps), 9)
        self.assertEqual(eps[:3], ['3000:bus', '3000:henry', '3000:ketchup'])
        self.assertTrue((run / 'console.log').is_file())

    def test_board_pins_the_brain_to_md_board_cpus(self):
        self.unix_socket(self.board / 'run/robotd.sock')
        result = self.run_script('scripts/board/stage3/run_board_brain.zsh', '3e', FAKE_LSOF_STATUS='0',
                                 MD_BOARD_CPUS='4-7')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        (_, args), = self.modules()
        self.assertEqual(args[args.index('--board-cpus') + 1], '4-7')

    def test_board_pins_to_three_big_cores_when_md_board_cpus_is_unset(self):
        self.unix_socket(self.board / 'run/robotd.sock')
        for given, want in ((None, '4-6'), ('', '')):            # core 7 is robotd's
            extra = {} if given is None else {'MD_BOARD_CPUS': given}
            result = self.run_script('scripts/board/stage3/run_board_brain.zsh', '3e', FAKE_LSOF_STATUS='0', **extra)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            _, args = self.modules()[-1]
            self.assertEqual(args[args.index('--board-cpus') + 1], want)

    def test_board_keeps_given_episodes(self):
        self.unix_socket(self.board / 'run/robotd.sock')
        result = self.run_script('scripts/board/stage3/run_board_brain.zsh', '3e', '--episodes', '3001:henry',
                                 '--mac-run', 'runs/mac', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        (_, args), = self.modules()
        self.assertEqual(args.count('--episodes'), 1)
        self.assertEqual(args[args.index('--episodes') + 1], '3001:henry')
        self.assertEqual(args[args.index('--mac-run') + 1], 'runs/mac')

    def test_board_stops_without_models_ports_or_robotd(self):
        self.assertNotEqual(self.run_script('scripts/board/stage3/run_board_brain.zsh', '3e',
                                            FAKE_LSOF_STATUS='0').returncode, 0)       # no robotd.sock
        self.unix_socket(self.board / 'run/robotd.sock')
        self.assertNotEqual(self.run_script('scripts/board/stage3/run_board_brain.zsh', '3e').returncode, 0)
        (self.root / 'models/yolo/yolo11n_toys/yolo11n_toys_int8.rknn').unlink()
        result = self.run_script('scripts/board/stage3/run_board_brain.zsh', '3e', FAKE_LSOF_STATUS='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('yolo11n_toys_int8.rknn', result.stderr)
        self.assertEqual(self.calls('python'), [])


if __name__ == '__main__':
    unittest.main()
