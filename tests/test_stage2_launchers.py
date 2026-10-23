"""Stage 2 zsh launchers (Mac parity, board NPU evaluation), with the fakes from test_launchers.py."""
from __future__ import annotations

import socket
import unittest

import test_launchers
from test_launchers import ZSH, put

# Board Python that writes the --out file it is given (the stage 2 predict command line).
BOARD_PREDICT = test_launchers.RECORD + '''out=''
while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
mkdir -p "$(dirname "$out")" && echo '{"id": "x", "dets": []}' > "$out"
'''

# Board Python for the deployment run: writes <--out>/summary.json.
BOARD_DEPLOY = test_launchers.RECORD + '''out=''
while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
mkdir -p "$out" && echo '{"pass": true}' > "$out/summary.json"
'''


@unittest.skipUnless(ZSH, 'zsh not installed')
class Stage2LauncherTests(unittest.TestCase):
    setUp_base = test_launchers.LauncherTests.setUp
    tearDown = test_launchers.LauncherTests.tearDown
    run_script = test_launchers.LauncherTests.run_script
    calls = test_launchers.LauncherTests.calls
    runs = test_launchers.LauncherTests.runs

    def setUp(self):
        self.setUp_base()
        self.pack = self.root / 'runs/20261024T120000Z_s2e_pack'
        put(self.pack / 'pack.json', '{}')
        put(self.pack / 'parity_ids.txt', 'test_300000\n')
        put(self.pack / 'stage2.json', (self.root / 'configs/vision/stage2.json').read_text())
        for name in ('yolo11n_toys_split_320.onnx', 'yolo11n_toys_fp16.rknn', 'yolo11n_toys_int8.rknn'):
            put(self.pack / 'models' / name, name)
        for split in ('test', 'test_far', 'test_walk'):
            put(self.pack / 'data/meta' / f'{split}.jsonl', '')
            put(self.pack / 'data/images' / split / f'{split}_0.png', 'png')

    def test_parity_renders_predicts_twice_and_compares_twice(self):
        result = self.run_script('scripts/vision/stage2/run_parity_mac.zsh', str(self.pack))
        self.assertEqual(result.returncode, 0, result.stderr)
        run = next(p for p in self.runs() if p.name.endswith('_s2e_mac_parity'))
        calls = self.calls('python')
        modules = [args[args.index('-m') + 1] for args, _ in calls]
        self.assertEqual(modules, ['microduck_toy.stage2.datagen', 'microduck_toy.stage2.predict',
                                   'microduck_toy.stage2.predict', 'microduck_toy.stage2.parity',
                                   'microduck_toy.stage2.parity'])
        render, cloud_frames, mac_frames, same, rendered = (args for args, _ in calls)
        self.assertEqual(render[-6:], ['--out', str(run / 'render'), '--splits', 'test', '--count', '200'])
        self.assertIn('--ids', cloud_frames)
        self.assertEqual(cloud_frames[cloud_frames.index('--data') + 1], str(self.pack / 'data'))
        self.assertEqual(mac_frames[mac_frames.index('--data') + 1], str(run / 'render'))
        model = str(self.pack / 'models/yolo11n_toys_split_320.onnx')
        self.assertEqual(cloud_frames[cloud_frames.index('--model') + 1], model)
        self.assertEqual(same[same.index('--rule') + 1], 'mac')
        self.assertEqual(rendered[rendered.index('--new-images') + 1], str(run / 'render/images/test'))
        self.assertIn('offline/vision-export/site-packages', calls[0][1]['PYTHONPATH'])
        self.assertTrue((run / 'console.log').is_file())

    def test_parity_refuses_a_stale_config_or_no_pack(self):
        put(self.pack / 'stage2.json', '{}')
        self.assertNotEqual(self.run_script('scripts/vision/stage2/run_parity_mac.zsh',
                                            str(self.pack)).returncode, 0)
        self.assertNotEqual(self.run_script('scripts/vision/stage2/run_parity_mac.zsh',
                                            str(self.root)).returncode, 0)
        self.assertEqual(self.calls('python'), [])

    def test_board_runs_both_models_on_three_splits_then_scores(self):
        put(self.board / 'bin/board_python', BOARD_PREDICT, 0o755)
        result = self.run_script('scripts/board/stage2/run_eval_board.zsh', str(self.pack))
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run = next(p for p in self.runs() if p.name.endswith('_s2e_board_npu'))
        board = self.calls('board_python')
        self.assertEqual(len(board), 6)
        first = board[0][0]
        self.assertEqual(first[1:9], ['-u', '-m', 'microduck_toy.stage2.predict', '--data', 'data',
                                      '--split', 'test', '--detector'])
        self.assertEqual(first[first.index('--model') + 1], 'models/yolo11n_toys_fp16.rknn')
        self.assertEqual(board[0][1]['MD_ROOT'], str(self.board / 'bundles' / run.name))
        bundle = self.board / 'bundles' / run.name
        for rel in ('microduck_toy/stage2/predict.py', 'configs/vision/stage2.json',
                    'models/yolo11n_toys_int8.rknn', 'data/images/test_walk/test_walk_0.png'):
            self.assertTrue((bundle / rel).is_file(), rel)
        for m in ('fp16', 'int8'):
            for s in ('test', 'test_far', 'test_walk'):
                self.assertTrue((run / 'board' / f'npu_{m}_{s}.jsonl').is_file())
        modules = [args[args.index('-m') + 1] for args, _ in self.calls('python')]
        self.assertEqual(modules, (['microduck_toy.stage2.evaluate'] * 3
                                   + ['microduck_toy.stage2.parity'] * 2) * 2)

    def board_socket(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.board / 'run/robotd.sock'))
        self.addCleanup(sock.close)

    def test_deploy_runs_each_backend_on_the_board_then_the_table(self):
        put(self.board / 'bin/board_python', BOARD_DEPLOY, 0o755)
        put(self.stubs / 'sudo', test_launchers.RECORD, 0o755)
        self.board_socket()
        result = self.run_script('scripts/board/stage2/run_deploy_board.zsh', str(self.pack),
                                 '--frames', '50', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run = next(p for p in self.runs() if p.name.endswith('_s2f_board_deploy'))
        board = self.calls('board_python')
        self.assertEqual([a[a.index('--backend') + 1] for a, _ in board], ['cpu', 'fp16', 'int8'])
        cpu = board[0][0]
        self.assertEqual(cpu[1:4], ['-u', '-m', 'microduck_toy.stage2.deploy'])
        self.assertEqual(cpu[cpu.index('--model') + 1], 'models/yolo11n_toys_split_320.onnx')
        self.assertEqual(cpu[cpu.index('--camera') + 1], '192.168.50.1:7901')
        self.assertEqual(cpu[cpu.index('--socket') + 1], str(self.board / 'run/robotd.sock'))
        self.assertEqual(cpu[cpu.index('--frames') + 1], '50')
        self.assertEqual(board[2][0][board[2][0].index('--model') + 1], 'models/yolo11n_toys_int8.rknn')
        bundle = self.board / 'bundles' / run.name
        for rel in ('microduck_toy/stage2/deploy.py', 'configs/vision/stage2.json',
                    'configs/control/stage1.json', 'models/yolo11n_toys_fp16.rknn'):
            self.assertTrue((bundle / rel).is_file(), rel)
        for b in ('cpu', 'fp16', 'int8'):
            self.assertTrue((run / 'board' / b / 'summary.json').is_file())
        mac = [args for args, _ in self.calls('python')]
        self.assertEqual([a[a.index('-m') + 1] for a in mac], ['microduck_toy.stage2.deploy'] * 2)
        self.assertIn('--place', mac[0])
        self.assertEqual(mac[1][mac[1].index('--table') + 1], str(run))
        self.assertTrue((run / 'layout.json').is_file())

    def test_deploy_stops_without_the_camera_or_board_robotd(self):
        self.assertNotEqual(self.run_script('scripts/board/stage2/run_deploy_board.zsh',
                                            str(self.pack)).returncode, 0)
        result = self.run_script('scripts/board/stage2/run_deploy_board.zsh', str(self.pack),
                                 FAKE_LSOF_STATUS='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('run_board_robotd.zsh s2', result.stderr)
        self.assertEqual(self.calls('board_python'), [])
        self.assertNotEqual(self.run_script('scripts/board/stage2/run_deploy_board.zsh', str(self.pack),
                                            '--backends', 'gpu', FAKE_LSOF_STATUS='0').returncode, 0)


if __name__ == '__main__':
    unittest.main()
