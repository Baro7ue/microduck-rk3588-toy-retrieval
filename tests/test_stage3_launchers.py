"""Stage 3 zsh launchers (Mac tests, board brain, batch runner), with the fakes from test_launchers.py."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import unittest

import test_launchers
from test_launchers import RECORD, ZSH, put

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


# The batch runner's Python: the real interpreter for stage3.batch, a fake episode for stage3.episode.
BATCH_PY = RECORD + '''case "$*" in
  *microduck_toy.stage3.episode*) exec "$REAL_PY" "$FAKE_EPISODE" "$@" ;;
  *) exec "$REAL_PY" "$@" ;;
esac
'''
# One episode process: what it does on its n-th attempt comes from $FAKE_SCENARIO ({"3000:henry":
# ["crash", "done:grasp_fail"]}; default done:delivered). "interrupt" is a Ctrl-C at the terminal: SIGINT
# to the whole process group, then saving like the real runner does.
FAKE_EPISODE = '''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
get = lambda k: args[args.index(k) + 1]
item, out = get('--episodes'), Path(get('--out'))
seed, target = item.split(':')
board = '--target' in args and get('--target') == 'board'
counter = Path(os.environ['FAKE_LOG']) / ('attempts_' + item.replace(':', '_'))
n = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(n + 1))
plan = json.loads(Path(os.environ['FAKE_SCENARIO']).read_text()).get(item, []) if os.environ.get('FAKE_SCENARIO') else []
what = plan[n] if n < len(plan) else 'done:delivered'
if what == 'die':
    print('Traceback: fake crash before the run folder'); sys.exit(1)
out.mkdir(parents=True, exist_ok=False)
print('运行目录：' + str(out), flush=True)
label = 'e_%s_%s' % (seed, target)
meta = {'target': 'board' if board else 'mac'}
outcome = what.split(':', 1)[1] if what.startswith('done:') else 'delivered'
end = 'error' if what == 'brain_error' else ('dropped' if outcome == 'delivered' else outcome)
result = {'part': 'e', 'label': label, 'seed': int(seed), 'target': target, 'outcome': outcome, 'brain_end': end,
          'delivered_strict': True, 'elapsed_s': 30.0, 'closes': [], 'falls': [], 'tof_stops': [], 'states_s': {},
          'latency_e2e_p99_ms': 44.0 if board else None}
if board:
    result['robotd'] = {'ticks_expected': 1500, 'missed': 0, 'hz_min': 49.9}
status = 0
if what == 'interrupt':
    try:
        os.killpg(os.getpgrp(), signal.SIGINT)
        time.sleep(5)
    except KeyboardInterrupt:
        meta['interrupted'] = True
        status = 130
        print('已中断；保存已完成的部分。', flush=True)
elif what == 'crash':
    meta['error'] = 'RuntimeError: fake'
    status = 1
    print('出错：RuntimeError: fake', flush=True)
(out / 'episodes' / label).mkdir(parents=True)
(out / 'episodes' / label / 'brain.jsonl').write_text('{}\\n')
(out / 'summary.json').write_text(json.dumps({'meta': meta, 'sim_health': {'ok': True}, 'results': [result],
                                              'trials': [{'seed': int(seed), 'target': target}]}))
print('结果：' + str(out / 'summary.json'), flush=True)
sys.exit(status)
'''


@unittest.skipUnless(ZSH, 'zsh not installed')
class BatchLauncherTests(unittest.TestCase):
    setUp_base = test_launchers.LauncherTests.setUp
    setUp_stage3 = Stage3LauncherTests.setUp          # it calls setUp_base, then adds the toy models
    tearDown = test_launchers.LauncherTests.tearDown
    calls = test_launchers.LauncherTests.calls
    unix_socket = Stage3LauncherTests.unix_socket

    def setUp(self):
        self.setUp_stage3()
        put(self.root / 'third_party/microduck_rl/.venv/bin/python', BATCH_PY, 0o755)
        put(self.stubs / 'pkill', RECORD, 0o755)
        fake = put(self.log.parent / 'fake_episode.py', FAKE_EPISODE)
        self.scenario = self.log.parent / 'scenario.json'
        self.env.update(REAL_PY=sys.executable, FAKE_EPISODE=str(fake), FAKE_LSOF_STATUS='0')
        self.unix_socket(self.root.parent / 'state/robotd.sock')

    def batch(self, *args, scenario=None, **env):
        if scenario is not None:
            self.scenario.write_text(json.dumps(scenario))
            env['FAKE_SCENARIO'] = str(self.scenario)
        # its own session: the fake Ctrl-C signals the whole process group, which must not be the tests'
        return subprocess.run([ZSH, '-f', str(self.root / 'scripts/stage3/run_batch.zsh'), *args], cwd=self.cwd,
                              env={**self.env, **env}, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=120, start_new_session=True)

    def episodes(self):
        out = []
        for args, _ in self.calls('python'):
            if 'microduck_toy.stage3.episode' in args:
                rest = args[args.index('-m') + 1:]
                out.append(rest)
        return out

    def batch_dir(self, name):
        return self.root / 'runs' / name

    def test_mac_runs_each_episode_in_its_own_folder_then_prints_the_table(self):
        r = self.batch('--target', 'mac', '--seeds', '3000-3000')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        d = self.batch_dir('batch_mac_cpu_stage3_layouts_3000-3000')
        eps = self.episodes()
        self.assertEqual([e[e.index('--episodes') + 1] for e in eps], ['3000:bus', '3000:henry', '3000:ketchup'])
        first = eps[0]
        self.assertEqual(first[:4], ['microduck_toy.stage3.episode', '3e', '--source', 'stream'])
        self.assertEqual(first[first.index('--out') + 1], str(d / 'episodes/3000_bus'))
        self.assertEqual(first[first.index('--backend') + 1], 'cpu')
        self.assertEqual(first[first.index('--layouts') + 1], 'configs/control/stage3_layouts.json')
        self.assertNotIn('--target', first)
        for name in ('3000_bus', '3000_henry', '3000_ketchup'):
            self.assertTrue((d / 'episodes' / name / 'console.log').is_file(), name)
        self.assertIn('运行目录', (d / 'episodes/3000_bus/console.log').read_text())
        doc = json.loads((d / 'batch.json').read_text())
        self.assertEqual(doc['settings']['episodes'], ['3000:bus', '3000:henry', '3000:ketchup'])
        self.assertTrue((d / 'bundle/SHA256SUMS').is_file())
        self.assertFalse((d / '.bundle.new').exists())
        self.assertEqual(json.loads((d / 'summary.json').read_text())['done'], 3)
        self.assertIn('完成 3/3', r.stdout)
        self.assertIn('完成 3/3', (d / 'console.log').read_text())

    def test_a_resume_skips_results_and_runs_failed_attempts_again(self):
        scenario = {'3000:henry': ['crash', 'done:grasp_fail']}
        r = self.batch('--target', 'mac', '--seeds', '3000', scenario=scenario)
        self.assertEqual(r.returncode, 1, r.stderr + r.stdout)
        d = self.batch_dir('batch_mac_cpu_stage3_layouts_3000')
        self.assertTrue((d / 'episodes/3000_henry.infra1/console.log').is_file())
        self.assertIn('再运行同一条命令', r.stdout)
        r = self.batch('--target', 'mac', '--seeds', '3000', scenario=scenario)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual([e[e.index('--episodes') + 1] for e in self.episodes()],
                         ['3000:bus', '3000:henry', '3000:ketchup', '3000:henry'])
        self.assertIn('跳过已有结果的 2 个', r.stdout)
        self.assertIn('3000_henry.infra1', r.stdout)
        summary = json.loads((d / 'summary.json').read_text())
        self.assertEqual(summary['totals']['ends'], {'delivered': 2, 'grasp_fail': 1})
        self.assertEqual(len(json.loads((d / 'batch.json').read_text())['invocations']), 2)

    def test_three_failed_attempts_in_a_row_stop_the_batch(self):
        scenario = {k: ['crash'] for k in ('3000:bus', '3000:henry')} | {'3000:ketchup': ['die']}
        r = self.batch('--target', 'mac', '--seeds', '3000-3001', scenario=scenario)
        self.assertEqual(r.returncode, 2, r.stderr + r.stdout)
        self.assertEqual(len(self.episodes()), 3)
        d = self.batch_dir('batch_mac_cpu_stage3_layouts_3000-3001')
        self.assertTrue((d / 'episodes/3000_ketchup.incomplete1/console.log').is_file())
        self.assertIn('连续 3 个回合没算数', r.stdout)

    def test_ctrl_c_saves_the_episode_stops_and_the_same_command_goes_on(self):
        scenario = {'3000:henry': ['interrupt']}
        r = self.batch('--target', 'mac', '--seeds', '3000', scenario=scenario)
        self.assertEqual(r.returncode, 130, r.stderr + r.stdout)
        d = self.batch_dir('batch_mac_cpu_stage3_layouts_3000')
        aside = d / 'episodes/3000_henry.interrupted1'
        self.assertTrue((aside / 'summary.json').is_file())
        self.assertIn('已中断；保存已完成的部分', (aside / 'console.log').read_text())   # tee outlived the Ctrl-C
        self.assertEqual(len(self.episodes()), 2)                                      # ketchup not started
        self.assertIn('接着跑', r.stdout)
        self.assertIn('  ./scripts/stage3/run_batch.zsh --target mac --seeds 3000\n', r.stdout)   # pasteable
        self.assertIn('完成 1/3', r.stdout)
        r = self.batch('--target', 'mac', '--seeds', '3000', scenario=scenario)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual([e[e.index('--episodes') + 1] for e in self.episodes()[2:]], ['3000:henry', '3000:ketchup'])

    def test_a_resume_with_changed_code_is_refused(self):
        self.assertEqual(self.batch('--target', 'mac', '--seeds', '3000', '--targets', 'bus').returncode, 0)
        brain = self.root / 'src/microduck_toy/stage3/brain.py'
        brain.write_text(brain.read_text() + '# changed\n')
        r = self.batch('--target', 'mac', '--seeds', '3000', '--targets', 'bus')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('code_sha256', r.stderr + r.stdout)
        self.assertEqual(len(self.episodes()), 1)
        d = self.batch_dir('batch_mac_cpu_stage3_layouts_3000_bus')
        self.assertFalse((d / '.bundle.new').exists())

    def test_formal_seeds_only_with_formal(self):
        args = ('--target', 'mac', '--layouts', 'configs/control/stage1_layouts.json', '--seeds', '1000',
                '--targets', 'bus')
        r = self.batch(*args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('formal', r.stderr)
        self.assertEqual(self.episodes(), [])
        r = self.batch(*args, '--formal')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        (ep,) = self.episodes()
        self.assertIn('--formal', ep)
        self.assertTrue(self.batch_dir('batch_mac_cpu_stage1_layouts_1000_bus').is_dir())

    def test_board_sends_one_checked_bundle_and_runs_the_brain_there(self):
        self.unix_socket(self.board / 'run/robotd.sock')
        args = ('--target', 'board', '--seeds', '3003', '--targets', 'bus,henry', '--', '--no-frames')
        r = self.batch(*args)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        name = 'batch_board_int8_stage3_layouts_3003_bus+henry'
        sha = json.loads((self.batch_dir(name) / 'batch.json').read_text())['settings']['code_sha256']
        remote = self.board / 'bundles' / f'{name}_{sha[:12]}'
        self.assertTrue((remote / 'microduck_toy/stage3/brain.py').is_file())
        self.assertTrue((remote / 'models/yolo/yolo11n_toys/yolo11n_toys_int8.rknn').is_file())
        eps = self.episodes()
        self.assertEqual(len(eps), 2)
        first = eps[0]
        self.assertEqual(first[first.index('--target') + 1], 'board')
        self.assertEqual(first[first.index('--board-bundle') + 1], str(remote))
        self.assertTrue(first[first.index('--board-out') + 1].startswith(str(self.board / 'run' / name / '3003_bus_')))
        self.assertEqual(first[first.index('--board-cpus') + 1], '4-6')
        self.assertEqual(first[first.index('--backend') + 1], 'int8')
        self.assertEqual(first[-1], '--no-frames')
        doc = json.loads((self.batch_dir(name) / 'batch.json').read_text())
        self.assertEqual(doc['settings']['extra'], '--no-frames')
        self.assertEqual(len(self.calls('pkill')), 1)              # a brain left over from a Ctrl-C
        self.assertIn('robotd 漏拍 0/3000', r.stdout)
        scp = len(self.calls('scp'))
        r = self.batch(*args)                                       # nothing left: the bundle is not sent again
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(len(self.calls('scp')), scp)
        self.assertEqual(len(self.episodes()), 2)

    def test_bad_arguments(self):
        for args in (('--target', 'x'), ('--target', 'mac', '--backend', 'int8'), ('--target', 'board', '--backend', 'gpu'),
                     ('--target', 'mac', '--', '--episodes', '3000:bus'), ('--target', 'mac', '--wat'),
                     ('--target', 'mac', '--seeds')):
            with self.subTest(args=args):
                self.assertNotEqual(self.batch(*args).returncode, 0)
        self.assertEqual(self.episodes(), [])

    def test_stops_before_any_episode_without_the_body_or_robotd(self):
        self.assertNotEqual(self.batch('--target', 'mac', FAKE_LSOF_STATUS='1').returncode, 0)
        os.unlink(self.root.parent / 'state/robotd.sock')
        r = self.batch('--target', 'mac')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('run_robotd.zsh s2', r.stderr)
        r = self.batch('--target', 'board')
        self.assertIn('run_board_robotd.zsh s2', r.stderr)
        self.assertEqual(self.episodes(), [])


if __name__ == '__main__':
    unittest.main()
