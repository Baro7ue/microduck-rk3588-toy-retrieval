"""env.sh and the zsh launchers, run in a throwaway copy whose path contains a space.

ssh, scp, git, lsof, cargo and the Python/robotd programs are replaced by small shell fakes.
The fake ssh runs the remote command with bash on a local "board" folder, so the board side
(sha256sum -c, PYTHONPATH, output folder) is exercised too.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
ZSH = shutil.which('zsh')
RUNTIME_COMMIT = '43d09674221da03e1a8f8f6f5cf66c21a9978b2c'
RL_COMMIT = 'cb70b792312d559a4da09064d92009079671815f'

# Records cwd + argv (NUL separated) and the environment of every call into $FAKE_LOG.
RECORD = '''#!/bin/sh
n=$(ls "$FAKE_LOG" | wc -l | tr -d ' ')
f="$FAKE_LOG/$(basename "$0").$n"
{ pwd; printf '%s\\n' "$@"; } > "$f.args"
env > "$f.env"
'''
FAKES = {
    'git': RECORD + '[ "$1" = -C ] && cd "$2"; cat .fake_head\n',
    'lsof': RECORD + 'exit "${FAKE_LSOF_STATUS:-1}"\n',
    'cargo': RECORD + 'echo cargo-fake\n',
    'rustc': RECORD + 'echo rustc-fake\n',
    'ssh': RECORD + '''while [ $# -gt 0 ]; do
  case $1 in -t) shift ;; -o|-L) shift 2 ;; *) break ;; esac
done
shift
exec bash -c "$1"
''',
    'scp': RECORD + '''while [ $# -gt 0 ]; do case $1 in -*) shift ;; *) break ;; esac; done
exec cp -R "${1#*:}" "${2#*:}"
''',
    'taskset': RECORD + 'shift 2\nexec "$@"\n',     # taskset -c LIST CMD...: run CMD (macOS has none)
}
MAC_PY = RECORD + 'cat > "$f.stdin"\necho fake-python\nexit "${FAKE_PY_STATUS:-0}"\n'
MJPYTHON = RECORD + 'echo "== startup assistance: fake"\n'
ROBOTD = RECORD + 'echo robotd-fake\n'
BOARD_PY = RECORD + '''out=''
while [ $# -gt 0 ]; do [ "$1" = --output-dir ] && out=$2; shift; done
mkdir -p "$out" && echo '{}' > "$out/detections.jsonl"
'''
BOARD_PYTHON3 = RECORD + '''mkdir -p "$MD_BOARD_ROOT/run/resources_fake" "$MD_BOARD_ROOT/run/resources_old"
echo samples > "$MD_BOARD_ROOT/run/resources_fake/summary.json"
printf '资源证据目录：%s\\r\\n' "$MD_BOARD_ROOT/run/resources_fake"
'''


def put(path: Path, text: str, mode: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode is not None:
        path.chmod(mode)
    return path


@unittest.skipUnless(ZSH, 'zsh not installed')
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name).resolve()   # macOS: /var is a link to /private/var, and launchers resolve links
        self.root = base / 'RK3588 Microduck copy'
        self.root.mkdir()
        for name in ('env.sh', 'scripts', 'src', 'configs', 'cpp'):
            src = REPO / name
            if src.is_dir():
                shutil.copytree(src, self.root / name, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copy(src, self.root / name)
        models = self.root / 'models/yolo/yolo11n_coco'
        for name in ('yolo11n_coco_fp32_320.onnx', 'export.json',
                     'yolo11n_coco_fp16_320.rknn', 'conversion.json'):
            put(models / name, name)
        for name in ('walk_17500.onnx', 'velstand.onnx'):
            put(self.root / 'models/policies' / name, name)
        runtime, rl = self.root / 'third_party/microduck', self.root / 'third_party/microduck_rl'
        put(runtime / '.fake_head', RUNTIME_COMMIT + '\n')
        put(rl / '.fake_head', RL_COMMIT + '\n')
        put(runtime / 'target/debug/robotd', ROBOTD, 0o755)
        put(rl / '.venv/bin/python', MAC_PY, 0o755)
        put(rl / '.venv/bin/mjpython', MJPYTHON, 0o755)
        put(rl / '.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.24.4.dylib', '')
        put(self.root / 'sim/scenes/demo/scene.xml', '<mujoco/>')

        self.stubs = base / 'stubs'
        for name, text in FAKES.items():
            put(self.stubs / name, text, 0o755)
        self.board = base / 'board'
        put(self.board / 'bin/robotd', ROBOTD, 0o755)
        put(self.board / 'bin/board_python', BOARD_PY, 0o755)
        (self.board / 'run').mkdir()

        self.log = base / 'log'
        self.log.mkdir()
        self.cwd = base / 'elsewhere'
        self.cwd.mkdir()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('MD_') and k not in ('PYTHONPATH', 'ZDOTDIR')}
        env.update(PATH=f'{self.stubs}:{os.environ["PATH"]}', HOME=str(base / 'home'),
                   FAKE_LOG=str(self.log), MD_MAC_STATE=str(base / 'state'),
                   MD_STARTUP_SOCKET=str(base / 'startup.sock'), MD_BOARD_ROOT=str(self.board),
                   MD_BOARD_ROBOTD=str(self.board / 'bin/robotd'),
                   MD_BOARD_PY=str(self.board / 'bin/board_python'),
                   MD_BOARD_ORT=str(self.board / 'lib/libonnxruntime.so.1.28.0'))
        self.env = env

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, rel: str, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([ZSH, '-f', str(self.root / rel), *args], cwd=self.cwd,
                              env={**self.env, **env}, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=60)

    def calls(self, name: str) -> list[tuple[list[str], dict]]:
        out = []
        files = sorted(self.log.glob(f'{name}.*.args'), key=lambda p: int(p.name.split('.')[-2]))
        for path in files:
            lines = path.read_text().split('\n')[:-1]
            env = dict(line.split('=', 1) for line in
                       Path(str(path)[:-5] + '.env').read_text().splitlines() if '=' in line)
            out.append((lines, env))
        return out

    def runs(self) -> list[Path]:
        return sorted(p for p in (self.root / 'runs').iterdir() if p.is_dir())

    def test_env_sh_from_zsh_and_bash(self):
        for shell in (ZSH, shutil.which('bash')):
            quoted = "'" + str(self.root / 'env.sh') + "'"
            command = (f'source {quoted} && source {quoted} && '
                       'printf "%s\\n" "$MD_ROOT" "$MD_RUNS" "$MD_RL" "$PYTHONPATH"')
            result = subprocess.run([shell, '-c', command], cwd=self.cwd, capture_output=True,
                                    text=True, env={**self.env, 'PYTHONPATH': '/x'})
            self.assertEqual(result.returncode, 0, result.stderr)
            root = str(self.root.resolve())
            self.assertEqual(result.stdout.splitlines(),
                             [root, root + '/runs', root + '/third_party/microduck_rl',
                              root + '/src:/x'], shell)

    def test_env_sh_board_paths_follow_the_board_root(self):
        names = ('MD_BOARD_ROOT', 'MD_BOARD_ROBOTD', 'MD_BOARD_ORT', 'MD_BOARD_PY')
        command = (f"source '{self.root}/env.sh' && printf '%s\\n' "
                   + ' '.join(f'"${name}"' for name in names) + ' "${MD_BOARD_SITE-unset}"')
        env = {k: v for k, v in self.env.items() if not k.startswith('MD_BOARD_')}
        for board in ('/home/linaro/microduck', '/x'):
            run_env = env if board == '/home/linaro/microduck' else {**env, 'MD_BOARD_ROOT': board}
            result = subprocess.run([ZSH, '-c', command], cwd=self.cwd, env=run_env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             [board, f'{board}/runtime/bin/robotd',
                              f'{board}/runtime/onnxruntime-linux-aarch64-1.28.0/lib/libonnxruntime.so.1.28.0',
                              f'{board}/runtime/venv/bin/python', 'unset'])

    def test_sim_body_defaults_for_the_board(self):
        result = self.run_script('scripts/sim/run_sim_body.zsh')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, env), = self.calls('mjpython')
        root = str(self.root.resolve())
        self.assertEqual(args, [root + '/third_party/microduck_rl',
                                root + '/src/microduck_toy/sim/bootstrap_body.py',
                                '--ducks', '1', '--host', '192.168.50.1', '--port', '7801',
                                '--keyframe', 'HOME', '--cameras', 'a', '--frame-port', '7901',
                                '--camera-fps', '15', '--headless'])
        self.assertEqual(env['MD_STARTUP_SOCKET'], self.env['MD_STARTUP_SOCKET'])
        self.assertEqual(env['PYTHONUNBUFFERED'], '1')
        run, = self.runs()
        self.assertRegex(run.name, r'^\d{8}T\d{6}Z_sim_body$')
        self.assertEqual((self.root / 'runs/current_sim_body_log').read_text().strip(),
                         str(run / 'body.log'))
        self.assertIn('startup assistance', (run / 'body.log').read_text())

    def test_sim_body_local_scene_without_camera_with_viewer(self):
        result = self.run_script('scripts/sim/run_sim_body.zsh', '--scene',
                                 str(self.root / 'sim/scenes/demo/scene.xml'),
                                 '--local', '--no-camera', '--viewer')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, env), = self.calls('mjpython')
        self.assertEqual(args[2:], ['--scene', str(self.root.resolve() / 'sim/scenes/demo/scene.xml'),
                                    '--ducks', '1', '--host', '127.0.0.1', '--port', '7801',
                                    '--keyframe', 'HOME'])
        self.assertEqual(env['MD_STARTUP_SOCKET'], self.env['MD_MAC_STATE'] + '/robotd.sock')

    def test_sim_body_refuses_wrong_commit_busy_port_and_bad_flags(self):
        put(self.root / 'third_party/microduck_rl/.fake_head', 'deadbeef\n')
        self.assertNotEqual(self.run_script('scripts/sim/run_sim_body.zsh').returncode, 0)
        put(self.root / 'third_party/microduck_rl/.fake_head', RL_COMMIT + '\n')
        self.assertNotEqual(self.run_script('scripts/sim/run_sim_body.zsh',
                                            FAKE_LSOF_STATUS='0').returncode, 0)
        self.assertNotEqual(self.run_script('scripts/sim/run_sim_body.zsh', '--bogus').returncode, 0)
        self.assertEqual(self.calls('mjpython'), [])

    def test_build_mac_runtime(self):
        result = self.run_script('scripts/sim/build_mac_runtime.zsh')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, _), = [call for call in self.calls('cargo') if 'build' in call[0]]
        self.assertEqual(args, [str(self.root.resolve() / 'third_party/microduck'),
                                'build', '--locked', '-p', 'robotd', '-p', 'robotctl', '-p', 'tof'])
        run, = self.runs()
        self.assertIn('cargo-fake', (run / 'build.log').read_text())

    def test_mac_robotd_renders_config_with_spaces_in_path(self):
        self.assertNotEqual(self.run_script('scripts/sim/run_mac_robotd.zsh').returncode, 0)
        result = self.run_script('scripts/sim/run_mac_robotd.zsh', FAKE_LSOF_STATUS='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, env), = self.calls('robotd')
        run = self.runs()[-1]
        root = str(self.root.resolve())
        self.assertEqual(args[1:], ['--sim', '127.0.0.1:7801', '--params', str(run / 'robotd.toml'),
                                    '--socket', self.env['MD_MAC_STATE'] + '/robotd.sock'])
        self.assertTrue(env['ORT_DYLIB_PATH'].endswith('libonnxruntime.1.24.4.dylib'))
        self.assertEqual(env['DUCK_RUNTIME_DIR'], self.env['MD_MAC_STATE'])
        toml = (run / 'robotd.toml').read_text()
        self.assertIn(f'walk = "{root}/models/policies/walk_17500.onnx"', toml)
        self.assertNotIn('@MD_ROOT@', toml.split('\n', 3)[3])

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_board_robotd_ships_config_and_policies_and_copies_its_log_back(self):
        result = self.run_script('scripts/board/run_board_robotd.zsh')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run, = self.runs()
        self.assertRegex(run.name, r'^\d{8}T\d{6}Z_board_robotd$')
        remote = self.board / 'bundles' / run.name
        (ssh_args, _), = [call for call in self.calls('ssh') if '-L' in call[0]]
        self.assertIn(f'{self.env["MD_STARTUP_SOCKET"]}:{self.board}/run/robotd.sock', ssh_args)
        self.assertIn('sha256sum -c --quiet SHA256SUMS &&', '\n'.join(ssh_args))
        (args, env), = self.calls('robotd')
        self.assertEqual(args, [str(remote), '--sim', '192.168.50.1:7801',
                                '--params', f'{remote}/robotd.toml',
                                '--socket', f'{self.board}/run/robotd.sock'])
        self.assertEqual(env['ORT_DYLIB_PATH'], self.env['MD_BOARD_ORT'])
        self.assertEqual(env['DUCK_RUNTIME_DIR'], f'{self.board}/run')
        toml = (remote / 'robotd.toml').read_text()
        self.assertEqual((run / 'bundle/robotd.toml').read_text(), toml)
        self.assertIn(f'walk = "{remote}/policies/walk_17500.onnx"', toml)
        self.assertIn(f'stand = "{remote}/policies/velstand.onnx"', toml)
        self.assertNotIn('@POLICY_DIR@', toml)
        self.assertNotIn(str(remote), toml.split('\n', 3)[1])   # comments stay readable
        self.assertEqual(sorted(p.name for p in (remote / 'policies').iterdir()),
                         ['velstand.onnx', 'walk_17500.onnx'])
        self.assertEqual((remote / 'policies/walk_17500.onnx').read_text(), 'walk_17500.onnx')
        sums = (remote / 'SHA256SUMS').read_text()
        for name in ('robotd.toml', 'policies/walk_17500.onnx', 'policies/velstand.onnx'):
            self.assertIn('  ' + name + '\n', sums)
        self.assertIn('robotd-fake', (self.board / 'run' / run.name / 'robotd.log').read_text())
        self.assertIn('robotd-fake', (run / 'board/robotd.log').read_text())

        (self.root / 'models/policies/velstand.onnx').unlink()
        ssh_calls = len(self.calls('ssh'))
        result = self.run_script('scripts/board/run_board_robotd.zsh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('velstand.onnx', result.stderr)
        self.assertEqual(len(self.calls('ssh')), ssh_calls)

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_board_yolo_cpu_and_npu_bundles(self):
        for flags, backend, module, model in (
                ((), 'cpu', 'microduck_toy.vision.yolo_camera_probe', 'yolo11n_coco_fp32_320.onnx'),
                (('--npu', '--duration', '5'), 'npu', 'microduck_toy.vision.rknn_runner',
                 'yolo11n_coco_fp16_320.rknn')):
            result = self.run_script('scripts/board/probe_board_yolo_camera.zsh', *flags)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            run = [p for p in self.runs() if p.name.endswith('_board_yolo_' + backend)][-1]
            args, env = self.calls('board_python')[-1]
            remote = self.board / 'bundles' / run.name
            self.assertEqual(args, [str(remote), '-u', '-m', module, '--model', 'models/' + model,
                                    '--duration', flags[-1] if flags else '30',
                                    '--output-dir', f'{self.board}/run/{run.name}'])
            self.assertEqual(env['PYTHONPATH'], str(remote))
            self.assertTrue((remote / 'microduck_toy/vision/yolo_camera_probe.py').is_file())
            self.assertFalse(list(remote.rglob('__pycache__')))
            self.assertTrue((run / 'board/detections.jsonl').is_file())
            self.assertIn('实验退出码：0', (run / 'console.log').read_text())
            remote_command = '\n'.join(self.calls('ssh')[-1][0])
            self.assertIn('sha256sum -c --quiet SHA256SUMS &&', remote_command)
            sums = (remote / 'SHA256SUMS').read_text()
            self.assertIn('  models/' + model, sums)
            self.assertIn('  microduck_toy/paths.py', sums)

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_stand_then_camera_and_stop_on_failed_stand(self):
        self.assertNotEqual(self.run_script('scripts/board/probe_board_stand_camera.zsh').returncode, 0)
        body_log = put(self.root / 'runs/x_sim_body/body.log', 'log')
        put(self.root / 'runs/current_sim_body_log', str(body_log) + '\n')
        result = self.run_script('scripts/board/probe_board_stand_camera.zsh', FAKE_PY_STATUS='3')
        self.assertEqual(result.returncode, 3)
        self.assertEqual(self.calls('ssh'), [])
        result = self.run_script('scripts/board/probe_board_stand_camera.zsh', '--npu')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        args, env = self.calls('python')[-1]
        self.assertEqual(args[1:], ['-u', '-', str(body_log)])
        self.assertEqual(env['MD_ROBOT_SOCKET'], self.env['MD_STARTUP_SOCKET'])
        self.assertTrue(env['MD_RUNS'].endswith('_board_stand'))
        stdin = Path(str(sorted(self.log.glob('python.*.args'))[-1])[:-5] + '.stdin').read_text()
        self.assertIn('from microduck_toy.rpc import stand_probe as stand', stdin)
        self.assertTrue([p for p in self.runs() if p.name.endswith('_board_yolo_npu')])

    @unittest.skipUnless(shutil.which('sha256sum') and shutil.which('bash'), 'needs GNU sha256sum')
    def test_resources_copies_only_the_printed_directory(self):
        put(self.stubs / 'python3', BOARD_PYTHON3, 0o755)
        result = self.run_script('scripts/board/measure_board_resources.zsh')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        run, = self.runs()
        self.assertEqual((run / 'board/summary.json').read_text(), 'samples\n')
        args, _ = self.calls('python3')[-1]
        self.assertEqual(args[1:], ['-u', 'measure_board_resources.py', '--duration', '660'])

    def test_run_tests_adds_vision_site_after_src(self):
        result = self.run_script('scripts/tools/run_tests.zsh', '-v')
        self.assertEqual(result.returncode, 0, result.stderr)
        (args, env), = self.calls('python')
        root = str(self.root.resolve())
        self.assertEqual(args, [root, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        self.assertEqual(env['PYTHONPATH'],
                         f'{root}/src:{root}/offline/vision-export/site-packages')


class ExecutableBitTests(unittest.TestCase):
    """Every launcher is run as scripts/.../x.zsh, so git must store it as 100755 (lib.zsh is
    only sourced). The tests above run them through zsh, which does not need the bit."""

    def test_launchers_are_executable(self):
        if (REPO / '.git').exists() and shutil.which('git'):
            out = subprocess.run(['git', '-C', str(REPO), 'ls-files', '-s', '--', 'scripts'],
                                 capture_output=True, text=True, check=True).stdout
            modes = {line.split('\t', 1)[1]: line.split()[0] for line in out.splitlines()}
        else:
            modes = {str(p.relative_to(REPO)): '100755' if os.access(p, os.X_OK) else '100644'
                     for p in (REPO / 'scripts').rglob('*.zsh')}
        bad = sorted(path for path, mode in modes.items()
                     if path.endswith('.zsh') and Path(path).name != 'lib.zsh' and mode != '100755')
        self.assertEqual(bad, [], 'chmod +x these launchers (git update-index --chmod=+x)')


if __name__ == '__main__':
    unittest.main()
