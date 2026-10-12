"""docs/reorg/board_reorg.zsh (Mac driver) and board_reorg_remote.sh (board side), against a fake board.

ssh and scp are fakes: ssh runs the remote command with bash on this machine, so the board side
(SHA256SUMS check, moves, copies, the move log, undo) really runs, on a throwaway folder that
MD_REORG_BOARD_HOME points the board script at instead of $HOME.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
ZSH = shutil.which('zsh')


def board_like_shell() -> bool:
    """The board script needs bash >= 4 and GNU tools (Debian on the board; macOS has bash 3.2)."""
    if not (shutil.which('bash') and shutil.which('sha256sum') and shutil.which('tac')):
        return False
    major = subprocess.run(['bash', '-c', 'echo ${BASH_VERSINFO[0]}'], capture_output=True, text=True)
    return major.stdout.strip().isdigit() and int(major.stdout) >= 4


NEW_ROBOTD = '/home/linaro/microduck/runtime/bin/robotd'
OLD_ROBOTD = '/home/linaro/microduck-g3-offline/microduck/target/release/robotd'

FAKES = {
    'ssh': '''#!/bin/sh
while [ $# -gt 0 ]; do
  case $1 in -t) shift ;; -o|-L) shift 2 ;; *) break ;; esac
done
shift
exec bash -c "$1"
''',
    'scp': '''#!/bin/sh
while [ $# -gt 0 ]; do case $1 in -*) shift ;; *) break ;; esac; done
exec cp -R "${1#*:}" "${2#*:}"
''',
    'pgrep': '#!/bin/sh\nexit "${FAKE_PGREP:-1}"\n',
    # "-m venv DIR" makes DIR/bin/python, which appends its arguments to $FAKE_LOG/venv_python.args.
    'python3.11': '''#!/bin/sh
[ "$1" = -m ] && [ "$2" = venv ] || exit 2
mkdir -p "$3/bin"
cat > "$3/bin/python" <<'EOF'
#!/bin/sh
{ printf '%s\\n' "$@"; echo --; } >> "$FAKE_LOG/venv_python.args"
echo fake-venv-python
EOF
chmod +x "$3/bin/python"
''',
}

BOARD = {
    'microduck-g3-offline.tar.gz': 'package',
    'microduck-g3-offline.tar.gz.sha256': 'abc  microduck-g3-offline.tar.gz\n',
    'onnxruntime-linux-aarch64-1.28.0.tgz': 'ort package',
    'onnxruntime-linux-aarch64-1.28.0.tgz.sha256': 'def  onnxruntime-linux-aarch64-1.28.0.tgz\n',
    'rust-1.89.0-aarch64-linux-gnu/archives/rustc.tar.xz': 'rustc installer',
    'lite2-2.3.2-cp311/wheelhouse/rknn_toolkit_lite2.whl': 'lite2 wheel',
    'lite2-2.3.2-cp311/requirements-lock.txt': 'lite2 lock',
    'venv-deb12u8/python3.11-venv.deb': 'deb',
    'board-baseline-20261006-qG5aeM.txt': 'baseline',
    'yolov5-rk3588-smoke/README.md': 'smoke',
    'microduck-board/robotd.toml': 'toml',
    'microduck-board/SHA256SUMS': 'sums',
    'microduck-board/sources.json': '{}',
    'microduck-board/policies/walk_17500.onnx': 'walk',
    'microduck-board/run/robotd-20261007-101111.log': 'robotd log',
    'microduck-board/run/G4/camera_cpu/summary.json': '{"frames": 450}',
    'microduck-board/bundles/20261012T080131Z_board_yolo_npu/SHA256SUMS': 'bundle sums',
    'microduck-board/vision-cpu-python/numpy/__init__.py': '',
    'microduck-board/yolo11n-cpu/wheelhouse/onnxruntime.whl': 'ort wheel',
    'microduck-board/yolo11n-cpu/requirements-lock.txt': 'ort lock',
    'microduck-board/yolo11n-camera/__pycache__/probe.cpython-311.pyc': 'cache',
    'microduck-g3-offline/README.md': 'package readme',
    'microduck-g3-offline/preparation.json': '{}',
    'microduck-g3-offline/tofd/identity.json': '{}',
    'microduck-g3-offline/microduck/robotd/src/main.rs': 'fn main() {}',
    'microduck-g3-offline/microduck/target/release/robotd': '#!/bin/sh\necho robotd\n',
    'onnxruntime-linux-aarch64-1.28.0/lib/libonnxruntime.so.1.28.0': 'ort library',
    'rk3588-npu-venv/bin/python': 'venv python',
    'rust-1.89.0/bin/cargo': 'cargo',
    'atk_app/gyro_test': 'vendor tool',
    '.bashrc': 'shell rc',
}
PYCACHE = 'microduck-board/yolo11n-camera/__pycache__'


def put(path: Path, text: str, mode: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode is not None:
        path.chmod(mode)
    return path


def snapshot(root: Path) -> dict[str, tuple]:
    """Every entry under root: kind plus content hash, executable bit or link target."""
    out = {}
    for path in sorted(root.rglob('*')):
        rel = str(path.relative_to(root))
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            out[rel] = ('link', os.readlink(path))
        elif stat.S_ISDIR(mode):
            out[rel] = ('dir',)
        elif stat.S_ISSOCK(mode):
            out[rel] = ('socket',)
        else:
            out[rel] = ('file', hashlib.sha256(path.read_bytes()).hexdigest(), bool(mode & 0o100))
    return out


@unittest.skipUnless(ZSH and board_like_shell(), 'needs zsh, bash >= 4 and GNU coreutils, as on the board')
class BoardReorgTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name).resolve()
        self.root = base / 'RK3588 Microduck copy'
        self.root.mkdir()
        for name in ('env.sh', 'scripts', 'configs', 'docs/reorg'):
            src = REPO / name
            if src.is_dir():
                shutil.copytree(src, self.root / name, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copy(src, self.root / name)
        (self.root / 'evidence').mkdir()

        self.board = base / 'board'
        for rel, text in BOARD.items():
            put(self.board / rel, text)
        (self.board / 'microduck-g3-offline/microduck/target/release/robotd').chmod(0o755)
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.bind(str(self.board / 'microduck-board/run/robotd.sock'))

        self.stubs = base / 'stubs'
        for name, text in FAKES.items():
            put(self.stubs / name, text, 0o755)
        self.log = base / 'log'
        self.log.mkdir()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('MD_') and k not in ('PYTHONPATH', 'ZDOTDIR')}
        env.update(PATH=f'{self.stubs}:{os.environ["PATH"]}', HOME=str(base / 'mac_home'),
                   FAKE_LOG=str(self.log), MD_REORG_BOARD_HOME=str(self.board),
                   MD_REORG_REMOTE_TMP=str(base / 'board_tmp'), MD_BOARD_ROBOTD=OLD_ROBOTD)
        self.env = env

    def tearDown(self):
        self.sock.close()
        self.tmp.cleanup()

    def reorg(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([ZSH, '-f', str(self.root / 'docs/reorg/board_reorg.zsh'), *args],
                              cwd=self.tmp.name, env={**self.env, **env}, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)

    def ok(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        result = self.reorg(*args, **env)
        self.assertEqual(result.returncode, 0, f'{args}\n{result.stdout}\n{result.stderr}')
        return result

    def refused(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        before = snapshot(self.board)
        result = self.reorg(*args, **env)
        self.assertNotEqual(result.returncode, 0, f'{args}\n{result.stdout}')
        self.assertEqual(snapshot(self.board), before, f'{args} changed the board')
        return result

    def test_backup_copies_board_evidence_and_verifies_hashes(self):
        before = snapshot(self.board)
        self.ok('backup', 'board_home_test')
        self.assertEqual(snapshot(self.board), before)
        dest = self.root / 'evidence/board_home_test'
        for rel in ('microduck-board/run/robotd-20261007-101111.log',
                    'microduck-board/run/G4/camera_cpu/summary.json',
                    'microduck-board/robotd.toml', 'microduck-board/SHA256SUMS',
                    'microduck-board/sources.json', 'board-baseline-20261006-qG5aeM.txt',
                    'yolov5-rk3588-smoke/README.md', 'microduck-g3-offline/README.md',
                    'microduck-g3-offline/preparation.json', 'microduck-g3-offline/tofd/identity.json'):
            self.assertEqual((dest / rel).read_text(), BOARD[rel], rel)
        for rel in ('microduck-board/run/robotd.sock', 'microduck-board/vision-cpu-python',
                    'microduck-g3-offline/microduck', 'rk3588-npu-venv'):
            self.assertFalse((dest / rel).exists(), rel)
        self.assertTrue((dest / 'backup_verified.txt').is_file())
        self.assertIn('microduck-board/run/G4/camera_cpu/summary.json',
                      (dest / 'board_sha256.txt').read_text())
        self.refused('backup', 'board_home_test')

    def test_apply_b2_needs_a_verified_backup(self):
        result = self.refused('apply', 'B2')
        self.assertIn('backup', result.stderr)

    def test_plan_lists_every_move_and_changes_nothing(self):
        before = snapshot(self.board)
        result = self.ok('plan', 'B2')
        self.assertEqual(snapshot(self.board), before)
        self.assertIn(f'{self.board}/microduck-g3-offline.tar.gz', result.stdout)
        self.assertIn(f'{self.board}/microduck/downloads/microduck-g3-offline.tar.gz', result.stdout)
        self.assertIn(f'{self.board}/microduck/archive/2026-10_ball_follow/yolov5-rk3588-smoke',
                      result.stdout)

    def test_full_cycle_then_undo_restores_the_board(self):
        original = snapshot(self.board)
        new = self.board / 'microduck'
        self.ok('backup', 'board_home_test')

        self.ok('apply', 'B2')
        self.assertEqual((new / 'downloads/lite2-2.3.2-cp311/requirements-lock.txt').read_text(),
                         'lite2 lock')
        self.assertTrue((new / 'downloads/microduck-g3-offline.tar.gz.sha256').is_file())
        self.assertTrue((new / 'archive/2026-10_ball_follow/board-baseline-20261006-qG5aeM.txt').is_file())
        self.assertEqual((new / 'downloads/vision_cpu_wheels/wheelhouse/onnxruntime.whl').read_text(),
                         'ort wheel')
        for rel in ('microduck-board/yolo11n-cpu/wheelhouse/onnxruntime.whl',
                    'microduck-board/robotd.toml', 'rk3588-npu-venv/bin/python',
                    'onnxruntime-linux-aarch64-1.28.0/lib/libonnxruntime.so.1.28.0',
                    'microduck-g3-offline/microduck/target/release/robotd'):
            self.assertTrue((self.board / rel).is_file(), f'B2 must leave {rel} in place')
        for rel in ('lite2-2.3.2-cp311', 'yolov5-rk3588-smoke', 'microduck-g3-offline.tar.gz'):
            self.assertFalse((self.board / rel).exists(), rel)
        self.refused('apply', 'B2')

        self.ok('apply', 'B4')
        robotd = new / 'runtime/bin/robotd'
        self.assertEqual(robotd.read_text(), BOARD['microduck-g3-offline/microduck/target/release/robotd'])
        self.assertTrue(os.access(robotd, os.X_OK))
        digest = hashlib.sha256(robotd.read_bytes()).hexdigest()
        self.assertEqual((new / 'runtime/bin/robotd.sha256').read_text(), f'{digest}  robotd\n')
        self.assertTrue((new / 'runtime/onnxruntime-linux-aarch64-1.28.0/lib/libonnxruntime.so.1.28.0').is_file())
        lock = new / 'runtime/venv_requirements_lock.txt'
        self.assertEqual(lock.read_text(), (REPO / 'configs/board/venv_requirements_lock.txt').read_text())
        calls = (self.log / 'venv_python.args').read_text().split('--\n')
        self.assertEqual(calls[0].split('\n')[:-1],
                         ['-m', 'pip', 'install', '--no-index', '--require-hashes',
                          '--find-links', f'{new}/downloads/lite2-2.3.2-cp311/wheelhouse',
                          '--find-links', f'{new}/downloads/vision_cpu_wheels/wheelhouse',
                          '-r', str(lock)])
        self.assertEqual(calls[1].split('\n')[:-1], ['-m', 'pip', 'check'])

        result = self.refused('apply', 'B6')
        self.assertIn('B5', result.stderr)
        self.ok('apply', 'B6', MD_BOARD_ROBOTD=NEW_ROBOTD)
        self.assertEqual(sorted(p.name for p in self.board.iterdir()),
                         ['.bashrc', 'atk_app', 'microduck'])
        archive = new / 'archive/2026-10_ball_follow'
        self.assertTrue((archive / 'microduck-board/run/G4/camera_cpu/summary.json').is_file())
        self.assertFalse((archive / 'microduck-board/yolo11n-camera/__pycache__').exists())
        self.assertTrue((archive / 'rk3588-npu-venv/bin/python').is_file())
        self.assertTrue((new / 'build/microduck-g3-offline/microduck/robotd/src/main.rs').is_file())
        self.assertTrue((new / 'toolchains/rust-1.89.0/bin/cargo').is_file())

        self.ok('apply', 'B7')
        self.assertEqual((new / 'README.md').read_text(),
                         (REPO / 'configs/board/README_board.md').read_text())
        status = self.ok('status').stdout
        for step in ('B2', 'B4', 'B6', 'B7'):
            self.assertIn(f'{step} 已执行', status)

        self.refused('undo', 'B2')
        for step in ('B7', 'B6', 'B4', 'B2'):
            self.ok('undo', step)
        restored = snapshot(self.board)
        expected = {k: v for k, v in original.items() if not k.startswith(PYCACHE)}
        self.assertEqual({k: v for k, v in restored.items() if not k.startswith('microduck/')
                          and k != 'microduck'}, expected)
        self.assertEqual(sorted(p.name for p in new.iterdir()), ['move_log.undone.tsv'])

    def test_apply_refuses_while_busy_or_when_target_exists(self):
        self.ok('backup', 'board_home_test')
        result = self.refused('apply', 'B2', FAKE_PGREP='0')
        self.assertIn('robotd', result.stdout + result.stderr)
        (self.board / 'microduck').mkdir()
        self.refused('apply', 'B2')
        (self.board / 'microduck').rmdir()
        self.refused('apply', 'B4')
        self.ok('apply', 'B2')
        self.refused('undo', 'B4')

    def test_one_existing_target_stops_the_whole_step(self):
        self.ok('backup', 'board_home_test')
        self.ok('apply', 'B2')
        put(self.board / 'microduck/runtime/bin/robotd', 'something already here')
        result = self.refused('apply', 'B4')
        self.assertIn('目标已存在', result.stdout)
        self.assertFalse((self.board / 'microduck/runtime/onnxruntime-linux-aarch64-1.28.0').exists())

    def test_backup_with_mismatching_hashes_is_not_marked_verified(self):
        # Hashes computed inside evidence/ (the Mac side of the comparison) come out wrong.
        real = shutil.which('shasum')
        put(self.stubs / 'shasum', f'#!/bin/sh\ncase $PWD in */evidence/*) echo "0000  ./x" ;; '
                                   f'*) exec {real} "$@" ;; esac\n', 0o755)
        result = self.reorg('backup', 'board_home_test')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('不一致', result.stdout + result.stderr)
        self.assertFalse((self.root / 'evidence/board_home_test/backup_verified.txt').exists())
        self.refused('apply', 'B2')

    def test_failed_undo_keeps_its_record_so_it_can_be_retried(self):
        put(self.stubs / 'rm', '#!/bin/sh\ncase "$*" in *runtime/onnxruntime*) '
                                '[ -n "$FAKE_RM_FAIL" ] && exit 1 ;; esac\n'
                                f'exec {shutil.which("rm")} "$@"\n', 0o755)
        self.ok('backup', 'board_home_test')
        self.ok('apply', 'B2')
        self.ok('apply', 'B4')
        result = self.reorg('undo', 'B4', FAKE_RM_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('B4 已执行', self.ok('status').stdout)
        self.ok('undo', 'B4')
        self.assertFalse((self.board / 'microduck/runtime/onnxruntime-linux-aarch64-1.28.0').exists())
        self.assertIn('B4 未执行', self.ok('status').stdout)

    def test_undo_never_deletes_outside_the_new_folder(self):
        self.ok('backup', 'board_home_test')
        self.ok('apply', 'B2')
        with open(self.board / 'microduck/move_log.tsv', 'a') as log:
            log.write(f'B2\tcp\t-\t{self.board}/atk_app\t2026-10-12T00:00:00Z\n')
        result = self.refused('undo', 'B2')
        self.assertIn('atk_app', result.stdout + result.stderr)

    def test_refuses_when_pgrep_fails_or_another_step_holds_the_lock(self):
        self.ok('backup', 'board_home_test')
        result = self.refused('apply', 'B2', FAKE_PGREP='3')
        self.assertIn('pgrep', result.stdout + result.stderr)
        with open('/tmp/microduck_reorg.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.refused('apply', 'B2')
        self.assertIn('另一个', result.stdout + result.stderr)
        self.ok('apply', 'B2')

    def test_backup_names_start_with_board_home(self):
        result = self.refused('backup', 'before_reorg')
        self.assertIn('board_home_', result.stderr)

    def test_precheck_and_inventory_save_read_only_reports(self):
        before = snapshot(self.board)
        self.ok('precheck')
        self.ok('inventory')
        self.assertEqual(snapshot(self.board), before)
        runs = sorted((self.root / 'runs').iterdir())
        precheck = [p for p in runs if p.name.endswith('_board_reorg_precheck')][0]
        text = (precheck / 'precheck.txt').read_text()
        for section in ('== 进程', '== 启动文件', '== robotd', '== 磁盘'):
            self.assertIn(section, text)
        inventory = [p for p in runs if p.name.endswith('_board_reorg_inventory')][0]
        self.assertIn('microduck-board', (inventory / 'inventory.txt').read_text())

    def test_usage_errors_reach_no_board(self):
        for args in ((), ('bogus',), ('apply',), ('apply', 'B3'), ('plan', 'B2', 'B4'),
                     ('status', 'x')):
            result = self.refused(*args)
            self.assertIn('用法', result.stderr, args)
        self.assertFalse((Path(self.env['MD_REORG_REMOTE_TMP'])).exists())


if __name__ == '__main__':
    unittest.main()
