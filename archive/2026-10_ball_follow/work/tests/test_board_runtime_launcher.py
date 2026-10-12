"""A real PTY and temporary processes verify the combined A/B launcher lifecycle."""
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest


LAUNCHER = Path(__file__).resolve().parents[1]/'run_board_runtime_ball.zsh'

LEAF = '''import json, os, pathlib, sys, time
root, name = pathlib.Path(os.environ['FIXTURE_ROOT']), sys.argv[1]
(root/(name+'.pid')).write_text(str(os.getpid()))
(root/(name+'.json')).write_text(json.dumps({'pid':os.getpid(), 'pgid':os.getpgrp()}))
while True:
    time.sleep(.1)
'''

WORKER = '''import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
(root/'worker.pid').write_text(str(os.getpid()))
(root/'worker.json').write_text(json.dumps({'pid':os.getpid(), 'pgid':os.getpgrp()}))
subprocess.Popen([sys.executable, str(root/'leaf.py'), 'leaf'])
while True:
    time.sleep(.1)
'''

BODY = '''import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
mode = os.environ['FIXTURE_MODE']
(root/'body.json').write_text(json.dumps({'pid':os.getpid(), 'pgid':os.getpgrp(),
                                        'args':sys.argv[1:]}))
subprocess.Popen([sys.executable, str(root/'worker.py')])
while not (root/'leaf.json').exists():
    time.sleep(.01)
if mode == 'early':
    print('A_EARLY_EXIT', flush=True)
    sys.exit(17)
if mode == 'wait':
    print('A_WAITING_FOR_PORTS', flush=True)
else:
    (root/'body_ready').touch()
    print('A_READY', flush=True)
while True:
    time.sleep(.1)
'''

BOARD = '''import json, os, pathlib, signal, sys, time
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
mode = os.environ['FIXTURE_MODE']
signal.signal(signal.SIGINT, signal.default_int_handler)
deadline = time.monotonic()+.5
while os.tcgetpgrp(0)!=os.getpgrp() and time.monotonic()<deadline:
    time.sleep(.005)
meta = {'pid':os.getpid(), 'pgid':os.getpgrp(), 'stdin_tty':os.isatty(0),
        'stdout_tty':os.isatty(1), 'foreground':os.tcgetpgrp(0)==os.getpgrp()}
(root/'board.json').write_text(json.dumps(meta))
print('B_FOREGROUND '+json.dumps(meta), flush=True)
try:
    if mode == 'password':
        print('fixture password:', flush=True)
        input()
    elif mode == 'running':
        print('B_RUNNING', flush=True)
        while True:
            time.sleep(.1)
except KeyboardInterrupt:
    print('B_INTERRUPTED', flush=True)
    sys.exit(130)
sys.exit(23 if mode == 'nonzero' else 0)
'''

LSOF = '''#!/bin/zsh
if [[ "$FIXTURE_MODE" == foreign ]]; then
    cat "$FIXTURE_ROOT/foreign.pid"
    exit 0
fi
[[ -f "$FIXTURE_ROOT/body_ready" ]] || exit 1
if [[ "$*" == *7801* ]]; then
    cat "$FIXTURE_ROOT/leaf.pid"
else
    cat "$FIXTURE_ROOT/worker.pid"
fi
'''


def process_active(pid):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat='],
                            capture_output=True, text=True, timeout=1)
    state = result.stdout.strip()
    return bool(state) and not state.startswith('Z')


class PTYTrial:
    def __init__(self, environment, launcher=LAUNCHER):
        sys.stdout.flush()
        sys.stderr.flush()
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execve('/bin/zsh', ['zsh', str(launcher)], environment)
        self.output = b''
        self.returncode = None

    def read(self):
        if select.select([self.fd], [], [], .05)[0]:
            try:
                self.output += os.read(self.fd, 65536)
            except OSError:
                pass  # The PTY reports EIO when the last terminal client exits.
        if self.returncode is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.returncode = os.waitstatus_to_exitcode(status)

    def await_output(self, marker, timeout=5):
        deadline = time.monotonic()+timeout
        while marker not in self.output and time.monotonic()<deadline:
            self.read()
            if self.returncode is not None:
                break
        if marker not in self.output:
            raise AssertionError(f'PTY did not reach {marker!r}:\n{self.output.decode(errors="replace")}')

    def ctrl_c(self):
        os.write(self.fd, b'\x03')

    def finish(self, timeout=8):
        deadline = time.monotonic()+timeout
        while self.returncode is None and time.monotonic()<deadline:
            self.read()
        if self.returncode is None:
            raise AssertionError(f'PTY launcher did not exit:\n{self.output.decode(errors="replace")}')
        self.read()
        return self.returncode

    def close(self):
        if self.returncode is None:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(self.pid, 0)
        os.close(self.fd)


class BoardRuntimeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(LAUNCHER.is_file(), 'the combined A/B launcher has not been implemented')
        self.temp = tempfile.TemporaryDirectory(prefix='ab-launcher-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'work').mkdir()
        (self.root/'bin').mkdir()
        for name, code in [('leaf.py', LEAF), ('worker.py', WORKER), ('body.py', BODY), ('board.py', BOARD)]:
            (self.root/name).write_text(code)
        for name, script in [('run_board_runtime_body_camera_ball_no_reflection.zsh', 'body.py'),
                             ('run_board_runtime_robotd.zsh', 'board.py')]:
            (self.root/'work'/name).write_text(
                '#!/bin/zsh\nexec "$FIXTURE_PYTHON" "$FIXTURE_ROOT/'+script+'" "$@"\n')
        (self.root/'bin/lsof').write_text(LSOF)
        (self.root/'bin/lsof').chmod(0o755)
        self.addCleanup(self.cleanup_fixture_processes)

    def metadata(self, name):
        return json.loads((self.root/(name+'.json')).read_text())

    def environment(self, mode):
        return {**os.environ, 'MD_RUNTIME_ROOT':str(self.root), 'FIXTURE_ROOT':str(self.root),
                'FIXTURE_PYTHON':sys.executable, 'FIXTURE_MODE':mode,
                'PATH':str(self.root/'bin')+':'+os.environ['PATH']}

    def start(self, mode, launcher=LAUNCHER):
        trial = PTYTrial(self.environment(mode), launcher)
        self.addCleanup(trial.close)
        return trial

    def cleanup_fixture_processes(self):
        # A failing launcher assertion cannot leave the fixture's own children alive.
        members = [json.loads(file.read_text()) for file in self.root.glob('*.json')]
        for member in members:
            group = member['pgid']
            if group != os.getpgrp() and process_active(member['pid']):
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # Darwin can reject a group containing only already-exited members.
                    for child in members:
                        if child['pgid'] == group and process_active(child['pid']):
                            try:
                                os.kill(child['pid'], signal.SIGKILL)
                            except ProcessLookupError:
                                pass

    def assert_body_tree_exited(self):
        members = [self.metadata(name) for name in ('body', 'worker', 'leaf')]
        self.assertEqual(len({member['pgid'] for member in members}), 1)
        self.assertEqual(members[0]['pgid'], members[0]['pid'], 'A must have its own job process group')
        deadline = time.monotonic()+2
        while time.monotonic()<deadline and any(process_active(member['pid']) for member in members):
            time.sleep(.02)
        self.assertTrue(all(not process_active(member['pid']) for member in members),
                        f'A descendants remain active: {members}')

    def assert_board_was_foreground(self):
        board = self.metadata('board')
        self.assertTrue(board['stdin_tty'])
        self.assertTrue(board['stdout_tty'])
        self.assertTrue(board['foreground'], 'B must own the foreground terminal during SSH interaction')
        self.assertEqual(board['pgid'], board['pid'])
        self.assertNotEqual(board['pgid'], self.metadata('body')['pgid'])
        self.assertFalse(process_active(board['pid']))

    def test_normal_board_exit_preserves_zero_and_reaps_body_descendants(self):
        trial = self.start('normal')
        self.assertEqual(trial.finish(), 0, trial.output.decode(errors='replace'))
        self.assertIn(b'A_READY', trial.output)
        self.assertIn(b'B_FOREGROUND', trial.output)
        self.assert_board_was_foreground()
        self.assert_body_tree_exited()

    def test_nonzero_board_exit_is_preserved_and_body_descendants_are_reaped(self):
        trial = self.start('nonzero')
        self.assertEqual(trial.finish(), 23, trial.output.decode(errors='replace'))
        self.assert_board_was_foreground()
        self.assert_body_tree_exited()

    def test_board_exit_before_fg_cannot_select_body_as_the_foreground_job(self):
        # Force the scheduling race: B exits while A remains alive before fg is evaluated.
        delayed = self.root/'delayed_launcher.zsh'
        delayed.write_text('\n'.join(('sleep .05\n'+line if line.strip().startswith('fg ')
                                       else line) for line in LAUNCHER.read_text().splitlines())+'\n')
        (self.root/'work/run_board_runtime_robotd.zsh').write_text(
            '#!/bin/zsh\nprint -r -- "$$" > "$FIXTURE_ROOT/fast_board.pid"\nexit 42\n')
        trial = self.start('normal', delayed)
        self.assertEqual(trial.finish(), 42, trial.output.decode(errors='replace'))
        self.assertFalse(process_active(int((self.root/'fast_board.pid').read_text())))
        self.assert_body_tree_exited()

    def test_body_early_exit_never_starts_board_and_reaps_orphan_descendants(self):
        trial = self.start('early')
        self.assertNotEqual(trial.finish(), 0)
        self.assertIn(b'A_EARLY_EXIT', trial.output)
        self.assertFalse((self.root/'board.json').exists())
        self.assert_body_tree_exited()

    def test_preexisting_foreign_listener_is_left_alive_and_neither_job_starts(self):
        foreign = subprocess.Popen([sys.executable, str(self.root/'leaf.py'), 'foreign'],
                                   env=self.environment('foreign'), start_new_session=True)

        def close_foreign():
            if foreign.poll() is None:
                foreign.terminate()
            foreign.wait(2)

        self.addCleanup(close_foreign)
        deadline = time.monotonic()+2
        while not (self.root/'foreign.pid').exists() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertTrue((self.root/'foreign.pid').exists())
        trial = self.start('foreign')
        self.assertNotEqual(trial.finish(), 0)
        self.assertTrue(process_active(foreign.pid))
        self.assertIsNone(foreign.poll())
        self.assertFalse((self.root/'body.json').exists())
        self.assertFalse((self.root/'board.json').exists())

    def test_ctrl_c_while_waiting_for_ports_reaps_only_new_body_tree(self):
        trial = self.start('wait')
        trial.await_output(b'A_WAITING_FOR_PORTS')
        trial.ctrl_c()
        self.assertEqual(trial.finish(), 130, trial.output.decode(errors='replace'))
        self.assertFalse((self.root/'board.json').exists())
        self.assert_body_tree_exited()

    def test_ctrl_c_at_password_prompt_interrupts_foreground_board_and_reaps_body(self):
        trial = self.start('password')
        trial.await_output(b'fixture password:')
        trial.ctrl_c()
        self.assertEqual(trial.finish(), 130, trial.output.decode(errors='replace'))
        self.assertIn(b'B_INTERRUPTED', trial.output)
        self.assert_board_was_foreground()
        self.assert_body_tree_exited()

    def test_ctrl_c_during_board_run_reaps_both_job_groups(self):
        trial = self.start('running')
        trial.await_output(b'B_RUNNING')
        trial.ctrl_c()
        self.assertEqual(trial.finish(), 130, trial.output.decode(errors='replace'))
        self.assertIn(b'B_INTERRUPTED', trial.output)
        self.assert_board_was_foreground()
        self.assert_body_tree_exited()

    def assert_parent_signal_cleans_both_groups(self, signum, exit_code):
        trial = self.start('running')
        trial.await_output(b'B_RUNNING')
        began = time.monotonic()
        os.kill(trial.pid, signum)
        self.assertEqual(trial.finish(timeout=4), exit_code, trial.output.decode(errors='replace'))
        self.assertLess(time.monotonic()-began, 4)
        self.assert_board_was_foreground()
        self.assert_body_tree_exited()

    def test_parent_sigterm_during_foreground_board_run_reaps_both_job_groups(self):
        self.assert_parent_signal_cleans_both_groups(signal.SIGTERM, 143)

    def test_parent_sighup_during_foreground_board_run_reaps_both_job_groups(self):
        self.assert_parent_signal_cleans_both_groups(signal.SIGHUP, 129)


if __name__ == '__main__':
    unittest.main()
