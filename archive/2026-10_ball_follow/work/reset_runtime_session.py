"""Stop only the verified local MicroDuck body and daemon for a fresh simulator trial."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time

PROJECT = Path('/Users/ethan/Documents/RK3588 Microduck')
CONTROL = '/private/tmp/md-runtime-ethan/robotd.sock'


def pids(*args: str) -> list[int]:
    result = subprocess.run(['/usr/sbin/lsof', '-t', *args], text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return sorted({int(line) for line in result.stdout.splitlines() if line.strip()})


def command(pid: int) -> str:
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ''


def matches(pid: int, role: str) -> bool:
    cmd = command(pid)
    if role == 'robotd':
        return (cmd.startswith(str(PROJECT / 'task/microduck/target/debug/robotd') + ' ')
                and '--sim 127.0.0.1:7801' in cmd and CONTROL in cmd)
    return (str(PROJECT / 'task/microduck_r1/.venv') in cmd
            and ('-m mjlab_microduck.sim.body_server' in cmd
                 or str(PROJECT / 'work/runtime_bootstrap_body.py') in cmd)
            and '--port 7801' in cmd)


def main() -> None:
    daemon_ids = pids(CONTROL) if Path(CONTROL).exists() else []
    # Subscribers may appear in lsof too; they are never stop targets.
    daemons = [pid for pid in daemon_ids if matches(pid, 'robotd')]
    bodies = pids('-nP', '-iTCP:7801', '-sTCP:LISTEN')
    if len(daemons) > 1 or len(bodies) > 1:
        raise RuntimeError('检测到多个候选进程，未停止任何进程。')
    if any(not matches(pid, 'body') for pid in bodies):
        raise RuntimeError('7801 端口并非本项目身体进程，未停止任何进程。')
    if daemon_ids and not daemons and not bodies:
        print('未发现本项目正在运行的控制程序或身体。', flush=True)
    for role, targets in [('robotd', daemons), ('body', bodies)]:
        for pid in targets:
            if not matches(pid, role):
                raise RuntimeError(f'{role} 进程身份发生变化，停止重启。')
            print(f'停止旧 {role}（PID {pid}），保留已有日志。', flush=True)
            os.kill(pid, signal.SIGINT)
            deadline = time.monotonic() + 4.0
            while command(pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{role} 尚未退出；请在对应终端按 Ctrl+C，再保留输出。')
                time.sleep(0.1)
    if pids('-nP', '-iTCP:7801', '-sTCP:LISTEN'):
        raise RuntimeError('7801 尚未释放，停止重启。')
    print('旧实验已停止，可以启动新的仿真身体。', flush=True)


if __name__ == '__main__':
    main()
