"""What every Stage 1 test shares: arguments, the run folder, startup, layouts, reset and polling.

A run folder runs/<UTC>_s1<test>_<config>/ holds
    truth.csv (+ truth.meta.json)  arena body truth, one row per 20 ms (stage1/metrics.py)
    state.jsonl                    robot.state at 50 Hz (skeleton dropped)
    events.jsonl                   every move/head/mouth/rpc/probe op, time.monotonic()
    trials.jsonl                   one line per trial, written as the trial ends
    summary.json                   scores, pass/fail against the plan, the trials again
Needs the arena body (scripts/sim/stage1/run_arena_body.zsh) and robotd
(scripts/sim/stage0/run_robotd.zsh s2, or the board launcher).
"""
from __future__ import annotations

import argparse
from collections import deque
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import time

from microduck_toy.control.robot import Events, Robot, UPRIGHT
from microduck_toy.paths import repo_root, runs_dir
from microduck_toy.stage0.link import Probe
from microduck_toy.stage1 import metrics


def stage1_config() -> dict:
    return json.loads((repo_root() / 'configs' / 'control' / 'stage1.json').read_text())


def stage0_facts() -> dict:
    return json.loads((repo_root() / 'configs' / 'control' / 'stage0_facts.json').read_text())


def default_socket() -> str:
    state = os.environ.get('MD_MAC_STATE', '/private/tmp/md-runtime-ethan')
    return os.environ.get('MD_ROBOT_SOCKET', f'{state}/robotd.sock')


def parser(test: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f'stage1 {test}', description=description)
    p.add_argument('--config', default='s2',
                   help='robotd config robotd was started with (Stage 1 uses s2); names the run')
    p.add_argument('--target', choices=('mac', 'board'), default='mac',
                   help='where robotd runs; recorded in the summary')
    p.add_argument('--socket', default=None,
                   help='robotd control socket (default $MD_ROBOT_SOCKET or $MD_MAC_STATE/robotd.sock;'
                        ' with --target board, $MD_STARTUP_SOCKET)')
    p.add_argument('--probe', default='127.0.0.1:7811', help='arena body test port')
    p.add_argument('--quick', action='store_true',
                   help='a few repetitions of each part, to check the script end to end')
    p.add_argument('--parts', default=None,
                   help='comma-separated subset of parts to run (default: all)')
    p.add_argument('--rescore', metavar='RUN_DIR', default=None,
                   help='score a finished run folder again (rewrites its summary.json); no robot')
    return p


def git_head() -> str | None:
    try:
        return subprocess.run(['git', '-C', str(repo_root()), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


class Summary:
    dir: Path
    meta: dict
    trials: list

    def write_summary(self, summary: dict) -> None:
        self.meta['finished_utc'] = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        path = self.dir / 'summary.json'
        path.write_text(json.dumps(metrics.clean({'meta': self.meta, **summary,
                                                  'trials': self.trials}),
                                   indent=2, ensure_ascii=False))
        print(f'结果：{path}', flush=True)


class Rescored(Summary):
    """A finished run folder, loaded again so a changed scorer can re-read it."""

    def __init__(self, directory: str):
        self.dir = Path(directory)
        saved = json.loads((self.dir / 'summary.json').read_text())
        self.meta = saved['meta']
        self.meta['rescored'] = True
        self.refs = self.meta['refs']
        self.cfg = self.meta['stage1_config']
        self.args = argparse.Namespace(quick=self.meta['quick'])
        path = self.dir / 'trials.jsonl'
        lines = path.read_text().splitlines() if path.exists() else []
        self.trials = [json.loads(line) for line in lines if line.strip()] or saved.get('trials', [])
        if not self.trials:
            raise RuntimeError(f'{self.dir}: no trials in trials.jsonl or summary.json')


class Session(Summary):
    def __init__(self, test: str, args: argparse.Namespace):
        self.args = args
        if args.socket is None:
            args.socket = (os.environ.get('MD_STARTUP_SOCKET', '/private/tmp/md-rk-startup-ethan.sock')
                           if args.target == 'board' else default_socket())
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.dir = runs_dir() / f'{stamp}_s1{test}_{args.config}'
        self.dir.mkdir(parents=True, exist_ok=False)
        print(f'运行目录：{self.dir}', flush=True)
        self.cfg = stage1_config()
        self.facts = stage0_facts()
        self.events = Events(self.dir / 'events.jsonl')
        host, port = args.probe.rsplit(':', 1)
        self.probe = Probe(host, int(port), self.events)
        info = self.probe.call('info')
        if 'floor' not in info:
            raise RuntimeError('the test port is not the Stage 1 arena body '
                               '(start scripts/sim/stage1/run_arena_body.zsh)')
        self.refs = info['refs']
        self.probe.call('record', path=str(self.dir / 'truth.csv'))
        self.robot = Robot(args.socket, self.dir, self.events)
        self.trials: list[dict] = []
        self.meta = {'test': test, 'config': args.config, 'target': args.target,
                     'socket': args.socket, 'quick': args.quick, 'parts': args.parts,
                     'repo_commit': git_head(), 'refs': self.refs, 'floor': info['floor'],
                     'camera': info.get('camera'), 'toys': info.get('toys'),
                     'tray_centre': info.get('tray_centre'), 'stage1_config': self.cfg,
                     'started_utc': stamp}

    # ── startup ──────────────────────────────────────────────────────────

    def start(self) -> None:
        robot = self.robot
        state = robot.wait(lambda s: True, 3.0, 'first robot.state')
        self.meta['models'] = robot.models
        self.meta['policies'] = robot.request('robot.policies', {})
        if state['policy'] == 'held':
            reply = robot.request('robot.enable', {'on': True})
            if not reply.get('accepted', True):
                raise RuntimeError(f'robot.enable refused: {reply}')
            print('已请求 robot.enable；等待策略接管……', flush=True)
        robot.wait(lambda s: s['policy'] in UPRIGHT, 10.0, 'policy takeover (stand/walk)')
        deadline = time.monotonic() + 6.0
        while self.probe.truth()['released'] != 1:
            if time.monotonic() > deadline:
                raise TimeoutError('arena body still holds the robot (startup support not ended)')
            time.sleep(0.05)
        print('策略已接管，起步支撑已解除。', flush=True)

    # ── per-trial helpers ────────────────────────────────────────────────

    def mark(self, label: str) -> float:
        self.probe.mark(label)
        now = time.monotonic()
        self.events.log('mark', label=label)
        return now

    def reset(self, head: str = 'H0', robot=(0.0, 0.0, 0.0), toys: dict | None = None,
              hold: float = 1.0, settle: float = 1.0) -> dict:
        """Zero the command, close the mouth, wait out a fall, stand the robot at `robot`
        (x, y, yaw) with `toys` ({short: [x, y, yaw]}) placed and every other toy parked."""
        self.robot.move(0.0, 0.0, 0.0)
        self.robot.mouth(0.0)
        self.robot.head(head)
        try:
            self.robot.wait(lambda s: s['policy'] in UPRIGHT, 4.0, 'upright policy')
        except TimeoutError:
            self.events.log('warning', text='reset while not upright')
        self.probe.call('layout', robot=list(robot), toys=toys or {}, hold_s=hold)
        time.sleep(hold + settle)
        return self.probe.truth()

    def place_in_trunk(self, toy: str, forward: float, left: float, yaw_rel: float = 0.0,
                       row: dict | None = None) -> dict:
        """Put a toy at (forward, left) in the robot's current heading frame (truth used to
        place it: a test fixture, not something the controller sees)."""
        row = row or self.probe.truth()
        c, s = math.cos(row['yaw']), math.sin(row['yaw'])
        x = row['x'] + c * forward - s * left
        y = row['y'] + s * forward + c * left
        return self.probe.call('toy', name=toy, x=x, y=y, yaw=row['yaw'] + yaw_rel)

    def poll(self, duration: float, until=None, every: float = 0.02) -> list[dict]:
        rows, end = [], time.monotonic() + duration
        while time.monotonic() < end:
            row = self.probe.truth()
            if not rows or row['mono'] != rows[-1]['mono']:
                rows.append(row)
                if until is not None and until(row, self.robot.snapshot()):
                    break
            time.sleep(every)
        return rows

    def wait_settled(self, max_s: float = 3.0, window_s: float = 0.5) -> float | None:
        start, rows = time.monotonic(), deque()
        while time.monotonic() - start < max_s:
            row = self.probe.truth()
            if not rows or row['mono'] != rows[-1]['mono']:
                rows.append(row)
                while rows and rows[0]['mono'] < row['mono'] - window_s - 0.1:
                    rows.popleft()
                if row['mono'] - start >= window_s:
                    settled = metrics.settle_time(list(rows), row['mono'] - window_s - 0.02,
                                                  window_s)
                    if settled is not None:
                        return settled
            time.sleep(0.02)
        return None

    def grasps_since(self, index: int) -> list[dict]:
        return self.probe.call('grasps', since=index)['grasps']

    def grasp_count(self) -> int:
        return int(self.probe.call('grasps', since=0)['count'])

    def trial(self, record: dict) -> None:
        # Opened per write (see stage0 common.py: an open handle left trials.jsonl empty on the Mac).
        self.trials.append(record)
        with (self.dir / 'trials.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')

    # ── end ──────────────────────────────────────────────────────────────

    def close(self) -> list[dict]:
        try:
            self.robot.move(0.0, 0.0, 0.0)
            self.robot.mouth(0.0)
            self.robot.head('H0')
            time.sleep(0.3)
        finally:
            self.meta['falls_seen_by_guard'] = self.robot.falls
            self.robot.close()
            self.probe.close()
            self.events.close()
        return metrics.read_truth(self.dir / 'truth.csv')


def run(test: str, args, body) -> tuple[int, Session, list[dict]]:
    session = Session(test, args)
    status = 0
    try:
        session.start()
        body(session)
    except KeyboardInterrupt:
        session.meta['interrupted'] = True
        status = 130
        print('已中断；保存已完成的部分。', flush=True)
    except Exception as error:
        session.meta['error'] = f'{type(error).__name__}: {error}'
        status = 1
        print(f'出错：{session.meta["error"]}', flush=True)
    rows = session.close()
    return status, session, rows


def main(test: str, doc: str, parts: dict, score, argv=None, check=None, extra=None) -> int:
    """Command line shared by the Stage 1 tests: run the chosen parts, or --rescore a folder."""
    p = parser(f'1{test}', doc.splitlines()[0])
    if extra is not None:
        extra(p)
    args = p.parse_args(argv)
    if args.rescore:
        done = Rescored(args.rescore)
        rows = metrics.read_truth(done.dir / 'truth.csv')
        done.write_summary({'truth': metrics.continuity(rows), **score(done, rows)})
        return 0
    chosen = args.parts.split(',') if args.parts else list(parts)
    unknown = set(chosen) - set(parts)
    if unknown:
        p.error(f'unknown parts {sorted(unknown)}; choose from {list(parts)}')

    def body(session):
        if check is not None:
            check(session)
        for name in chosen:
            print(f'== {name}', flush=True)
            parts[name](session)

    status, session, rows = run(test, args, body)
    session.write_summary({'truth': metrics.continuity(rows), **score(session, rows)})
    return status
