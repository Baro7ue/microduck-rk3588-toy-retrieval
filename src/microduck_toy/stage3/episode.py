"""Stage 3 runs: lay out the field, run the brain, judge each episode on the simulator's truth.

    python -m microduck_toy.stage3.episode 3ab|3c|3d|3e [--parts ...] [--episodes 3000:bus,...]
        [--quick] [--eyes yolo|oracle] [--source stream|probe] [--backend cpu]
        [--target mac|board] [--rescore runs/<dir>]
    python -m microduck_toy.stage3.episode layouts         (writes configs/control/stage3_layouts.json)

Needs the arena body with its camera (scripts/sim/stage1/run_arena_body.zsh --camera) and robotd s2.
On the Mac the brain runs in this process (camera from 127.0.0.1:7901, YOLO FP32 on the CPU, ToF from
the sim body's 7801); with --target board it runs on the RK3588, one ssh per episode
(scripts/board/stage3/run_board_brain.zsh prepares the code bundle).

Tests (plan sections 5-9):
    3ab  ab       the 30 dev episodes (seeds 3000-3009 x bus, henry, ketchup) until held: 3A search and
                  confirm, 3B pick, one run
    3c   c_drop   from the -x staging point with the toy loaded, tray look + drop at d = 0.19/0.21/
                  0.23 m x 3 targets x 3 (heading +-10 deg, sideways +-5 cm off what the brain is told)
         c_route  9 pick spots (dev layouts 3000-3002 x 3 targets): toy loaded, carry, drop
    3d   d_fall   9 full episodes, pushed over once while walking in GOTO, APPROACH or CARRY
         d_tof    6 walks at a wall (H0 vx 0.3 and H1 vx 0.25; head-on and +-30 deg)
    3e   e        the 30 dev episodes until delivered (the Stage 3 gate)
Everything the brain did is in episodes/<label>/brain.jsonl; trials.jsonl has one line per episode;
summary.json the scores against the plan. Scoring reads only truth.csv and the brain's logs.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import shlex
import socket
import subprocess
import sys
import threading
import time

from microduck_toy.control.pick import ground_pick
from microduck_toy.control.robot import Events, Robot
from microduck_toy.paths import repo_root, runs_dir
from microduck_toy.stage0.link import Probe
from microduck_toy.stage1 import common, metrics
from microduck_toy.stage3 import brain as brain_lib, measure
from microduck_toy.stage3.eyes import StateHistory
from microduck_toy.stage3.tof import TofReader
from microduck_toy.stage3.world import odom_of, wrap
from microduck_toy.vision import views

TARGETS = list(brain_lib.TARGETS)
DISTRACTORS = ['croc', 'thomas']
TOYS = TARGETS + DISTRACTORS
TESTS = {'3ab': ('ab',), '3c': ('c_drop', 'c_route'), '3d': ('d_fall', 'd_tof'), '3e': ('e',),
         '3p': ('m_view', 'm_tof', 'm_reach', 'm_spin')}
C_SEED = 3100
PUSH_STATES = ('GOTO', 'APPROACH', 'CARRY')
# A drop walk that carried the trunk this much past its plan stepped onto the rim and lurched into
# the tray (cloud 3C: +13 and +18 cm, against a +1 to +5 cm overshoot otherwise).
STUMBLE_M = 0.08
PUSH_DIRS = {'bus': ('forward', 1.05, 0.0), 'henry': ('left', 0.0, 1.05),
             'ketchup': ('backward', -1.05, 0.0)}           # stage0_facts getup push_to_fall_mps_mac_s2
PUSH_TRIES, PUSH_STEP, PUSH_FALL_S = 2, 0.2, 1.5           # still up 1.5 s later: once more, 20% harder
PUSH_WALKS = 3                                            # ... and again on the next walk in the state


# ── layouts ─────────────────────────────────────────────────────────────────

def layouts_path() -> Path:
    return repo_root() / 'configs' / 'control' / 'stage3_layouts.json'


def make_layouts() -> dict:
    from microduck_toy.scene import layout
    s1 = common.stage1_config()
    s3 = json.loads((repo_root() / 'configs' / 'control' / 'stage3.json').read_text())
    base, n = int(s3['episode']['dev_seed_base']), int(s3['episode']['dev_seeds'])
    formal = range(int(s1['layout']['formal_seed_base']),
                   int(s1['layout']['formal_seed_base']) + int(s1['layout']['formal_seeds']))
    if any(base + i in formal for i in range(n)):
        raise SystemExit('dev seeds overlap the formal seeds')
    doc = {'about': f'Stage 3 dev layouts (plan section 2): seeds {base}-{base + n - 1} of '
                    'scene/layout.py with the Stage 1 rules (configs/control/stage1.json "layout"). '
                    'Each layout runs once per target: 10 x 3 = 30 dev episodes. The formal seeds '
                    '1000-1019 are for Stage 4 only.',
           'layouts': [layout.make_layout(base + i, s1['layout'], s1['arena']) for i in range(n)]}
    layouts_path().write_text(json.dumps(doc, indent=1) + '\n')
    return doc


def load_layouts() -> list[dict]:
    return json.loads(layouts_path().read_text())['layouts']


def episode_layout(lay: dict, target: str) -> dict:
    """The layout op's arguments: robot pose, the target in the target slot, both distractors."""
    toys = {target: lay['slots']['target']}
    toys.update({k: v for k, v in lay['slots'].items() if k != 'target'})
    return {'robot': lay['robot'], 'toys': toys}


def dev_episodes(spec: str | None, quick: bool) -> list[dict]:
    lays = {l['seed']: l for l in load_layouts()}
    if spec:
        out = []
        for item in spec.split(','):
            seed, target = item.split(':')
            out.append({'seed': int(seed), 'target': target, 'layout': lays[int(seed)]})
        return out
    if quick:
        seeds = sorted(lays)[:3]
        return [{'seed': s, 'target': t, 'layout': lays[s]} for s, t in zip(seeds, TARGETS)]
    return [{'seed': s, 'target': t, 'layout': lays[s]} for s in sorted(lays) for t in TARGETS]


# ── the runner ──────────────────────────────────────────────────────────────

def rpc_once(path: str, method: str, params: dict, timeout: float = 3.0) -> dict:
    """One request on robotd's socket, no intents sent (the board brain owns those)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(path)
        client.sendall((json.dumps({'jsonrpc': '2.0', 'id': 7, 'method': method,
                                    'params': params}) + '\n').encode())
        stream = client.makefile('rb')
        while True:
            reply = json.loads(stream.readline())
            if reply.get('id') == 7:
                return reply


class Runner(common.Session):
    """A run folder, the probe, and (on the Mac) the robot, eyes and ToF the brain uses.
    Inherits the Stage 1 Session's reset/mark/grasp/trial helpers; its own constructor."""

    def __init__(self, test: str, args: argparse.Namespace):
        self.args = args
        self.board = args.target == 'board'
        if args.socket is None:
            args.socket = (os.environ.get('MD_STARTUP_SOCKET', '/private/tmp/md-rk-startup-ethan.sock')
                           if self.board else common.default_socket())
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.dir = runs_dir() / f'{stamp}_s3{test}_{args.config}{"_board" if self.board else ""}'
        self.dir.mkdir(parents=True, exist_ok=False)
        print(f'运行目录：{self.dir}', flush=True)
        self.configs = brain_lib.load_configs()
        self.cfg = self.configs['stage1']
        self.s3 = self.configs['stage3']
        self.facts = common.stage0_facts()
        self.events = Events(self.dir / 'events.jsonl')
        host, port = args.probe.rsplit(':', 1)
        self.probe = Probe(host, int(port), self.events)
        info = self.probe.call('info')
        if 'floor' not in info:
            raise RuntimeError('the test port is not the arena body (scripts/sim/stage1/run_arena_body.zsh)')
        if not info.get('camera'):
            raise RuntimeError('the arena body has no camera; start run_arena_body.zsh with --camera')
        self.refs = info['refs']
        self.probe.call('record', path=str(self.dir / 'truth.csv'))
        self.history = StateHistory()
        self.robot = None if self.board else Robot(args.socket, self.dir, self.events,
                                                   on_state=self.history)
        self.eyes = self.tof = None
        self.trials: list[dict] = []
        self.meta = {'test': test, 'config': args.config, 'target': args.target,
                     'socket': args.socket, 'quick': args.quick, 'parts': args.parts,
                     'eyes': args.eyes, 'source': args.source, 'backend': args.backend,
                     'repo_commit': common.git_head(), 'refs': self.refs, 'floor': info['floor'],
                     'toys': info.get('toys'), 'tray_centre': info.get('tray_centre'),
                     'stage1_config': self.cfg, 'stage3_config': self.s3, 'started_utc': stamp,
                     'mac_run': str(Path(args.mac_run).resolve()) if args.mac_run else None,
                     'board_cpus': (args.board_cpus or None) if self.board else None}

    def start(self) -> None:
        if self.board:
            reply = rpc_once(self.args.socket, 'robot.enable', {'on': True})
            self.events.log('rpc', method='robot.enable', reply=reply)
            deadline = time.monotonic() + 15.0
            while self.probe.truth()['released'] != 1:
                if time.monotonic() > deadline:
                    raise TimeoutError('arena body still holds the robot (startup support not ended)')
                time.sleep(0.05)
            print('板上 robotd 已接管，起步支撑已解除。', flush=True)
            return
        super().start()
        eye_probe = None
        if self.args.source == 'probe':           # its own connection: a view op waits for a frame
            host, port = self.args.probe.rsplit(':', 1)
            eye_probe = Probe(host, int(port))
        self.eyes = brain_lib.make_eyes(self.args, self.configs, self.history, eye_probe)
        host = '127.0.0.1'
        self.tof = TofReader((host, 7801), self.robot, self.s3['tof'],
                             log=lambda e, **f: self.events.log(e, **f))
        self.tof.start()
        self.meta['models'] = self.robot.models

    def reset(self, head='H0', robot=(0.0, 0.0, 0.0), toys=None, hold=1.0, settle=1.0) -> dict:
        if not self.board:
            return super().reset(head, robot, toys, hold, settle)
        self.probe.call('layout', robot=list(robot), toys=toys or {}, hold_s=hold)
        time.sleep(hold + settle)
        return self.probe.truth()

    def close(self) -> list[dict]:
        try:                        # how long the body took to draw each frame (the camera lag)
            self.meta['camera_draw'] = self.probe.call('info').get('camera_draw')
        except Exception:
            pass
        for thing in (self.tof, self.eyes):
            if thing is not None:
                try:
                    thing.close()
                except Exception:
                    pass
        if not self.board:
            return super().close()
        self.probe.close()
        self.events.close()
        return metrics.read_truth(self.dir / 'truth.csv')

    # ── one episode ─────────────────────────────────────────────────────

    def ep_dir(self, label: str) -> Path:
        d = self.dir / 'episodes' / label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def brain(self, label: str, target: str, start, **kw) -> brain_lib.Brain:
        frames = None if self.args.no_frames else self.ep_dir(label) / 'frames'
        return brain_lib.Brain(self.robot, self.eyes, self.tof, self.configs, target, start,
                               log_path=self.ep_dir(label) / 'brain.jsonl', frames_dir=frames, **kw)

    def finish(self, rec: dict, label: str) -> dict:
        """Stamp the end, wait judge_after_s, keep the truth row the episode is judged on."""
        rec['end_mark'] = self.mark(f'{label}_end')
        time.sleep(float(self.s3['episode']['judge_after_s']))
        rec['judge_row'] = self.probe.truth()
        rec['judge_mark'] = self.mark(f'{label}_judge')
        rec['grasps'] = self.grasps_since(rec['grasps_from'])
        if not self.board:
            self.robot.mouth(1.0)               # let go of anything still held
            time.sleep(0.8)
            self.robot.mouth(0.0)
        return rec

    def full(self, ep: dict, part: str, until: str, push=None) -> dict:
        lay = episode_layout(ep['layout'], ep['target'])
        label = f'{part}_{ep["seed"]}_{ep["target"]}'
        self.reset(head='H0', robot=lay['robot'], toys=lay['toys'])
        if not self.board:
            self.wait_settled(3.0)
        rec = {'part': part, 'label': label, 'seed': ep['seed'], 'target': ep['target'],
               'layout': lay, 'until': until, 'grasps_from': self.grasp_count(),
               'start': self.mark(label), 'offset': 0.0}
        if self.board:
            rec['brain'] = self.remote(label, ep['target'], lay['robot'], until, rec)
            return self.finish(rec, label)
        pusher = None
        b = self.brain(label, ep['target'], lay['robot'], until=until)
        if push is not None:
            pusher = Pusher(self, b, *push)
            b.on_state = pusher.on_state
            rec['push'] = {'state': push[0], 'dir': push[1]}
        n0 = len(self.eyes.timings)
        rec['brain'] = b.run()
        rec['brain']['latency'] = self.eyes.latency(n0)
        if pusher is not None:
            pusher.stop()
            rec['push'].update(pusher.done)
        return self.finish(rec, label)

    def remote(self, label: str, target: str, start, until: str, rec: dict) -> dict:
        """The brain on the board, one ssh; its sync lines give the board-to-Mac clock offset."""
        a = self.args
        remote_out = f'{a.board_out}/episodes/{label}'
        cmd = board_command(a, remote_out, target, start, until)
        offsets = []
        proc = subprocess.Popen(['ssh', a.board_host, cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        with (self.ep_dir(label) / 'console.log').open('w') as console:
            for line in proc.stdout:
                now = time.monotonic()
                console.write(line)
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict) and 'sync' in msg:
                    offsets.append(now - float(msg['sync']))
                    if msg.get('event') == 'state':
                        print(f'  {label}: {msg.get("state")}', flush=True)
        status = proc.wait()
        rec['offset'] = min(offsets) if offsets else None
        rec['board_status'] = status
        local = self.ep_dir(label)
        subprocess.run(['scp', '-q', f'{a.board_host}:{remote_out}/*', str(local)], check=False)
        path = local / 'result.json'
        return json.loads(path.read_text()) if path.exists() else {'end': 'error',
                                                                     'error': f'board exit {status}'}

    # ── 3C fixtures ─────────────────────────────────────────────────────

    def load_toy(self, toy: str, rep: int, label: str) -> dict:
        """Get the toy into the beak where the robot stands: GroundPick with the toy at 1B's ideal
        spot (up to 3 tries), else 1B's fixture (stage1/s1g_carry.py load, without its move)."""
        from microduck_toy.stage1 import s1b_grasp
        pick_cfg = self.cfg['pick']
        out = {'picks': [], 'by': None}
        self.robot.head('H2')
        time.sleep(1.0)
        for k in range(3):
            self.place_in_trunk(toy, float(pick_cfg['ideal_toy_forward_m']),
                                float(pick_cfg['ideal_toy_left_m']), math.radians(60.0 * (rep % 3)))
            time.sleep(0.3)
            result = ground_pick(self.robot, float(self.configs['facts1']['grasp']['phi_close']),
                                 float(pick_cfg['period_s']), float(self.cfg['grasp']['held_threshold_deg']),
                                 confirm_s=float(pick_cfg['confirm_after_s']),
                                 mark=lambda tag: self.mark(f'{label}_load{k}_{tag}'))
            out['picks'].append(result)
            if result.get('held') and not result.get('fallen'):
                out['by'] = 'pick'
                return out
            self.robot.mouth(1.0)
            time.sleep(0.8)
            self.robot.mouth(0.0)
            time.sleep(1.0)
        s1b_grasp.hold_at_tip(self, toy)
        out['by'] = 'fixture'
        return out


class Pusher:
    """3D d_fall: push the robot while the brain walks in `state`, until it falls."""

    def __init__(self, runner: Runner, brain, state: str, direction: str, vx: float, vy: float):
        self.runner, self.brain = runner, brain
        self.state, self.direction, self.vx, self.vy = state, direction, vx, vy
        self.done: dict = {'pushed': False}
        self.finished = False
        self.armed = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def on_state(self, state: str) -> None:
        if state == self.state and not self.finished:
            self.armed.set()

    def stop(self) -> None:
        """Episode over: no push may land in the next one."""
        self.finished = True
        self.armed.set()

    def _run(self) -> None:
        try:
            self._walks()
        finally:
            self.finished = True

    def _walks(self) -> None:
        robot = self.runner.robot
        # Stage 0's push-to-fall speed was found standing; walking, velstand rides out 1.05 m/s about
        # one time in three (cloud 3D: 6/9 fell). A third push right after the second threw the robot
        # about 1 m, so a robot still up after two waits for its next walk in the state (another
        # B step or route leg, or the next time the brain enters the state) and is pushed there.
        # Any fall after the first push ends the pushing, also one slower than PUSH_FALL_S.
        self.done['speeds'] = []
        falls0 = len(robot.falls)
        fell = lambda: len(robot.falls) > falls0
        for walk in range(PUSH_WALKS):
            if not self.armed.wait(170.0) or self.finished or fell():
                return
            try:
                robot.wait(lambda s: self.finished or (s['policy'] == 'walk' and self.brain.state == self.state),
                           20.0, 'walking')
            except TimeoutError:
                if not self.done['pushed']:
                    self.done['why'] = f'never walked in {self.state}'
                return
            time.sleep(0.4)
            if self.finished or fell():
                return
            if self.brain.state != self.state:
                if not self.done['pushed']:
                    self.done['why'] = 'left the state before the push'
                self.armed.clear()
                continue
            for k in range(PUSH_TRIES):
                gain = 1.0 + PUSH_STEP * k
                mark = self.runner.mark(f'push_{self.direction}_{walk}_{k}')
                reply = self.runner.probe.call('push', vx=self.vx * gain, vy=self.vy * gain)
                if not self.done['pushed']:
                    self.done.update(pushed=True, mono=mark, reply=reply)
                    self.done.pop('why', None)
                self.done['speeds'].append(round(math.hypot(self.vx, self.vy) * gain, 3))
                self.done['walks'] = walk + 1
                try:
                    robot.wait(lambda s: s['safety'].get('fallen'), PUSH_FALL_S, 'fall')
                    return
                except TimeoutError:
                    pass
                if self.brain.state != self.state:
                    break
            try:                                    # this walk ends; the next one gets the next pushes
                robot.wait(lambda s: self.finished or s['policy'] != 'walk' or self.brain.state != self.state,
                           30.0, 'walk end')
            except TimeoutError:
                return
            if fell():
                return
            if self.brain.state != self.state:
                self.armed.clear()


def board_command(a, remote_out: str, target: str, start, until: str) -> str:
    """The shell line ssh runs on the board for one episode. With --board-cpus the brain is pinned
    (taskset -c; the A76 big cores 4-7 preprocess a frame in 7.0 ms against 18.3 ms on the A55s,
    configs/vision/stage2_facts.json deploy_board.preprocess_cores)."""
    pin = f'taskset -c {shlex.quote(a.board_cpus)} ' if a.board_cpus else ''
    return (f'cd {shlex.quote(a.board_bundle)} && MD_ROOT="$PWD" PYTHONPATH="$PWD" '
            f'{pin}{shlex.quote(a.board_py)} -u -m microduck_toy.stage3.brain --target {target} '
            f'--start {",".join(f"{v:.5f}" for v in start)} '
            f'--socket {shlex.quote(a.board_socket)} --camera {a.sim_host}:7901 '
            f'--tof {a.sim_host}:7801 --backend {a.backend} --until {until} '
            f'--out {shlex.quote(remote_out)}')


# ── parts ───────────────────────────────────────────────────────────────────

def part_ab(r: Runner) -> None:
    eps = dev_episodes(r.args.episodes, r.args.quick)
    for k, ep in enumerate(eps):
        rec = r.full(ep, 'ab', 'held')
        r.trial(rec)
        b = rec['brain']
        print(f'ab {k + 1}/{len(eps)} {ep["seed"]} {ep["target"]}: {b["end"]}, closes {b["closes"]}, '
              f'{b["elapsed_s"]:.0f} s', flush=True)


def part_e(r: Runner) -> None:
    eps = dev_episodes(r.args.episodes, r.args.quick)
    for k, ep in enumerate(eps):
        rec = r.full(ep, 'e', 'delivered')
        r.trial(rec)
        b = rec['brain']
        print(f'e {k + 1}/{len(eps)} {ep["seed"]} {ep["target"]}: {b["end"]}, closes {b["closes"]}, '
              f'{b["elapsed_s"]:.0f} s, 判定 {quick_verdict(r, rec)}', flush=True)


def quick_verdict(r: Runner, rec: dict) -> str:
    return judge_end(rec, r.meta['tray_centre'], r.s3)['outcome']


def c_trials(quick: bool, ds) -> list[dict]:
    out, i = [], 0
    for d in ds:
        for rep in range(1 if quick else 3):
            for t in (TARGETS[:1] if quick else TARGETS):
                rng = random.Random(C_SEED + i)
                out.append({'i': i, 'seed': C_SEED + i, 'd': float(d), 'rep': rep, 'target': t,
                            'dyaw_deg': rng.uniform(-10.0, 10.0), 'dleft_m': rng.uniform(-0.05, 0.05)})
                i += 1
    return out


def part_c_drop(r: Runner) -> None:
    from microduck_toy.stage3 import tray as tray_lib
    from microduck_toy.stage3.world import Field
    s3 = r.s3
    fld = Field.from_config(r.cfg, s3)
    sx, sy = tray_lib.staging_point('x-', fld, float(s3['carry']['staging_m']))
    ds = [float(r.args.drop_d)] if r.args.drop_d else s3['drop']['d_sweep_m']
    trials = c_trials(r.args.quick, ds)
    for k, t in enumerate(trials):
        label = f'c_drop_{t["i"]}_{t["target"]}_{t["d"]:.2f}'
        true = (sx, sy + t['dleft_m'], math.radians(t['dyaw_deg']))
        r.reset(head='H2', robot=true)
        r.wait_settled(3.0)
        odom0 = odom_of(r.robot.snapshot())
        rec = {'part': 'c_drop', 'label': label, **t, 'true_start': list(true),
               'told_start': [sx, sy, 0.0], 'grasps_from': r.grasp_count(), 'offset': 0.0}
        rec['load'] = r.load_toy(t['target'], t['rep'], label)
        rec['start'] = r.mark(label)
        b = r.brain(label, t['target'], (sx, sy, 0.0), odom0=odom0, start_state='TRAY_LOOK',
                    holding=True, drop_d=t['d'], side='x-')
        rec['brain'] = b.run()
        r.trial(r.finish(rec, label))
        print(f'c_drop {k + 1}/{len(trials)} d={t["d"]:.2f} {t["target"]}: loaded by '
              f'{rec["load"]["by"]}, {rec["brain"]["end"]}, 判定 {quick_verdict(r, rec)}', flush=True)


def part_c_route(r: Runner) -> None:
    lays = load_layouts()[:1 if r.args.quick else 3]
    targets = TARGETS[:1] if r.args.quick else TARGETS
    k = 0
    for lay in lays:
        for target in targets:
            rng = random.Random(C_SEED + 500 + k)
            tx, ty, _ = lay['slots']['target']
            yaw = rng.uniform(-math.pi, math.pi)
            tip = r.facts['head_geometry_mac']['H2']['tip_forward_cm'] / 100 + 0.02
            robot = (tx - tip * math.cos(yaw), ty - tip * math.sin(yaw), yaw)
            others = {n: v for n, v in lay['slots'].items() if n != 'target'}
            label = f'c_route_{lay["seed"]}_{target}'
            r.reset(head='H2', robot=robot, toys=others)
            r.wait_settled(3.0)
            odom0 = odom_of(r.robot.snapshot())
            known = [(n, (v[0] + rng.gauss(0, 0.03), v[1] + rng.gauss(0, 0.03))) for n, v in others.items()]
            rec = {'part': 'c_route', 'label': label, 'seed': lay['seed'], 'target': target,
                   'true_start': list(robot), 'known': known, 'layout': {'robot': list(robot), 'toys': others},
                   'grasps_from': r.grasp_count(), 'offset': 0.0}
            rec['load'] = r.load_toy(target, k, label)
            rec['start'] = r.mark(label)
            b = r.brain(label, target, robot, odom0=odom0, start_state='CARRY', holding=True,
                        known=known)
            rec['brain'] = b.run()
            r.trial(r.finish(rec, label))
            print(f'c_route {k + 1}/{len(lays) * len(targets)} {lay["seed"]} {target}: '
                  f'{rec["brain"]["end"]}, 判定 {quick_verdict(r, rec)}', flush=True)
            k += 1


def start_to_target(lay: dict) -> float:
    return math.dist(lay['robot'][:2], lay['slots']['target'][:2])


def part_d_fall(r: Runner) -> None:
    # GOTO only happens when the target is confirmed beyond the B start (0.65 m), so the GOTO pushes
    # get the three layouts with the farthest target; APPROACH and CARRY the next six.
    lays = sorted(load_layouts(), key=start_to_target, reverse=True)
    trials = []
    for i, state in enumerate(PUSH_STATES):
        for j, target in enumerate(TARGETS):
            trials.append((state, target, lays[(i * 3 + j) % len(lays)]))
    if r.args.quick:
        trials = trials[:1]
    for k, (state, target, lay) in enumerate(trials):
        name, vx, vy = PUSH_DIRS[target]
        rec = r.full({'seed': lay['seed'], 'target': target, 'layout': lay}, f'd_fall_{state.lower()}',
                     'delivered', push=(state, name, vx, vy))
        r.trial(rec)
        print(f'd_fall {k + 1}/{len(trials)} {state} {target} push {name}: pushed '
              f'{rec["push"].get("pushed")}, {rec["brain"]["end"]}, 判定 {quick_verdict(r, rec)}', flush=True)


D_TOF = (('H0', 0.3, 0.0), ('H0', 0.3, 30.0), ('H0', 0.3, -30.0),
         ('H1', 0.25, 0.0), ('H1', 0.25, 30.0), ('H1', 0.25, -30.0))
D_TOF_START_M = 0.75          # trunk to the -x wall at the start


def part_d_tof(r: Runner) -> None:
    half = float(r.cfg['arena']['half_size_m'])
    trials = D_TOF[:2] if r.args.quick else D_TOF
    for k, (head, vx, angle) in enumerate(trials):
        label = f'd_tof_{head}_{angle:+.0f}'
        yaw = math.pi + math.radians(angle)
        robot = (-half + D_TOF_START_M, 0.0, yaw)
        r.reset(head=head, robot=robot)
        r.wait_settled(3.0)
        rec = {'part': 'd_tof', 'label': label, 'head': head, 'vx': vx, 'angle_deg': angle,
               'grasps_from': r.grasp_count(), 'start': r.mark(label), 'offset': 0.0}
        b = r.brain(label, 'bus', robot)
        b.t0 = time.monotonic()
        b.head(head)
        time.sleep(1.0)
        try:
            out = b.walk(1.5, 0.0, vx)
            rec['walk'] = out
        except Exception as error:              # a fall or the clock: recorded, not fatal
            rec['walk'] = {'error': f'{type(error).__name__}: {error}'}
        time.sleep(1.5)
        rec['brain'] = {'end': 'tof_test', 'tof_stops': b.tof_stops, 'closes': 0,
                        'elapsed_s': round(time.monotonic() - b.t0, 3), 'history': []}
        rec['end_mark'] = r.mark(f'{label}_end')
        rec['judge_row'] = r.probe.truth()
        rec['grasps'] = []
        r.trial(rec)
        print(f'd_tof {k + 1}/{len(trials)} {head} {angle:+.0f}°: stops {len(b.tof_stops)}', flush=True)


PARTS = {'ab': part_ab, 'c_drop': part_c_drop, 'c_route': part_c_route, 'd_fall': part_d_fall,
         'd_tof': part_d_tof, 'e': part_e, **measure.PARTS}


# ── judging (truth only) ────────────────────────────────────────────────────

def in_tray(row: dict, toy: str, centre, half: float, max_z: float) -> bool:
    return (abs(row[f'{toy}_x'] - centre[0]) < half and abs(row[f'{toy}_y'] - centre[1]) < half
            and row[f'{toy}_z'] < max_z)


def judge_end(rec: dict, centre, s3: dict, rows: list[dict] | None = None) -> dict:
    """The episode's one end reason (plan section 2) from the brain's end and the truth."""
    ep = s3['episode']
    row, target = rec['judge_row'], rec['target']
    half, strict, max_z = float(ep['delivered_half_m']), float(ep['strict_half_m']), float(ep['delivered_max_z_m'])
    out = {'delivered_main': in_tray(row, target, centre, half, max_z),
           'delivered_strict': in_tray(row, target, centre, strict, max_z),
           'others_in_tray': [t for t in TOYS if t != target and in_tray(row, t, centre, half, max_z)],
           'target_from_tray_m': [round(row[f'{target}_x'] - centre[0], 4),
                                  round(row[f'{target}_y'] - centre[1], 4), round(row[f'{target}_z'], 4)]}
    held = set()
    for g in rec.get('grasps', []):
        if g.get('result') == 'grasped' and g.get('candidate'):
            held.add(g['candidate'])
    if rows is not None:
        for r in metrics.window(rows, rec['start'], rec.get('judge_mark', rec['end_mark'])):
            if r.get('holding'):
                held.add(r['holding'])
    out['held_toys'] = sorted(held)
    wrong = bool(held - {target}) or bool(out['others_in_tray'])
    end = rec['brain']['end']
    if end == 'dropped':
        outcome = 'delivered' if out['delivered_main'] else ('wrong_toy' if wrong else 'drop_missed')
    elif wrong:
        outcome = 'wrong_toy'
    else:
        outcome = end
    out['outcome'] = outcome
    return out


def falls_in(rows: list[dict], start: float, end: float) -> list[dict]:
    """Each truth fall (down for FALL_HOLD_S) in the window and how long until up again."""
    out, t = [], start
    while True:
        onset = metrics.fall_time(rows, t, end)
        if onset is None:
            return out
        up = metrics.first_hold(rows, lambda r: not metrics.down(r), onset, 0.5, end + 10.0)
        out.append({'onset': onset, 'up_after_s': None if up is None else round(up - onset, 3)})
        if up is None:
            return out
        t = up + 0.5


def read_brain_log(run_dir: Path, rec: dict) -> list[dict]:
    path = run_dir / 'episodes' / rec['label'] / 'brain.jsonl'
    if not path.exists():
        return []
    off = rec.get('offset') or 0.0
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            row['mono'] = row['mono'] + off
            for key in ('mono_frame',):
                if key in row and row[key] is not None:
                    row[key] += off
            rows.append(row)
    return rows


def truth_bearing(row: dict, toy: str) -> float:
    return wrap(math.atan2(row[f'{toy}_y'] - row['y'], row[f'{toy}_x'] - row['x']) - row['yaw'])


def in_picture(row: dict, toy: str, view: str = 'window', margin_deg: float = 2.0) -> bool:
    """Is the toy's centre above the bottom edge of the picture a look uses, by the true camera pose?
    The search window is the 360 rows from 30 above the horizon (configs/vision/stage2.json), so the
    floor nearer than its bottom edge is out of it: with H0 standing that is about 0.33 m from the
    trunk (Mac 3ab: 7 of the 10 looks that "missed" had the toy 0.10-0.26 m away)."""
    pitch = row['cam_pitch']
    if view == 'window':
        horizon = views.FRAME_H / 2 - views.FOCAL_PX * math.tan(pitch)
        bottom = views.window_top(horizon) + views.WINDOW
    else:
        bottom = views.FRAME_H
    edge = pitch + math.atan((bottom - views.FRAME_H / 2) / views.FOCAL_PX)
    toy_dep = math.atan2(row['cam_z'] - row[f'{toy}_z'],
                         math.hypot(row[f'{toy}_x'] - row['cam_x'], row[f'{toy}_y'] - row['cam_y']))
    return toy_dep <= edge - math.radians(margin_deg)


def score_stops(log: list[dict], rows: list[dict], b: dict, s1: dict) -> list[dict]:
    """Each ToF stop with the walls in the truth: straight ahead, and the nearest one the sensor's
    +-22.5 deg could see (as forward distance, the stop rule's measure); false = none within 0.3 m."""
    out = []
    for st in b.get('tof_stops', []):
        ev = next((e for e in log if e['event'] == 'tof_stop' and abs(e['el'] - st['el']) < 0.05), None)
        if ev is None:
            continue
        row = metrics.at_mark(rows, ev['mono'])
        seen = wall_in_view(row, s1)
        out.append({'state': st.get('state'), 'wall_ahead_m': round(wall_ahead(row, s1), 3),
                    'wall_in_view_m': round(seen, 3), 'false': seen > 0.3})
    return out


def score_episode(run_dir: Path, rec: dict, rows: list[dict], meta: dict) -> dict:
    s3 = meta['stage3_config']
    s1 = meta['stage1_config']
    centre = meta['tray_centre']
    b = rec['brain']
    end_t = rec.get('end_mark', rec['start'])
    out = {k: rec.get(k) for k in ('part', 'label', 'seed', 'target', 'd', 'head', 'angle_deg')
           if rec.get(k) is not None}
    out['brain_end'] = b['end']
    out['elapsed_s'] = b.get('elapsed_s')
    if rec['part'] == 'd_tof':              # a walk at a wall, not an episode
        out['tof_stops'] = score_stops(read_brain_log(run_dir, rec), rows, b, s1)
        out['falls'] = falls_in(rows, rec['start'], end_t)
        out.update(score_tof_walk(rows, rec, s1))
        return out
    target = rec['target']
    out.update(judge_end(rec, centre, s3, rows))
    out['falls'] = falls_in(rows, rec['start'], end_t)
    out['falls_seen'] = len(b.get('falls', []))
    log = read_brain_log(run_dir, rec)
    # Non-target toys pushed more than disturbed_m.
    if rec.get('layout'):
        a, z = metrics.at_mark(rows, rec['start']), metrics.at_mark(rows, end_t)
        out['disturbed'] = [t for t in rec['layout'].get('toys', {}) if t != target
                            and metrics.displacement(a, z, t) > float(s3['episode']['disturbed_m'])]
    # Closes: per close the weld result; wrong grab = a non-target welded.
    off = rec.get('offset') or 0.0
    windows = [(p['trigger'] + off, p.get('confirm', p.get('handback', p['trigger'] + 6.0)) + off + 0.5)
               for p in b.get('picks', []) if p.get('trigger') is not None]
    closes = [g for g in rec.get('grasps', []) if g.get('result') in
              ('grasped', 'missed', 'out_of_reach', 'already_holding')
              and any(lo <= g['mono'] <= hi for lo, hi in windows)]
    out['closes'] = [{'result': g.get('result'), 'toy': g.get('candidate') or g.get('nearest'),
                      'e_xy_m': g.get('e_xy', g.get('nearest_exy'))} for g in closes]
    out['wrong_grabs'] = sum(1 for c in out['closes'] if c['result'] == 'grasped' and c['toy'] != target)
    out['grasped'] = sum(1 for c in out['closes'] if c['result'] == 'grasped' and c['toy'] == target)
    tx, ty = s1['inching']['target_m']
    park = []
    for p in b.get('picks', []):
        if p.get('trigger') is not None:
            row = metrics.at_mark(rows, p['trigger'] + (rec.get('offset') or 0.0))
            f, l = metrics.toy_in_trunk(row, target)
            park.append([round(f - tx, 4), round(l - ty, 4)])
    out['park_err_m'] = park
    out['tof_stops'] = score_stops(log, rows, b, s1)
    # Search (3A): hand-overs into the B segment, looks, spin bearings, time to confirm.
    out.update(score_search(log, rows, rec, target))
    out['states_s'] = {}
    for name, t0, t1 in b.get('history', []):
        if t1 is not None:
            out['states_s'][name] = round(out['states_s'].get(name, 0.0) + t1 - t0, 3)
    if any(e['event'] == 'tray_look' for e in log):
        out.update(score_tray(log, rows, rec, meta))
    if rec.get('push'):
        out['push'] = rec['push']
    lat = (b.get('latency') or {}).get('e2e') or {}
    out['latency_e2e_p99_ms'] = lat.get('p99')
    if 'board_status' in rec:
        out['board_status'] = rec['board_status']
        out['robotd'] = episode_ticks(run_dir / 'episodes' / rec['label'] / 'state.jsonl')
    return out


def stage2_deploy() -> dict:
    return json.loads((repo_root() / 'configs' / 'vision' / 'stage2.json').read_text())['deploy']


def episode_ticks(path: Path) -> dict | None:
    """robotd's loop over one board episode, from the brain's robot.state log (2F's rule)."""
    from microduck_toy.stage2.deploy import TickTap
    if not path.exists():
        return None
    tap = TickTap()
    for line in path.read_text().splitlines():
        if line.strip():
            state = json.loads(line)
            tap(state, float(state.get('mono', 0.0)))
    return tap.summary(stage2_deploy()['stage0_robotd'])


def wall_ahead(row: dict, s1: dict) -> float:
    """Distance from the trunk along its heading to the inner face of the walls."""
    half = float(s1['arena']['half_size_m'])
    c, s = math.cos(row['yaw']), math.sin(row['yaw'])
    best = math.inf
    for v, d in ((row['x'], c), (row['y'], s)):
        if abs(d) > 1e-6:
            wall = half if d > 0 else -half
            best = min(best, (wall - v) / d)
    return best


TOF_HALF_FOV_DEG = 22.5        # VL53L5CX 45 deg per axis (third_party/microduck_rl/src/mjlab_microduck/sim/tof.py:29)


def wall_in_view(row: dict, s1: dict) -> float:
    """Nearest wall return inside the ToF's horizontal field of view, as distance along the heading
    (a wall met at a slant is seen first by the side zones: cloud 3D, 0.38 m straight ahead, 0.18 at the edge)."""
    best = math.inf
    for k in range(-10, 11):
        b = math.radians(TOF_HALF_FOV_DEG) * k / 10
        ray = wall_ahead({**row, 'yaw': row['yaw'] + b}, s1)
        best = min(best, ray * math.cos(b))
    return best


def relative_error(b_event: dict, row: dict, target: str) -> float | None:
    """The handed-over spot against the true toy, both seen from the robot (brain pose vs true pose):
    what the B segment acts on, without the odometry drift that to_target_m also holds."""
    pose = b_event.get('pose')
    if not pose:
        return None
    sx, sy = b_event['spot']['xy']
    bx, by = sx - pose[0], sy - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    brain = (c * bx + s * by, -s * bx + c * by)
    tx, ty = row[f'{target}_x'] - row['x'], row[f'{target}_y'] - row['y']
    c, s = math.cos(row['yaw']), math.sin(row['yaw'])
    true = (c * tx + s * ty, -s * tx + c * ty)
    return round(math.dist(brain, true), 3)


def score_search(log: list[dict], rows: list[dict], rec: dict, target: str) -> dict:
    out = {}
    toys = list((rec.get('layout') or {}).get('toys', {}))
    b_events = [e for e in log if e['event'] == 'b_segment']
    if b_events:
        e = b_events[0]
        row = metrics.at_mark(rows, e['mono'])
        xy = e['spot']['xy']
        dists = {t: math.dist(xy, (row[f'{t}_x'], row[f'{t}_y'])) for t in toys}
        out['handover'] = {'to_target_m': round(dists.get(target, math.inf), 3),
                           'relative_m': relative_error(e, row, target),
                           'correct': dists.get(target, math.inf) <= 0.15,
                           'distractor': any(d <= 0.15 for t, d in dists.items() if t != target)
                           and dists.get(target, math.inf) > 0.15,
                           'el': e['el']}
    confirmed = next((e for e in log if e['event'] == 'confirmed'), None)
    out['confirm_s'] = confirmed['el'] if confirmed else None
    looks = []
    for e in log:
        if e['event'] != 'look':
            continue
        row = metrics.at_mark(rows, e['mono'])
        d = math.hypot(row[f'{target}_x'] - row['x'], row[f'{target}_y'] - row['y'])
        brg = math.degrees(truth_bearing(row, target))
        seen_by_view = in_picture(row, target, e.get('view', 'window'))
        should = d <= 1.6 and abs(brg) <= 20.0 and row[f'{target}_z'] < 0.06 and seen_by_view
        looks.append({'result': e.get('result'), 'hits': e.get('hits'), 'truth_m': round(d, 3),
                      'truth_bearing_deg': round(brg, 1), 'in_picture': seen_by_view, 'should_see': should,
                      'saw': (e.get('hits') or 0) >= 3})
    out['looks'] = looks
    errs = []
    for e in log:
        if e['event'] != 'spin_frame':
            continue
        for s in e.get('seen', []):
            if s['cls'] != target or s.get('bearing') is None:
                continue
            row_f = metrics.at_mark(rows, e['mono_frame']) if e.get('mono_frame') else None
            row_r = metrics.at_mark(rows, e['mono'])
            if row_f is None:
                continue
            truth = truth_bearing(row_f, target)
            errs.append({'frame_deg': round(math.degrees(wrap(s['bearing'] - truth)), 2),
                         'receipt_deg': round(math.degrees(wrap(s['bearing'] - truth_bearing(row_r, target))), 2),
                         'dist_m': round(s.get('dist') or 0.0, 3)})
    out['spin_bearing_err'] = errs
    tray = [s['xy'] for e in log if e['event'] == 'spin_frame' for s in e.get('seen', [])
            if s['cls'] == 'tray' and s.get('xy')]
    out['spin_tray_xy'] = tray[:20]
    return out


def score_tray(log: list[dict], rows: list[dict], rec: dict, meta: dict) -> dict:
    centre = meta['tray_centre']
    out = {}
    looks = [e for e in log if e['event'] == 'tray_look']
    if looks:
        e = looks[-1]
        row = metrics.at_mark(rows, e['mono'])
        truth = metrics.point_in(row, centre[0], centre[1])
        out['tray_look'] = {'source': e.get('source'), 'used': e.get('used'),
                            'measured': e.get('centre'), 'truth': [round(v, 4) for v in truth],
                            'err_m': [round(e['centre'][0] - truth[0], 4), round(e['centre'][1] - truth[1], 4)]}
    side = next((e for e in reversed(log) if e['event'] == 'side'), None)
    if looks and side:
        row = metrics.at_mark(rows, looks[-1]['mono'])
        out['staging_err_m'] = [round(row['x'] - side['staging'][0], 4), round(row['y'] - side['staging'][1], 4)]
        out['staging_yaw_err_deg'] = round(math.degrees(wrap(row['yaw'] - side['face_yaw'])), 1)
    drop = next((e for e in log if e['event'] == 'drop_open'), None)
    if drop:
        row = metrics.at_mark(rows, drop['mono'])
        c = metrics.point_in(row, centre[0], centre[1])
        out['trunk_to_centre_m'] = round(math.hypot(*c), 4)
        if looks:
            # the drop walk: planned on odometry vs what the trunk really moved (coast included)
            before = metrics.at_mark(rows, looks[-1]['mono'])
            moved = math.hypot(row['x'] - before['x'], row['y'] - before['y'])
            planned = (drop.get('plan') or {}).get('distance')
            out['drop_walk_m'] = {'planned': None if planned is None else round(planned, 4),
                                  'truth': round(moved, 4),
                                  'yaw_change_deg': round(math.degrees(wrap(row['yaw'] - before['yaw'])), 1),
                                  'planned_bearing_deg': None if not drop.get('plan') else
                                  round(math.degrees(drop['plan']['bearing']), 1)}
            out['stumbled'] = planned is not None and moved - planned > STUMBLE_M
        t = rec['target']
        out['toy_in_beak_m'] = [round(v, 4) for v in metrics.point_in(row, row[f'{t}_x'], row[f'{t}_y'])]
        span = metrics.window(rows, rec['start'], rec.get('judge_mark', rec['end_mark']))
        out['tray_contact'] = any(r.get('tray_contact') for r in span)
    out['holding_at_start'] = bool(rec.get('load', {}).get('by'))
    return out


def score_tof_walk(rows: list[dict], rec: dict, s1: dict) -> dict:
    half = float(s1['arena']['half_size_m'])
    span = metrics.window(rows, rec['start'], rec['end_mark'])
    gap = [r['x'] + half for r in span]
    tip = [r['tip_x'] + half for r in span]
    out = {'stopped': bool(rec['brain']['tof_stops']),
           'min_trunk_to_wall_m': round(min(gap), 3) if gap else None,
           'min_tip_to_wall_m': round(min(tip), 3) if tip else None,
           'wall_contact': any(r.get('wall_contact') for r in span),
           'fell': metrics.fall_time(rows, rec['start'], rec['end_mark']) is not None}
    out['pass'] = bool(out['stopped'] and (out['min_trunk_to_wall_m'] or 0) >= 0.10
                       and not out['wall_contact'] and not out['fell'])
    return out


def pct(values, q) -> float | None:
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


def verdicts(results: list[dict], meta: dict | None = None) -> dict:
    by: dict[str, list] = {}
    for r in results:                       # d_fall parts are d_fall_<state>
        by.setdefault('d_fall' if r['part'].startswith('d_fall') else r['part'], []).append(r)
    out = {}
    if 'ab' in by:
        out['3a'], out['3b'] = verdict_3a(by['ab']), verdict_3b(by['ab'])
    if 'e' in by:
        out['3e'] = verdict_3e(by['e'])
        if (meta or {}).get('target') == 'board':
            out['3f'] = verdict_3f(by['e'], meta)
    if 'c_drop' in by:
        out['3c_drop'] = verdict_c_drop(by['c_drop'])
    if 'c_route' in by:
        out['3c_route'] = verdict_c_route(by['c_route'])
    if 'd_fall' in by:
        out['3d_fall'] = verdict_d_fall(by['d_fall'])
    if 'd_tof' in by:
        out['3d_tof'] = verdict_d_tof(by['d_tof'], results)
    return out


def verdict_3a(rs: list[dict]) -> dict:
    hand = [r.get('handover') for r in rs]
    looks = [l for r in rs for l in r.get('looks', []) if l['should_see']]
    errs = [abs(e['frame_deg']) for r in rs for e in r.get('spin_bearing_err', [])]
    errs_r = [abs(e['receipt_deg']) for r in rs for e in r.get('spin_bearing_err', [])]
    confirm = [r['confirm_s'] for r in rs if r.get('confirm_s') is not None]
    out = {'episodes': len(rs),
           'found_and_handed_over': sum(1 for h in hand if h and h['correct']),
           'distractor_into_b': sum(1 for h in hand if h and h['distractor']),
           'look_seen_rate': metrics.rate([l['saw'] for l in looks]),
           'confirm_s_mean': round(sum(confirm) / len(confirm), 2) if confirm else None,
           'spin_bearing_err_deg': {'frame_p50': pct(errs, 0.5), 'frame_p90': pct(errs, 0.9),
                                    'receipt_p50': pct(errs_r, 0.5), 'receipt_p90': pct(errs_r, 0.9),
                                    'n': len(errs)},
           'disturbed': sum(len(r.get('disturbed', [])) for r in rs),
           'ends': count(r['outcome'] for r in rs)}
    n = len(rs)
    out['pass'] = (out['found_and_handed_over'] >= math.ceil(28 / 30 * n) and out['distractor_into_b'] == 0
                   and (out['look_seen_rate']['rate'] or 0) >= 0.95) if n >= 30 else None
    out['rule'] = '>= 28/30 found and handed over, 0 distractors into B, standing looks see the target within 1.6 m >= 95%'
    return out


def verdict_3b(rs: list[dict]) -> dict:
    closes = [c for r in rs for c in r['closes']]
    park = [p for r in rs for p in r['park_err_m']]
    held3 = [any(c['result'] == 'grasped' for c in r['closes'][:3]) for r in rs if r.get('handover')]
    out = {'closes': len(closes), 'grasped': sum(r['grasped'] for r in rs),
           'success_per_close': round(sum(r['grasped'] for r in rs) / len(closes), 3) if closes else None,
           'held_within_3': metrics.rate(held3),
           'wrong_grabs': sum(r['wrong_grabs'] for r in rs),
           'park_rms_m': round(math.sqrt(sum(a * a + b * b for a, b in park) / len(park)), 4) if park else None,
           'falls': sum(len(r['falls']) for r in rs)}
    out['pass'] = ((out['success_per_close'] or 0) >= 0.5 and (out['held_within_3']['rate'] or 0) >= 0.8
                   and out['wrong_grabs'] == 0) if len(rs) >= 30 else None
    out['rule'] = '>= 0.5 per close, >= 80% held within 3 closes, 0 wrong grabs (watch: park RMS <= 2.5 cm, falls)'
    return out


def count(items) -> dict:
    out: dict = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def verdict_3e(rs: list[dict]) -> dict:
    n = len(rs)
    out = {'episodes': n, 'delivered': sum(r['outcome'] == 'delivered' for r in rs),
           'delivered_strict': sum(r['delivered_strict'] and r['outcome'] == 'delivered' for r in rs),
           'wrong_toy': sum(r['outcome'] == 'wrong_toy' for r in rs),
           'ends': count(r['outcome'] for r in rs),
           'mean_s': round(sum(r['elapsed_s'] or 0 for r in rs) / n, 1) if n else None,
           'falls': sum(len(r['falls']) for r in rs),
           'getup_s': [f['up_after_s'] for r in rs for f in r['falls']],
           'tof_stops': sum(len(r['tof_stops']) for r in rs),
           'false_tof_stops': sum(sum(s['false'] for s in r['tof_stops']) for r in rs),
           'disturbed': sum(len(r.get('disturbed', [])) for r in rs),
           'tray_contact': sum(bool(r.get('tray_contact')) for r in rs),
           'stumbles': sum(bool(r.get('stumbled')) for r in rs),
           'state_s_mean': {}}
    names = {k for r in rs for k in r['states_s']}
    for k in sorted(names):
        out['state_s_mean'][k] = round(sum(r['states_s'].get(k, 0.0) for r in rs) / n, 2)
    out['pass'] = (out['delivered'] >= math.ceil(0.7 * n) and out['wrong_toy'] == 0
                   and all(r['outcome'] in ('delivered', 'drop_missed', 'grasp_fail', 'search_fail',
                                            'wrong_toy', 'timeout') for r in rs)) if n >= 30 else None
    out['rule'] = 'delivered >= 21/30, 0 wrong toy, every episode ends with a reason within 180 s'
    return out


def verdict_3f(rs: list[dict], meta: dict) -> dict:
    """Board closed loop (plan 9): the first 9 episodes of 3E with the brain on the RK3588."""
    d = stage2_deploy()
    ticks = [r['robotd'] for r in rs if r.get('robotd')]
    p99 = [r['latency_e2e_p99_ms'] for r in rs if r.get('latency_e2e_p99_ms') is not None]
    total = sum(t.get('ticks_expected') or 0 for t in ticks)
    missed = sum(t.get('missed') or 0 for t in ticks)
    rate0 = d['stage0_robotd']['missed'] / d['stage0_robotd']['ticks']
    allowed = max(1, math.ceil(rate0 * total))
    mac = None
    if meta.get('mac_run'):
        done = json.loads((Path(meta['mac_run']) / 'summary.json').read_text())
        want = {(r['seed'], r['target']) for r in rs}
        mac = sum(1 for r in done.get('results', []) if r.get('part') == 'e'
                  and (r.get('seed'), r.get('target')) in want and r['outcome'] == 'delivered')
    out = {'n': len(rs), 'ended_with_reason': sum(r['outcome'] != 'error' for r in rs),
           'delivered': sum(r['outcome'] == 'delivered' for r in rs), 'mac_delivered_same': mac,
           'e2e_p99_ms_worst_episode': max(p99) if p99 else None, 'e2e_p99_ms': p99,
           'latency_logs': len(p99), 'robotd_ticks': total, 'robotd_missed': missed,
           'robotd_missed_allowed': allowed,
           'robotd_hz_min': min((t['hz_min'] for t in ticks if t.get('hz_min')), default=None),
           'ends': count(r['outcome'] for r in rs)}
    ok = (out['ended_with_reason'] == len(rs) and out['latency_logs'] == len(rs)
          and len(ticks) == len(rs) and missed <= allowed
          and (out['e2e_p99_ms_worst_episode'] or math.inf) <= float(d['e2e_p99_ms_max']))
    if len(rs) < 9:
        out['pass'] = None
    elif mac is None:
        out['pass'] = None
        out['needs'] = 'rescore with --mac-run runs/<the Mac 3E run> to compare deliveries'
    else:
        out['pass'] = bool(ok and out['delivered'] >= mac - 2)
    out['rule'] = (f'9/9 end with a reason; delivered >= the Mac\'s on the same 9 - 2; robotd missed <= '
                   f'max(1, Stage 0 rate x ticks); detection e2e P99 <= {d["e2e_p99_ms_max"]} ms in every '
                   'episode; a latency log for every episode')
    return out


def verdict_c_drop(rs: list[dict]) -> dict:
    by_d: dict = {}
    for r in rs:
        by_d.setdefault(f'{r["d"]:.2f}', []).append(r)
    table = {}
    for d, group in sorted(by_d.items()):
        errs = [r['tray_look']['err_m'] for r in group if r.get('tray_look')]
        table[d] = {'n': len(group), 'delivered': sum(r['delivered_main'] for r in group),
                    'strict': sum(r['delivered_strict'] for r in group),
                    'falls': sum(len(r['falls']) for r in group),
                    'tray_contact': sum(bool(r.get('tray_contact')) for r in group),
                    'stumbles': sum(bool(r.get('stumbled')) for r in group),
                    'trunk_to_centre_m': [r.get('trunk_to_centre_m') for r in group],
                    'tray_err_fwd_m': [e[0] for e in errs], 'tray_err_left_m': [e[1] for e in errs],
                    'toy_in_beak_m': [r.get('toy_in_beak_m') for r in group],
                    'toy_from_centre_m': [r.get('target_from_tray_m') for r in group],
                    'drop_walk_over_m': [round(r['drop_walk_m']['truth'] - r['drop_walk_m']['planned'], 4)
                                         for r in group if (r.get('drop_walk_m') or {}).get('planned') is not None]}
    safe = {d: t for d, t in table.items() if t['falls'] == 0}
    chosen = None
    if safe:
        chosen = max(safe, key=lambda d: (safe[d]['delivered'], -safe[d]['stumbles'], float(d)))
    out = {'by_d': table, 'chosen_d': chosen}
    out['pass'] = (chosen is not None and table[chosen]['delivered'] >= 8 and table[chosen]['n'] >= 9)
    out['rule'] = ('among d with 0 falls, the most deliveries (ties: fewer stumbles into the tray, then larger d);'
                   ' the chosen d needs >= 8/9')
    return out


def verdict_c_route(rs: list[dict]) -> dict:
    out = {'n': len(rs), 'delivered': sum(r['outcome'] == 'delivered' for r in rs),
           'drops_on_way': sum(1 for r in rs if r['brain_end'] != 'dropped' or r.get('recovered')),
           'falls': sum(len(r['falls']) for r in rs), 'ends': count(r['outcome'] for r in rs),
           'tray_contact': sum(bool(r.get('tray_contact')) for r in rs),
           'stumbles': sum(bool(r.get('stumbled')) for r in rs)}
    out['pass'] = out['delivered'] >= 8 and out['falls'] == 0 if len(rs) >= 9 else None
    out['rule'] = 'delivered >= 8/9, 0 drops on the way, 0 falls'
    return out


def verdict_d_fall(rs: list[dict]) -> dict:
    out = {'n': len(rs), 'pushed': sum(bool(r.get('push', {}).get('pushed')) for r in rs),
           'fell': sum(bool(r['falls']) for r in rs),
           'ended_with_reason': sum(r['outcome'] != 'error' for r in rs),
           'delivered': sum(r['outcome'] == 'delivered' for r in rs),
           'getup_s': [f['up_after_s'] for r in rs for f in r['falls']],
           'not_felled': [r['label'] for r in rs if not r['falls']],
           'by_state': {}}
    for r in rs:
        k = r['push']['state'] if r.get('push') else '?'
        out['by_state'].setdefault(k, []).append(r['outcome'])
    out['pass'] = (out['ended_with_reason'] == len(rs) and out['pushed'] == len(rs)
                   and out['fell'] == out['pushed']) if len(rs) >= 9 else None
    out['rule'] = ('9/9 pushed over in the named state and ending with a reason in 180 s (delivered >= 6/9 '
                   'watched); guard timing in 1E terms from the falls list. A robot the pushes did not fell '
                   '(not_felled) fails the rule as a push problem, not a brain one')
    return out


def verdict_d_tof(rs: list[dict], everything: list[dict]) -> dict:
    out = {'n': len(rs), 'stopped': sum(r['stopped'] for r in rs), 'passed': sum(r['pass'] for r in rs),
           'min_trunk_to_wall_m': [r['min_trunk_to_wall_m'] for r in rs],
           'false_stops_all_parts': sum(sum(s['false'] for s in r.get('tof_stops', []))
                                        for r in everything)}
    out['pass'] = out['passed'] == len(rs) and out['false_stops_all_parts'] == 0 if len(rs) >= 6 else None
    out['rule'] = '6/6 stop with the trunk >= 0.10 m from the wall, no wall contact; 0 false stops'
    return out


SIM_STALL_S = 0.035          # a 20 ms step that waited longer than this
SIM_MAX_STALLED = 0.02       # share of such steps a usable run may have
ODOM_RATIO_OK = (0.93, 1.07)  # robotd's odometry path over the true path while walking
RESET_JUMP_M = 0.05           # one 20 ms truth row moving further is a reset, not a step (walks and
                              # pushes stay under 3 cm a row; Mac 3p resets moved 10-20 cm)


def sim_health(rows: list[dict], state_path: Path | None = None) -> dict:
    """Did the sim body keep real 20 ms steps, and did robotd's odometry measure the walks right?

    The Mac body used to draw camera frames on the physics thread: 17% of its steps waited ~70 ms and
    the next ones ran back to back, and robotd's contact odometry counted ~30% short (3ab 2026-10-26).
    A run like that does not measure the brain, so the verdicts need `ok`."""
    gaps = sorted(b['mono'] - a['mono'] for a, b in zip(rows, rows[1:]) if 0.0 < b['t'] - a['t'] < 0.03)
    out: dict = {'steps': len(gaps)}
    if gaps:
        out.update(stalled=round(sum(g > SIM_STALL_S for g in gaps) / len(gaps), 4),
                   step_ms_p50=round(gaps[len(gaps) // 2] * 1000, 1),
                   step_ms_p99=round(gaps[int(len(gaps) * 0.99)] * 1000, 1))
    out['odom_ratio'] = odom_ratio(rows, state_path) if state_path and state_path.exists() else None
    ratio = out['odom_ratio']
    out['ok'] = (out.get('stalled', 0.0) <= SIM_MAX_STALLED
                 and (ratio is None or ODOM_RATIO_OK[0] <= ratio['ratio'] <= ODOM_RATIO_OK[1]))
    out['rule'] = (f'<= {SIM_MAX_STALLED:.0%} of steps longer than {SIM_STALL_S * 1000:.0f} ms; odometry / '
                   f'truth path while walking within {ODOM_RATIO_OK[0]}-{ODOM_RATIO_OK[1]}')
    return out


def gate_on_health(verdict: dict, health: dict) -> dict:
    """No gate passes on a sim that did not keep real time (each verdict gets sim_ok)."""
    for v in verdict.values():
        v['sim_ok'] = health['ok']
        if not health['ok'] and v.get('pass'):
            v['pass'] = False
            v['rule'] += ' (and a sim that kept real time: sim_health.ok)'
    return verdict


def odom_ratio(rows: list[dict], state_path: Path, every: int = 10) -> dict | None:
    """robotd's odometry path over the true trunk path, on 0.2 s pieces where the trunk moved
    2-20 cm (walking: not standing sway, not a fall) and no single row jumped (a reset: Mac 3p moved the
    robot 10-20 cm twice, which odometry never sees, and read 0.71 for a run that walked at 0.98)."""
    monos = [r['mono'] for r in rows]
    jumps = [k for k, (a, b) in enumerate(zip(rows, rows[1:]))
             if math.hypot(b['x'] - a['x'], b['y'] - a['y']) > RESET_JUMP_M]
    odo, true = 0.0, 0.0
    pieces = 0
    last = None
    with state_path.open() as f:
        for i, line in enumerate(f):
            if i % every:
                continue
            try:
                state = json.loads(line)
                pos = state['odom']['position']
                now = (float(state['mono']), float(pos[0]), float(pos[1]))
            except (ValueError, KeyError, TypeError):
                continue
            if last is not None:
                ia = min(bisect.bisect_left(monos, last[0]), len(rows) - 1)
                ib = min(bisect.bisect_left(monos, now[0]), len(rows) - 1)
                a, b = rows[ia], rows[ib]
                d_true = math.hypot(b['x'] - a['x'], b['y'] - a['y'])
                k = bisect.bisect_left(jumps, ia)
                if 0.02 < d_true < 0.2 and not (k < len(jumps) and jumps[k] < ib):
                    odo += math.hypot(now[1] - last[1], now[2] - last[2])
                    true += d_true
                    pieces += 1
            last = now
    if pieces < 20:
        return None
    return {'ratio': round(odo / true, 3), 'pieces': pieces, 'true_m': round(true, 2)}


def score(s, rows) -> dict:
    episodes = [rec for rec in s.trials if not rec['part'].startswith('m_')]
    results = [score_episode(s.dir, rec, rows, s.meta) for rec in episodes]
    health = sim_health(rows, s.dir / 'state.jsonl')
    from microduck_toy.stage3 import vision_loop        # it imports this module's geometry
    out = {'sim_health': health, 'results': results,
           'verdict': gate_on_health(verdicts(results, s.meta), health),
           'vision_loop': vision_loop.summarise(vision_loop.judge_run(s.dir, episodes, rows))}
    if len(episodes) < len(s.trials):
        out['measured_3p'] = measure.summarise(s.trials, rows)
    return out


# ── command line ────────────────────────────────────────────────────────────

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='stage3', description=__doc__.split('\n')[0])
    p.add_argument('test', choices=[*TESTS, 'layouts'])
    p.add_argument('--config', default='s2')
    p.add_argument('--target', choices=('mac', 'board'), default='mac')
    p.add_argument('--socket', default=None)
    p.add_argument('--probe', default='127.0.0.1:7811')
    p.add_argument('--camera', default='127.0.0.1:7901')
    p.add_argument('--source', choices=('stream', 'probe'), default='stream',
                   help='stream: the camera port (Mac, board); probe: the view op (cloud twin)')
    p.add_argument('--eyes', choices=('yolo', 'oracle'), default='yolo')
    p.add_argument('--backend', choices=('cpu', 'fp16', 'int8'), default=None,
                   help='default cpu on the Mac, int8 on the board')
    p.add_argument('--model', default=None)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--parts', default=None)
    p.add_argument('--episodes', default=None, help='e.g. 3000:bus,3001:henry (default: all 30)')
    p.add_argument('--drop-d', type=float, default=None, help='c_drop: one d instead of the sweep')
    p.add_argument('--no-frames', action='store_true',
                   help="don't save the robot's-eye pictures (episodes/<label>/frames/)")
    p.add_argument('--rescore', metavar='RUN_DIR', default=None)
    p.add_argument('--mac-run', metavar='RUN_DIR', default=None,
                   help='board 3e: the Mac 3E run whose same episodes the board is compared with')
    p.add_argument('--board-host', default=os.environ.get('MD_BOARD_HOST', 'RK3588'))
    p.add_argument('--board-py', default=os.environ.get('MD_BOARD_PY', '/home/linaro/microduck/runtime/venv/bin/python'))
    p.add_argument('--board-bundle', default=None)
    p.add_argument('--board-out', default=None)
    p.add_argument('--board-socket', default=os.environ.get('MD_BOARD_ROOT', '/home/linaro/microduck') + '/run/robotd.sock')
    p.add_argument('--sim-host', default=os.environ.get('MD_SIM_HOST', '192.168.50.1'))
    p.add_argument('--board-cpus', default=os.environ.get('MD_BOARD_CPUS', ''),
                   help='taskset -c list for the board brain (env.sh MD_BOARD_CPUS); empty = not pinned')
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.test == 'layouts':
        doc = make_layouts()
        print(f'{layouts_path()}: {len(doc["layouts"])} layouts')
        return 0
    if args.backend is None:
        args.backend = 'int8' if args.target == 'board' else 'cpu'
    if args.rescore:
        done = common.Rescored(args.rescore)
        done.meta['stage3_config'] = done.meta.get('stage3_config') or done.cfg
        if args.mac_run:
            done.meta['mac_run'] = str(Path(args.mac_run).resolve())
        rows = metrics.read_truth(done.dir / 'truth.csv')
        scored = score(done, rows)
        done.write_summary({'truth': metrics.continuity(rows), **scored})
        print_health(scored.get('sim_health'))
        print_vision(scored.get('vision_loop'))
        print_traces(done.dir, rows)
        return 0
    if args.target == 'board':
        missing = [n for n in ('board_bundle', 'board_out') if getattr(args, n) is None]
        if missing:
            raise SystemExit('board runs come from scripts/board/stage3/run_board_brain.zsh (needs --'
                             + ', --'.join(n.replace('_', '-') for n in missing) + ')')
        allowed = {'ab', 'e'}
    else:
        allowed = set(PARTS)
    chosen = args.parts.split(',') if args.parts else list(TESTS[args.test])
    bad = [c for c in chosen if c not in TESTS[args.test] or c not in allowed]
    if bad:
        raise SystemExit(f'parts {bad} are not in {args.test} (or not on the board): {list(TESTS[args.test])}')

    def body(session):
        for name in chosen:
            print(f'== {name}', flush=True)
            PARTS[name](session)

    session = Runner(args.test, args)
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
    try:
        scored = score(session, rows)
    except Exception as error:              # keep the run; fix the scorer, then --rescore
        scored = {'score_error': f'{type(error).__name__}: {error}'}
        print(f'计分出错（数据已保存，修好后用 --rescore 重算）：{scored["score_error"]}', flush=True)
        status = status or 1
    session.write_summary({'truth': metrics.continuity(rows), **scored})
    print_health(scored.get('sim_health'))
    print_vision(scored.get('vision_loop'))
    print_traces(session.dir, rows)
    return status


def print_health(h: dict | None) -> None:
    if not h or 'stalled' not in h:
        return
    ratio = h['odom_ratio']['ratio'] if h.get('odom_ratio') else None
    line = (f"仿真节拍：{h['stalled']:.1%} 的 20 ms 拍子超过 {SIM_STALL_S * 1000:.0f} ms（p99 {h['step_ms_p99']} ms）"
            + (f"；里程计/真值路程 {ratio}" if ratio is not None else ''))
    print(line, flush=True)
    if not h['ok']:
        print('警告：这次仿真没有按真实时间稳定运行，或里程计量错了路程；判定一律不算通过，先查仿真身体。', flush=True)


def print_traces(run_dir: Path, rows: list[dict]) -> None:
    """The bird's-eye trace of every episode (stage3/trace.py); a missing matplotlib only skips them."""
    from microduck_toy.stage3 import trace
    try:
        paths = trace.plot_run(run_dir, rows=rows)
    except Exception as error:
        print(f'俯视轨迹图没画成（{type(error).__name__}: {error}）；数据不受影响。', flush=True)
        return
    if paths:
        print(f'俯视轨迹图：每回合 episodes/<回合>/trace.png；总览 {paths[-1]}', flush=True)


def print_vision(v: dict | None) -> None:
    if not v or not v.get('frames'):
        return
    t, p = v['recall_targets'], v['precision_targets']
    if t['rate'] is None or p['rate'] is None:
        return
    print(f"闭环视觉（转圈和站着看的帧对真值）：目标召回 {t['rate']:.1%}（{t['k']}/{t['n']}），"
          f"精确 {p['rate']:.1%}；1.6 m 内认错类 {v['wrong_class_within_1_6m']} 个；"
          f"详表：python -m microduck_toy.stage3.vision_loop <运行目录>", flush=True)


if __name__ == '__main__':
    sys.exit(main())
