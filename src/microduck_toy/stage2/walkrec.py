"""Stage 2 test_walk (plan 2A step 7): detector test frames taken while robotd really walks.

    python -m microduck_toy.stage2.walkrec --data runs/<UTC>_s2a_dataset --floor grid \
        [--episodes 15] [--socket <robotd.sock>] [--probe 127.0.0.1:7811]

Needs the Stage 1 arena body on the same floor with the camera rendering on demand
(run_arena_body.zsh --floor grid --camera --oracle-fps 0) and robotd with config s2. Adds frames
to the dataset folder as split test_walk, in the same layout as datagen:

    DATA/images/test_walk/<id>.png, DATA/labels/test_walk/<id>.txt
    DATA/meta/test_walk.jsonl      one line per frame, appended as each episode ends

and logs the run (events, robot.state, episodes.jsonl) in runs/<UTC>_s2_walkrec_<floor>/.

One episode per seed (configs/vision/stage2.json datagen.splits.test_walk and datagen.walk): the
seed's search scene from datagen.sample gives the layout, the search head and the focus toy; seeds
whose floor is not --floor are skipped, so grid and wood runs never share a seed. The robot then
1. at the search head, turns spin_deg / 2 right, then sweeps spin_deg left without stopping, a
   window frame at the middle of each of spin_frames equal slices of odometry yaw (search view;
   the seed put the toys around the starting heading, so a sweep, not a full circle, sees them);
2. turns to the focus toy (bearing from the arena truth: a test fixture, the robot is not scored),
   puts the head at H1 and walks towards it (approach.walk's gait and heading PI) until it is
   stop_short_m away, a letterbox frame at the middle of each of walk_frames equal slices of
   odometry distance (approach view).
Every frame is taken while the robot is stepping (turning or walking), never standing.
Each frame is the arena body's oracle render (image + segmentation boxes at the same instant); the
window and the ranging state come from the next robot.state, as Stage 3 will see them (meta
horizon_row; truth.camera.horizon_row is the render's own, to measure how far off the robot's is). Labels go
through datagen.label_objects, so they follow the same rules as every other split.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from microduck_toy.control import approach, ranging
from microduck_toy.control.robot import Events, Robot, UPRIGHT
from microduck_toy.paths import runs_dir
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage0.link import Probe
from microduck_toy.stage1 import common
from microduck_toy.stage2 import datagen
from microduck_toy.vision import uyvy, views

SPLIT = 'test_walk'


def episode_seeds(cfg: dict, stage1: dict, floor: str, count: int) -> list[int]:
    sc = cfg['datagen']['splits'][SPLIT]
    seeds, seed = [], int(sc['seed_base'])
    while len(seeds) < count:
        if datagen.sample(seed, SPLIT, cfg, stage1)['floor'] == floor:
            seeds.append(seed)
        seed += 1
    return seeds


class Recorder:
    def __init__(self, data: Path, robot: Robot, probe: Probe, cfg: dict, stage1: dict):
        self.data, self.robot, self.probe = data, robot, probe
        self.cfg, self.stage1 = cfg, stage1
        for sub in ('images', 'labels'):
            (data / sub / SPLIT).mkdir(parents=True, exist_ok=True)
        (data / 'meta').mkdir(exist_ok=True)
        self.seq = -1
        self.halves = {short: toy_lib.catalogue()[toy_lib.resolve(short)].half
                       for short in datagen.SHORT_TOYS}

    def frame(self, fid: str, scene: dict, view: str, phase: str, head) -> dict:
        """One oracle render now; writes the input PNG and labels, returns the meta line."""
        v = self.probe.call('view', after=self.seq, rgb=True, timeout=15.0)
        self.seq = v['seq']
        state = self.robot.next_states(1, timeout=5.0)[0]
        upright = np.frombuffer(base64.b64decode(v['rgb_b64']), np.uint8).reshape(v['rgb_shape'])
        frame = uyvy.round_trip_upright(upright)
        cam_pos, cam_rot = ranging.camera_in_level(state)
        horizon = views.horizon_row(cam_rot)
        if view == 'search':
            mapping = views.window_mapping(horizon, int(self.cfg['input']['window_margin_rows']))
        else:
            mapping = views.letterbox_mapping()
        img = views.to_input(np.ascontiguousarray(frame[:, :, ::-1]), mapping)

        t = v['truth']
        trunk, heading = t['trunk'], float(t['yaw'])
        truth = {name: (*t[name][:3], self.halves[name]) for name in scene['toys']}
        truth['tray'] = (float(t['tray'][0]), float(t['tray'][1]), 0.0, None)
        c, s = math.cos(heading), math.sin(heading)
        if t.get('camera'):
            cam_xy = t['camera']['pos'][:2]
        else:
            cam_xy = (trunk[0] + c * cam_pos[0] - s * cam_pos[1],
                      trunk[1] + s * cam_pos[0] + c * cam_pos[1])
        labels, objects, dropped = datagen.label_objects(v['boxes'], truth, trunk[:2], heading,
                                                         cam_xy, mapping, self.cfg)
        cv2.imwrite(str(self.data / 'images' / SPLIT / f'{fid}.png'), img)
        (self.data / 'labels' / SPLIT / f'{fid}.txt').write_text(datagen.yolo_rows(labels))
        cam = state['frames']['camera']
        return {'id': fid, 'seed': scene['seed'], 'split': SPLIT, 'floor': scene['floor'],
                'view': view, 'phase': phase, 'head': list(head), 'focus': scene['focus'],
                'toys': scene['toys'], 'mapping': mapping.as_dict(),
                'horizon_row': round(horizon, 2) if horizon is not None else None,
                'objects': objects, 'dropped': dropped,
                'state': {'frames': {'camera': {'pos': cam['pos'], 'quat': cam['quat']}},
                          'imu': {'quat': state['imu']['quat']},
                          'odom': {'position': state['odom']['position']}},
                'robot_t': state['t'], 'policy': state['policy'],
                'truth': {'trunk': trunk, 'yaw': heading, 'camera': t.get('camera')},
                'render_seq': v['seq']}

    def fallen(self) -> bool:
        st = self.robot.snapshot()
        return st['policy'] not in UPRIGHT or bool(st['safety'].get('fallen'))

    def episode(self, seed: int) -> tuple[list[dict], dict]:
        w = self.cfg['datagen']['walk']
        acfg = self.stage1['approach']
        scene = datagen.sample(seed, SPLIT, self.cfg, self.stage1)
        head = tuple(scene['head'])
        robot = self.robot
        robot.move(0.0, 0.0, 0.0)
        robot.head(head)
        try:
            robot.wait(lambda st: st['policy'] in UPRIGHT and not st['safety'].get('fallen'), 8.0,
                       'upright before the layout')
        except TimeoutError:
            pass
        self.probe.call('layout', robot=scene['robot'], toys=scene['toys'], hold_s=1.0)
        time.sleep(1.0 + float(w['settle_s']))
        falls0 = len(robot.falls)
        metas, info = [], {'seed': seed, 'floor': scene['floor'], 'focus': scene['focus'],
                           'head': list(head)}

        # 1. Sweep at the search head, frames while turning.
        n_spin = int(w['spin_frames'])
        sweep = math.radians(float(w['spin_deg']))
        approach.turn(robot, -sweep / 2, acfg)
        if not self.fallen():
            info['spin'] = self.spin(scene, seed, sweep, [sweep * (k + 0.5) / n_spin
                                                          for k in range(n_spin)], head, metas)
        # 2. Turn to the focus toy, H1, walk to stop_short_m before it.
        row = self.probe.truth()
        tx, ty = scene['toys'][scene['focus']][:2]
        dx, dy = tx - row['x'], ty - row['y']
        bearing = approach.wrap(math.atan2(dy, dx) - row['yaw'])
        if abs(bearing) > math.radians(acfg['turn_tol_deg']) and not self.fallen():
            approach.turn(robot, bearing, acfg)
        robot.head('H1')
        time.sleep(float(w['settle_s']))
        row = self.probe.truth()
        dx, dy = tx - row['x'], ty - row['y']
        dist = math.hypot(dx, dy) - float(w['stop_short_m'])
        bearing = approach.wrap(math.atan2(dy, dx) - row['yaw'])
        info['walk_m'] = round(dist, 3)
        n_walk = int(w['walk_frames'])
        marks = [dist * (k + 0.5) / n_walk for k in range(n_walk)]
        if dist > 0.05 and not self.fallen():
            info['walk'] = self.walk(scene, seed, dist, bearing, marks, metas)
        robot.move(0.0, 0.0, 0.0)
        info['frames'] = len(metas)
        info['falls'] = len(robot.falls) - falls0
        return metas, info

    def spin(self, scene, seed, sweep, marks, head, metas) -> dict:
        """Turn left by `sweep` at approach.turn's speed, a frame as the odometry yaw passes each
        of `marks`."""
        cfg = self.stage1['approach']
        robot = self.robot
        _, _, yaw0 = approach.odom_of(robot.snapshot())
        robot.move(0.0, 0.0, cfg['turn_wz_radps'])
        turned, k, ticks = 0.0, 0, 0
        budget = int((sweep / (0.6 * cfg['turn_wz_radps']) + 4.0) * 50)
        while k < len(marks) and ticks < budget:
            state = robot.next_states(1, timeout=5.0)[0]
            ticks += 1
            turned = approach.wrap(approach.odom_of(state)[2] - yaw0)
            if turned >= marks[k]:
                if self.fallen():
                    break
                metas.append(self.frame(f'{SPLIT}_{seed:06d}_s{k}', scene, 'search', 'spin', head))
                k += 1
        robot.move(0.0, 0.0, 0.0)
        return {'turned_deg': round(math.degrees(turned), 1), 'frames': k, 'ticks': ticks}

    def walk(self, scene, seed, distance, bearing, marks, metas) -> dict:
        """approach.walk with frames at the odometry distances in `marks`; walking resumes after
        each frame (the body does not step while it renders)."""
        cfg = self.stage1['approach']
        robot = self.robot
        x0, y0, yaw0 = approach.odom_of(robot.snapshot())
        ref = yaw0 + bearing
        ux, uy = math.cos(ref), math.sin(ref)
        goal = max(0.0, distance - cfg['walk_slide_m'])
        kp, ki, cap = cfg['heading_kp'], cfg['heading_ki'], cfg['heading_max_radps']
        integral, last_t, travelled, k = 0.0, None, 0.0, 0
        ticks, budget = 0, int((goal / 0.08 + 4.0) * 50)
        while travelled < goal and ticks < budget:
            if k < len(marks) and travelled >= marks[k]:
                if self.fallen():
                    break
                metas.append(self.frame(f'{SPLIT}_{seed:06d}_w{k}', scene, 'approach', 'walk',
                                        robot.head_intent))
                k += 1
                last_t = None
            state = robot.next_states(1, timeout=5.0)[0]
            ticks += 1
            x, y, yaw = approach.odom_of(state)
            travelled = (x - x0) * ux + (y - y0) * uy
            t = float(state['t'])
            dt_ = 0.02 if last_t is None else max(1e-3, min(0.1, t - last_t))
            last_t = t
            err = approach.wrap(ref - yaw)
            integral = max(-cap / max(ki, 1e-9), min(cap / max(ki, 1e-9), integral + err * dt_))
            wz = max(-cap, min(cap, kp * err + ki * integral))
            robot.move(cfg['walk_vx_mps'], 0.0, wz)
        robot.move(0.0, 0.0, 0.0)
        return {'goal_m': round(goal, 3), 'travelled_m': round(travelled, 3), 'frames': k,
                'ticks': ticks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--floor', choices=('grid', 'wood'), required=True)
    ap.add_argument('--episodes', type=int, default=None, help='default: half the config count')
    ap.add_argument('--socket', default=None, help='robotd control socket')
    ap.add_argument('--probe', default='127.0.0.1:7811')
    args = ap.parse_args(argv)
    cfg, stage1 = datagen.load_configs()
    n = args.episodes or int(cfg['datagen']['splits'][SPLIT]['episodes']) // 2
    seeds = episode_seeds(cfg, stage1, args.floor, n)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = runs_dir() / f'{stamp}_s2_walkrec_{args.floor}'
    run.mkdir(parents=True)
    print(f'运行目录：{run}', flush=True)
    events = Events(run / 'events.jsonl')
    host, port = args.probe.rsplit(':', 1)
    probe = Probe(host, int(port), events)
    info = probe.call('info')
    if info.get('floor') != args.floor:
        raise SystemExit(f'the arena body has floor {info.get("floor")}, not {args.floor}')
    robot = Robot(args.socket or common.default_socket(), run, events)
    done = set()
    meta_path = args.data / 'meta' / f'{SPLIT}.jsonl'
    if meta_path.exists():
        done = {json.loads(x)['seed'] for x in meta_path.read_text().splitlines()}
    try:
        state = robot.wait(lambda s: True, 5.0, 'first robot.state')
        if state['policy'] == 'held':
            robot.request('robot.enable', {'on': True})
        robot.wait(lambda s: s['policy'] in UPRIGHT, 15.0, 'policy takeover')
        deadline = time.monotonic() + 10.0
        while probe.truth()['released'] != 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        rec = Recorder(args.data, robot, probe, cfg, stage1)
        for i, seed in enumerate(seeds):
            if seed in done:
                continue
            t0 = time.monotonic()
            metas, ep = rec.episode(seed)
            ep['wall_s'] = round(time.monotonic() - t0, 1)
            with meta_path.open('a') as f:
                for m in metas:
                    f.write(json.dumps(m) + '\n')
            with (run / 'episodes.jsonl').open('a') as f:
                f.write(json.dumps(ep) + '\n')
            print(f'{i + 1}/{len(seeds)} seed {seed}: {ep["frames"]} frames, '
                  f'walk {ep.get("walk_m")} m, {ep["wall_s"]} s', flush=True)
    finally:
        robot.move(0.0, 0.0, 0.0)
        robot.head('H0')
        time.sleep(0.3)
        robot.close()
        probe.close()
        events.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
