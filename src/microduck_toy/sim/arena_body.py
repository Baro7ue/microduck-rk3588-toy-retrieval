"""Stage 1 sim body: probe_body.py (truth log + test port) in the Stage 1 field, with a beak.

    cd third_party/microduck_rl
    .venv/bin/python <repo>/src/microduck_toy/sim/arena_body.py --floor grid --probe-port 7811 \\
        --ducks 1 --host 127.0.0.1 --port 7801 --keyframe HOME --headless [--cameras a]

Every argument except --floor, --stage1-config, --probe-host and --probe-port goes to the upstream
body_server unchanged. What this adds to probe_body.py:

* The field (scene/arena.py): walls, tray, the five toys, parked outside until a script places them.
* The mouth. robotd sends 15 joint targets and the upstream body drops the mouth (index 9). Here
  the mouth target is kept: when it goes from open to closed (open fraction across 0.5) the nearest
  toy within 4 cm (horizontal) and 4.5 cm (height) of the beak tip is welded to jaw_soft with
  probability 1 - (e / 4 cm)^2 (microduck-lab world/arena.py:1573-1631, Apache-2.0, re-written).
  It is let go when the mouth opens or the robot falls. While a toy is held, the mouth angle sent
  back to robotd is the angle the toy blocks it at, so code can check a grasp from robot.state
  joints the way it would read a stalled servo. Every close is logged (op "grasps").
* The oracle detector (with --cameras a): every rendered head-camera frame also gets a segmentation
  render; each toy's and the tray's box in the upright 360x640 image is kept with the frame number
  (op "view"). Boxes only, as a detector gives them; the truth beside them is for scoring.

Test port ops added to probe_body.py's (info, truth, record, mark, push, pose, object):

    {"op": "toy", "name": "bus", "x": 0.3, "y": 0.0, "yaw": 0.0}     place a toy at rest
    {"op": "park", "names": null}                                     park toys (null = all)
    {"op": "layout", "robot": [x, y, yaw], "toys": {"bus": [x, y, yaw], ...}, "hold_s": 1.0}
    {"op": "grasps", "since": 0}                                      closes logged since index
    {"op": "grasp_seed", "seed": 7}                                   reseed the grasp draw
    {"op": "release"}                                                 let go of a held toy
    {"op": "view", "after": -1, "rgb": false, "timeout": 1.0}         next oracle frame after seq
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from pathlib import Path
import sys
import threading
import time

import mujoco
import numpy as np

from mjlab_microduck.sim import body_server as sim
from mjlab_microduck.sim import camera as sim_camera

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from microduck_toy.paths import repo_root  # noqa: E402
from microduck_toy.scene import arena  # noqa: E402
from microduck_toy.scene import render as scene_render  # noqa: E402
from microduck_toy.scene import toys as toy_lib  # noqa: E402
from microduck_toy.sim import bootstrap_body, probe_body  # noqa: E402
from microduck_toy.stage0 import metrics  # noqa: E402

MOUTH_CLOSED = math.radians(-5.0)   # duck-control/src/model.rs:62-63
MOUTH_OPEN = math.radians(30.0)
PORTRAIT_K = 3                      # np.rot90 turns from the raw frame to the upright one

TOY_COLUMNS = [f'{toy_lib.SHORT[n]}_{k}' for n in toy_lib.SHORT for k in ('x', 'y', 'z', 'yaw')]
COLUMNS = list(metrics.COLUMNS) + ['mouth_open', 'mouth_q', 'holding'] + TOY_COLUMNS


def load_config(path: str | None = None) -> dict:
    target = Path(path) if path else repo_root() / 'configs' / 'control' / 'stage1.json'
    return json.loads(target.read_text())


def open_fraction(target: float) -> float:
    return (target - MOUTH_CLOSED) / (MOUTH_OPEN - MOUTH_CLOSED)


class ArenaBody(bootstrap_body.StartupBody):
    """The startup body, plus the mouth slot the upstream body drops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouth_target = MOUTH_CLOSED
        self.mouth_reported = MOUTH_CLOSED

    def set_targets(self, wire_targets: list[float]) -> None:
        super().set_targets(wire_targets)
        with self.world.lock:
            self.mouth_target = float(wire_targets[sim.MOUTH_INDEX])

    def sensors(self) -> dict:
        reading = super().sensors()
        with self.world.lock:
            reading['positions'][sim.MOUTH_INDEX] = self.mouth_reported
        return reading


class ArenaCamera(sim_camera.Camera):
    """The head camera, plus a segmentation render of the same scene for the oracle boxes.

    The upstream loop calls render() 15 times a second on the physics thread. `fps` thins that out:
    15 renders every call (the upstream rate), 0 renders only when a "view" op is waiting. A
    software renderer (OSMesa in the cloud) takes about half a second per frame, during which the
    physics does not step; on the Mac a frame is about 12 ms.
    """

    fps = 15.0

    def __init__(self, model, name, *args, **kwargs):
        super().__init__(model, name, *args, **kwargs)
        self.seg = mujoco.Renderer(model, height=self.height, width=self.width)
        self.seg.enable_segmentation_rendering()
        self.seq = -1
        self.view = None
        self.cond = threading.Condition()
        self.wanted = False
        self.last = -math.inf

    def render(self, world) -> None:
        now = time.monotonic()
        with self.cond:
            due = self.wanted or (self.fps > 0 and now - self.last >= 1.0 / self.fps - 1e-3)
        if not due:
            return
        self.last = now
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera)
            self.seg.update_scene(world.data, camera=self.camera)
            truth = world.view_truth()
            truth['camera'] = self.pose_truth(world.data)
        rgb = self.renderer.render()
        seg = self.seg.render()
        boxes = world.oracle_boxes(seg)
        packed = sim_camera.to_uyvy(rgb)
        with self.lock:
            self.latest = packed
        with self.cond:
            self.seq += 1
            self.wanted = False
            self.view = {'seq': self.seq, 'rgb': rgb, 'boxes': boxes, 'truth': truth}
            self.cond.notify_all()

    def pose_truth(self, data) -> dict:
        """The camera at render time: world position, and the upright-image row of the horizon
        (the same formula robot.state gives the Stage 2 window), for checking the robot's own."""
        xmat = data.cam_xmat[self.camera].reshape(3, 3)
        rot = np.stack([xmat[:, 1], xmat[:, 0], -xmat[:, 2]], axis=1)  # upright OpenCV -> world
        ahead = rot[:, 2].copy()
        ahead[2] = 0.0
        horizon = None
        if np.linalg.norm(ahead) > 1e-6:
            v = rot.T @ (ahead / np.linalg.norm(ahead))
            if v[2] > 1e-6:
                horizon = 320.0 + 180.0 / math.tan(math.radians(22.5)) * v[1] / v[2]
        return {'pos': [float(x) for x in data.cam_xpos[self.camera]],
                'horizon_row': None if horizon is None else round(float(horizon), 2)}

    def next_view(self, after: int, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        with self.cond:
            while self.view is None or self.view['seq'] <= after:
                self.wanted = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)
            return self.view


class ArenaWorld(probe_body.ProbeWorld):
    floor = 'grid'
    config: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Connections arrive only after startup, so the handler can be swapped now.
        self.server.RequestHandlerClass = ArenaHandler
        self.arena = arena.ArenaIds(self.model)
        grasp = self.config['grasp']
        self.tol_xy, self.tol_z = float(grasp['tol_xy_m']), float(grasp['tol_z_m'])
        self.open_threshold = float(grasp['open_threshold'])
        self.blocked_per_m = math.radians(float(grasp['blocked_deg_per_cm'])) * 100.0
        self.rng = np.random.default_rng(int(grasp['seed']))
        self.holding: str | None = None
        self.was_open = False
        self.down_since: float | None = None
        self.grasp_log: list[dict] = []
        cam = self.model.camera('head_camera').id
        self.fovy = float(self.model.cam_fovy[cam])
        self.geom_owner = scene_render.owners(self.model, self.arena)

    # ── the beak ──────────────────────────────────────────────────────────

    def step(self, times: int = 1) -> None:
        super().step(times)
        if not self.bodies or self.ids is None:
            return
        body = self.bodies[0]
        with self.lock:
            is_open = open_fraction(body.mouth_target) > self.open_threshold
            if self.was_open and not is_open:
                self._close_beak()
            elif is_open and self.holding is not None:
                self._release('opened')
            self.was_open = is_open
            row = self.last_row
            if row is not None and metrics.down(row):
                self.down_since = row['t'] if self.down_since is None else self.down_since
                if self.holding is not None and row['t'] - self.down_since >= metrics.FALL_HOLD_S:
                    self._release('fell')
            else:
                self.down_since = None
            body.mouth_reported = self._reported_mouth(body.mouth_target)

    def _reported_mouth(self, target: float) -> float:
        if self.holding is None:
            return target
        half = self.arena.toys[self.holding]['half']
        return max(target, MOUTH_CLOSED + self.blocked_per_m * 2 * min(half[1], half[2]))

    def _close_beak(self) -> None:
        data = self.data
        tip = data.site_xpos[self.arena.tip].copy()
        entry = {'t': float(data.time), 'mono': time.monotonic(), 'tip': tip.round(5).tolist(),
                 'holding_before': self.holding}
        best, best_err, nearest = None, math.inf, None
        for name, toy in self.arena.toys.items():
            p = data.xpos[toy['body']]
            exy = float(math.hypot(p[0] - tip[0], p[1] - tip[1]))
            ez = float(abs(p[2] - tip[2]))
            if nearest is None or exy < nearest[1]:
                nearest = (name, exy, ez)
            if exy < self.tol_xy and ez < self.tol_z and exy < best_err:
                best, best_err = name, exy
        if nearest is not None:
            entry.update(nearest=toy_lib.SHORT[nearest[0]], nearest_exy=nearest[1],
                         nearest_ez=nearest[2])
        if self.holding is not None:
            entry.update(result='already_holding')
        elif best is None:
            entry.update(result='out_of_reach')
        else:
            p_ok = 1.0 - (best_err / self.tol_xy) ** 2
            draw = float(self.rng.random())
            entry.update(candidate=toy_lib.SHORT[best], e_xy=best_err, p=p_ok, draw=draw)
            if draw <= p_ok:
                self._weld(best)
                entry['result'] = 'grasped'
            else:
                entry['result'] = 'missed'
        self.grasp_log.append(entry)
        print('== beak closed: ' + json.dumps({k: entry.get(k) for k in
              ('result', 'nearest', 'nearest_exy', 'nearest_ez', 'p')}), flush=True)

    def _weld(self, name: str) -> None:
        model, data = self.model, self.data
        toy = self.arena.toys[name]
        jaw, b = self.arena.jaw, toy['body']
        rot_jaw = data.xmat[jaw].reshape(3, 3)
        rel_p = rot_jaw.T @ (data.xpos[b] - data.xpos[jaw])
        qj, qb = data.xquat[jaw], data.xquat[b]
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, np.array([qj[0], -qj[1], -qj[2], -qj[3]]), qb)
        eq = toy['eq']
        model.eq_data[eq, 0:3] = 0.0
        model.eq_data[eq, 3:6] = rel_p
        model.eq_data[eq, 6:10] = rel_q
        model.eq_data[eq, 10] = 1.0
        data.eq_active[eq] = 1
        self.holding = name

    def _release(self, why: str) -> None:
        if self.holding is None:
            return
        self.data.eq_active[self.arena.toys[self.holding]['eq']] = 0
        self.grasp_log.append({'t': float(self.data.time), 'mono': time.monotonic(),
                               'result': f'released_{why}', 'toy': toy_lib.SHORT[self.holding]})
        print(f'== released {toy_lib.SHORT[self.holding]} ({why})', flush=True)
        self.holding = None

    # ── ops (world lock held, between batches) ───────────────────────────

    def _op_toy(self, name: str, x: float, y: float, yaw: float = 0.0,
                z: float | None = None) -> dict:
        full = toy_lib.resolve(name)
        if self.holding == full:
            self._release('moved')
        arena.place_toy(self.model, self.data, self.arena, full, x, y, yaw, z)
        mujoco.mj_forward(self.model, self.data)
        return {'toy': toy_lib.SHORT[full]}

    def _op_park(self, names: list[str] | None = None) -> dict:
        chosen = [toy_lib.resolve(n) for n in names] if names else list(self.arena.toys)
        if self.holding in chosen:
            self._release('parked')
        for name in chosen:
            arena.park_toy(self.model, self.data, self.arena, name)
        mujoco.mj_forward(self.model, self.data)
        return {'parked': [toy_lib.SHORT[n] for n in chosen]}

    def _op_pose(self, *args, **kwargs) -> dict:
        self._release('reset')
        return super()._op_pose(*args, **kwargs)

    def _op_layout(self, robot: list[float], toys: dict, hold_s: float = 1.0) -> dict:
        self._op_park()
        for name, (x, y, yaw) in toys.items():
            arena.place_toy(self.model, self.data, self.arena, toy_lib.resolve(name), x, y, yaw)
        x, y, yaw = robot
        return self._op_pose(kind='stand', x=x, y=y, yaw=yaw, hold_s=hold_s)

    def _op_release(self) -> dict:
        held = self.holding
        self._release('asked')
        return {'released': toy_lib.SHORT[held] if held else None}

    # ── oracle boxes ─────────────────────────────────────────────────────

    def oracle_boxes(self, seg: np.ndarray) -> list[dict]:
        """Boxes in the upright image (360 wide x 640 high) from a raw 640x360 segmentation."""
        return scene_render.boxes_from_seg(seg, self.geom_owner)

    def view_truth(self) -> dict:
        """Truth at render time, for scoring only (called with the lock held)."""
        data = self.data
        body = self.bodies[0]
        quat = data.qpos[body.trunk + 3: body.trunk + 7]
        out = {'t': float(data.time),
               'trunk': [float(v) for v in data.qpos[body.trunk: body.trunk + 3]],
               'yaw': probe_body.quat_yaw(quat)}
        for name in self.arena.toys:
            out[toy_lib.SHORT[name]] = list(arena.toy_pose(data, self.arena, name))
        out['tray'] = [float(v) for v in self.arena.tray_centre]
        return out

    # ── truth rows ───────────────────────────────────────────────────────

    def _record(self) -> None:
        """probe_body's truth row (same fields, same rounding) plus the beak and the toys."""
        body, data, ids = self.bodies[0], self.data, self.ids
        with self.lock:
            quat = data.qpos[body.trunk + 3: body.trunk + 7].copy()
            pos = data.qpos[body.trunk: body.trunk + 3].copy()
            lin = data.qvel[body.trunk_dof: body.trunk_dof + 3].copy()
            ang_local = data.qvel[body.trunk_dof + 3: body.trunk_dof + 6].copy()
            tip = data.site_xpos[ids['tip']].copy()
            jaw_z = float(data.xpos[ids['jaw']][2])
            cam = data.site_xpos[ids['cam']].copy()
            forward = data.site_xmat[ids['cam']].reshape(3, 3)[:, 0]
            neck, head = float(data.qpos[ids['neck']]), float(data.qpos[ids['head']])
            sim_time = float(data.time)
            n = int(data.ncon)
            pairs = list(zip(data.contact.geom1[:n].tolist(), data.contact.geom2[:n].tolist()))
            torque, kp, released = body.torque_on, body.kp, body.released
            mouth = (open_fraction(body.mouth_target), body.mouth_reported)
            holding = toy_lib.SHORT[self.holding] if self.holding else ''
            toys = {toy_lib.SHORT[name]: arena.toy_pose(data, self.arena, name)
                    for name in self.arena.toys}
        floor = ids['floor']
        touching = {b if a == floor else a for a, b in pairs if floor in (a, b)}
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        gravity = -rot.T[:, 2]
        row = {
            't': sim_time, 'mono': time.monotonic(),
            'x': pos[0], 'y': pos[1], 'z': pos[2], 'yaw': probe_body.quat_yaw(quat),
            'gx': gravity[0], 'gy': gravity[1], 'gz': gravity[2],
            'vx': lin[0], 'vy': lin[1], 'vz': lin[2], 'wz': float((rot @ ang_local)[2]),
            'tip_x': tip[0], 'tip_y': tip[1], 'tip_z': tip[2], 'jaw_z': jaw_z,
            'cam_x': cam[0], 'cam_y': cam[1], 'cam_z': cam[2],
            'cam_pitch': math.asin(max(-1.0, min(1.0, -float(forward[2])))),
            'neck_q': neck, 'head_q': head,
            'lfoot': int(bool(touching & ids['left'])),
            'rfoot': int(bool(touching & ids['right'])),
            # Toys are not the robot; ids['robot'] is the robot's own geoms only.
            'other_floor': len((touching & ids['robot']) - ids['left'] - ids['right']),
            'torque': int(bool(torque)), 'kp': float(kp), 'released': int(bool(released)),
            'mouth_open': mouth[0], 'mouth_q': mouth[1], 'holding': holding,
        }
        for short, (x, y, z, yaw) in toys.items():
            row.update({f'{short}_x': x, f'{short}_y': y, f'{short}_z': z, f'{short}_yaw': yaw})
        row = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
               for k, v in row.items()}
        with self.rec_lock:
            row['mark'] = '|'.join(self.marks)
            self.marks.clear()
            self.last_row = row
            if self.csv_writer is not None:
                self.csv_writer.writerow(row)
                if int(round(sim_time / 0.02)) % 25 == 0:
                    self.csv_file.flush()

    def start_record(self, path: str | None) -> dict:
        with self.rec_lock:
            if self.csv_file is not None:
                self.csv_file.close()
                self.csv_file = self.csv_writer = None
            if path is None:
                return {'recording': None}
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = target.open('x', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=COLUMNS)
            self.csv_writer.writeheader()
            target.with_suffix('.meta.json').write_text(
                json.dumps({**self.info(), 'csv': str(target)}, indent=2))
            return {'recording': str(target)}

    def info(self) -> dict:
        out = super().info()
        out.update(columns=list(COLUMNS), floor=self.floor, fovy_deg=self.fovy,
                   tray_centre=[float(v) for v in self.arena.tray_centre],
                   toys={toy_lib.SHORT[n]: {'half': list(t['half']), 'rest_z': t['rest_z']}
                         for n, t in self.arena.toys.items()},
                   camera=bool(self.bodies and self.bodies[0].camera is not None))
        return out


class ArenaHandler(probe_body.ProbeHandler):
    def handle(self) -> None:
        world: ArenaWorld = self.server.world
        for raw in self.rfile:
            try:
                request = json.loads(raw)
                op = request.pop('op')
                if op == 'info':
                    answer = world.info()
                elif op == 'truth':
                    answer = {'row': world.last_row}
                elif op == 'record':
                    answer = world.start_record(request.get('path'))
                elif op == 'mark':
                    with world.rec_lock:
                        world.marks.append(str(request['label']))
                    answer = {'marked': request['label']}
                elif op in ('push', 'pose', 'object', 'toy', 'park', 'layout', 'release'):
                    answer = world.submit(op, request)
                elif op == 'grasps':
                    since = int(request.get('since', 0))
                    with world.lock:
                        answer = {'grasps': world.grasp_log[since:], 'count': len(world.grasp_log),
                                  'holding': toy_lib.SHORT[world.holding] if world.holding else None}
                elif op == 'grasp_seed':
                    with world.lock:
                        world.rng = np.random.default_rng(int(request['seed']))
                    answer = {'seed': int(request['seed'])}
                elif op == 'view':
                    answer = view_answer(world, int(request.get('after', -1)),
                                         bool(request.get('rgb', False)),
                                         float(request.get('timeout', 1.0)))
                else:
                    raise ValueError(f'unknown op {op!r}')
            except Exception as error:
                answer = {'error': f'{type(error).__name__}: {error}'}
            self.wfile.write((json.dumps(answer) + '\n').encode())
            self.wfile.flush()


def view_answer(world: ArenaWorld, after: int, rgb: bool, timeout: float) -> dict:
    body = world.bodies[0] if world.bodies else None
    if body is None or body.camera is None:
        raise RuntimeError('no head camera: start the body with --cameras a')
    view = body.camera.next_view(after, timeout)
    if view is None:
        raise TimeoutError(f'no camera frame after {after} within {timeout} s')
    answer = {'seq': view['seq'], 'boxes': view['boxes'], 'truth': view['truth'],
              'size': [view['rgb'].shape[0], view['rgb'].shape[1]]}
    if rgb:
        upright = np.ascontiguousarray(np.rot90(view['rgb'], PORTRAIT_K))
        answer['rgb_b64'] = base64.b64encode(upright.tobytes()).decode()
        answer['rgb_shape'] = list(upright.shape)
    return answer


def install(floor: str, config: dict) -> None:
    """Swap the upstream classes at runtime, as bootstrap_body.py does; no file changes."""
    rl_root = Path(sim.__file__).resolve().parents[3]

    def build_world(scene: Path, count: int) -> mujoco.MjModel:
        if count != 1:
            raise SystemExit('arena_body.py runs one duck; use --ducks 1')
        return arena.build_spec(config['arena'], rl_root, floor).compile()

    ArenaWorld.floor = floor
    ArenaWorld.config = config
    sim.build_world = build_world
    sim.Body = ArenaBody
    sim.World = ArenaWorld
    sim.Camera = ArenaCamera


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-host', default='127.0.0.1')
    parser.add_argument('--probe-port', type=int, default=7811)
    parser.add_argument('--floor', choices=('grid', 'wood'), default='grid')
    parser.add_argument('--stage1-config', default=None)
    parser.add_argument('--oracle-fps', type=float, default=15.0,
                        help='head-camera renders per second; 0 = only when a view op asks')
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if '--scene' in rest:
        raise SystemExit('arena_body.py builds its own scene; drop --scene')
    ArenaWorld.probe_address = (own.probe_host, own.probe_port)
    install(own.floor, load_config(own.stage1_config))
    ArenaCamera.fps = own.oracle_fps
    print(f'== Stage 1 arena ({own.floor} floor); startup assistance enabled', flush=True)
    sim.main()


if __name__ == '__main__':
    main()
