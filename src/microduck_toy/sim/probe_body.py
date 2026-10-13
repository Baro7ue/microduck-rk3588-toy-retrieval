"""Stage 0 sim body: bootstrap_body.py plus a ground-truth log and a test port.

    cd third_party/microduck_rl
    .venv/bin/python <repo>/src/microduck_toy/sim/probe_body.py --probe-port 7811 \\
        --ducks 1 --host 127.0.0.1 --port 7801 --keyframe HOME --headless

Every argument except --probe-host/--probe-port goes to the upstream body_server unchanged.
After every 20 ms batch (one robotd tick) the body's true state is computed: trunk pose and
velocity, mouth_tip, head camera, foot and non-foot floor contacts. The test port answers one
JSON object per line:

    {"op": "info"}                                  references (STAND keyframe heights, columns)
    {"op": "truth"}                                 the latest row
    {"op": "record", "path": ".../truth.csv"}       start a CSV (path null stops it)
    {"op": "mark", "label": "push_front_1"}         label the next CSV row
    {"op": "push", "vx": 0.8, "vy": 0.0}            add trunk velocity, heading frame, m/s
    {"op": "pose", "kind": "stand", "x": 0, "y": 0, "yaw": 0, "hold_s": 1.0}
    {"op": "pose", "kind": "back" | "front" | "side", "tilt_deg": 95, "fold": 0.0, "hold_s": 0}
    {"op": "object", "name": "toy_a", "x": 0.3, "y": 0.0, "z": 0.02, "yaw": 0.0}

The truth is for scoring only: nothing here reaches robotd, which still sees only the
upstream sensor protocol. Upstream repositories stay unchanged; classes are swapped at runtime
exactly as bootstrap_body.py does.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import queue
import socketserver
import sys
import threading
import time

import mujoco
import numpy as np

from mjlab_microduck.sim import body_server as sim

# Run as a script by the sim's Python: make the repository's src/ importable whatever PYTHONPATH says.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from microduck_toy.sim import bootstrap_body  # noqa: E402
from microduck_toy.stage0.metrics import COLUMNS  # noqa: E402

# Fallbacks if a custom scene has no STAND keyframe; measured from scene.xml at cb70b79.
STAND_TRUNK_Z = 0.12
STAND_JAW_Z = 0.2326
APPLY_TIMEOUT_S = 2.0


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def lying_quat(kind: str, tilt: float, yaw: float, side: float) -> np.ndarray:
    """Yaw, then the lab's tilt (getup.py _getup_spawn): back = pitch -tilt, front = +tilt."""
    pitch, roll = {'back': (-tilt, 0.0), 'front': (tilt, 0.0), 'side': (0.0, side * tilt)}[kind]
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return quat_mul(yaw_quat(yaw), np.array([cp * cr, cp * sr, sp * cr, sp * sr]))


def quat_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class ProbeWorld(bootstrap_body.StartupWorld):
    """The startup world, plus truth after every batch and ops applied between batches."""

    probe_address = ('127.0.0.1', 7811)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops: queue.Queue = queue.Queue()
        self.rec_lock = threading.Lock()
        self.csv_file = self.csv_writer = None
        self.marks: list[str] = []
        self.last_row: dict | None = None
        self.hold_until: float | None = None
        self.ids = None
        self.refs = self._stand_refs()
        self.server = ProbeServer(self.probe_address, ProbeHandler)
        self.server.world = self
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        host, port = self.server.server_address[:2]
        print(f'== probe: truth and test port on {host}:{port}', flush=True)

    # ── references and ids ────────────────────────────────────────────────

    def _stand_refs(self) -> dict:
        model = self.model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        refs = {'stand_trunk_z': STAND_TRUNK_Z, 'stand_jaw_z': STAND_JAW_Z, 'source': 'fallback'}
        if 'STAND' in names:
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, names.index('STAND'))
            mujoco.mj_forward(model, data)
            refs = {'stand_trunk_z': float(data.qpos[2]),
                    'stand_jaw_z': float(data.xpos[model.body('jaw_soft').id][2]),
                    'source': 'STAND keyframe'}
        return refs

    def _resolve(self) -> None:
        model = self.model
        body = self.bodies[0]
        trunk = model.body(body.prefix + 'trunk_base').id
        robot = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == trunk}

        def foot(name):
            foot_body = model.geom_bodyid[model.geom(body.prefix + name).id]
            return {g for g in robot if model.geom_bodyid[g] == foot_body}

        self.ids = {
            'trunk': trunk,
            'robot': robot,
            'left': foot('left_foot_collision'),
            'right': foot('right_foot_collision'),
            'floor': model.geom('floor').id,
            'tip': model.site(body.prefix + 'mouth_tip').id,
            'jaw': model.body(body.prefix + 'jaw_soft').id,
            # The head_camera *site*: its +x is the view direction. The MJCF camera of the same
            # name faces backwards and sim/camera.py turns it only when frames are served.
            'cam': model.site(body.prefix + 'head_camera').id,
            'neck': body.qpos_adr[body.to_wire.index(sim.JOINT_NAMES.index('neck_pitch'))],
            'head': body.qpos_adr[body.to_wire.index(sim.JOINT_NAMES.index('head_pitch'))],
        }

    # ── stepping ──────────────────────────────────────────────────────────

    def step(self, times: int = 1) -> None:
        if self.bodies and self.ids is None:
            self._resolve()
        self._apply_ops()
        super().step(times)
        if self.ids is not None:
            self._record()

    def _apply_ops(self) -> None:
        while True:
            try:
                op, params, done, reply = self.ops.get_nowait()
            except queue.Empty:
                break
            try:
                with self.lock:
                    reply.update(getattr(self, f'_op_{op}')(**params))
                    reply['t'] = float(self.data.time)
            except Exception as error:  # a bad request must not stop the physics
                reply['error'] = f'{type(error).__name__}: {error}'
            done.set()
        if self.hold_until is not None and self.bodies:
            with self.lock:
                if self.data.time >= self.hold_until:
                    body = self.bodies[0]
                    # Same rule as StartupBody.set_torque: never release before startup support ends.
                    body.released = bool(getattr(body, 'support_ended', True))
                    self.hold_until = None

    def submit(self, op: str, params: dict) -> dict:
        if self.ids is None:
            raise RuntimeError('body not ready yet')
        done, reply = threading.Event(), {}
        self.ops.put((op, params, done, reply))
        if not done.wait(APPLY_TIMEOUT_S):
            raise TimeoutError(f'{op}: physics did not apply it within {APPLY_TIMEOUT_S} s')
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    # ── ops (called with the world lock held, between batches) ────────────

    def _op_push(self, vx: float = 0.0, vy: float = 0.0) -> dict:
        body, data = self.bodies[0], self.data
        yaw = quat_yaw(data.qpos[body.trunk + 3: body.trunk + 7])
        c, s = math.cos(yaw), math.sin(yaw)
        data.qvel[body.trunk_dof + 0] += c * vx - s * vy
        data.qvel[body.trunk_dof + 1] += s * vx + c * vy
        return {'pushed': [vx, vy], 'yaw': yaw}

    def _op_pose(self, kind: str = 'stand', x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 hold_s: float | None = None, tilt_deg: float = 95.0, fold: float = 0.0,
                 side: float = 1.0) -> dict:
        body, data, model = self.bodies[0], self.data, self.model
        if kind not in ('stand', 'back', 'front', 'side'):
            raise ValueError(f'unknown pose {kind!r}')
        for slot, wire in enumerate(body.to_wire):
            data.qpos[body.qpos_adr[slot]] = sim.HOME_POSE[wire]
        if kind == 'stand':
            quat, hold = yaw_quat(yaw), 1.0 if hold_s is None else hold_s
        else:
            quat = lying_quat(kind, math.radians(tilt_deg), yaw, 1.0 if side >= 0 else -1.0)
            hold = 0.0 if hold_s is None else hold_s
            # Legs between straight and folded, as getup.py spawns them (left +, right -).
            for name, sign, amount in (('left_hip_pitch', 1, 1.1), ('right_hip_pitch', -1, 1.1),
                                       ('left_knee', 1, 1.2), ('right_knee', -1, 1.2)):
                slot = body.to_wire.index(sim.JOINT_NAMES.index(name))
                data.qpos[body.qpos_adr[slot]] += sign * amount * fold
        data.qpos[body.trunk: body.trunk + 3] = [x, y, sim.HOME_TRUNK_Z]
        data.qpos[body.trunk + 3: body.trunk + 7] = quat
        data.qvel[body.trunk_dof: body.trunk_dof + 6] = 0.0
        data.qvel[body.qvel_adr] = 0.0
        if kind != 'stand':
            # Lower the robot until its lowest geom is 5 mm above the floor.
            mujoco.mj_forward(model, data)
            data.qpos[body.trunk + 2] += 0.005 - self._lowest_point()
        mujoco.mj_forward(model, data)
        body.remember()
        body.released = False
        self.hold_until = data.time + max(0.0, float(hold))
        return {'pose': kind, 'trunk_z': float(data.qpos[body.trunk + 2]), 'hold_s': hold}

    def _lowest_point(self) -> float:
        model, data = self.model, self.data
        lowest = math.inf
        for g in self.ids['robot']:
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            center = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g, :3]
            half = np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_aabb[g, 3:]
            lowest = min(lowest, float(center[2] - half))
        return lowest

    def _op_object(self, name: str, x: float, y: float, z: float, yaw: float = 0.0) -> dict:
        model, data = self.model, self.data
        bid = model.body(name).id
        joint = model.body_jntadr[bid]
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f'{name!r} has no free joint')
        qadr, vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        data.qpos[qadr: qadr + 3] = [x, y, z]
        data.qpos[qadr + 3: qadr + 7] = yaw_quat(yaw)
        data.qvel[vadr: vadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return {'object': name}

    # ── truth ─────────────────────────────────────────────────────────────

    def _record(self) -> None:
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
        floor = ids['floor']
        touching = {b if a == floor else a for a, b in pairs if floor in (a, b)}
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        gravity = -rot.T[:, 2]
        row = {
            't': sim_time, 'mono': time.monotonic(),
            'x': pos[0], 'y': pos[1], 'z': pos[2], 'yaw': quat_yaw(quat),
            'gx': gravity[0], 'gy': gravity[1], 'gz': gravity[2],
            'vx': lin[0], 'vy': lin[1], 'vz': lin[2], 'wz': float((rot @ ang_local)[2]),
            'tip_x': tip[0], 'tip_y': tip[1], 'tip_z': tip[2], 'jaw_z': jaw_z,
            'cam_x': cam[0], 'cam_y': cam[1], 'cam_z': cam[2],
            'cam_pitch': math.asin(max(-1.0, min(1.0, -float(forward[2])))),
            'neck_q': neck, 'head_q': head,
            'lfoot': int(bool(touching & ids['left'])),
            'rfoot': int(bool(touching & ids['right'])),
            'other_floor': len((touching & ids['robot']) - ids['left'] - ids['right']),
            'torque': int(bool(torque)), 'kp': float(kp), 'released': int(bool(released)),
        }
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
            meta = {**self.info(), 'csv': str(target)}
            target.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
            return {'recording': str(target)}

    def info(self) -> dict:
        return {'columns': list(COLUMNS), 'refs': self.refs,
                'timestep': float(self.model.opt.timestep),
                'probe': list(self.server.server_address[:2])}


class ProbeHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        world: ProbeWorld = self.server.world
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
                elif op in ('push', 'pose', 'object'):
                    answer = world.submit(op, request)
                else:
                    raise ValueError(f'unknown op {op!r}')
            except Exception as error:
                answer = {'error': f'{type(error).__name__}: {error}'}
            self.wfile.write((json.dumps(answer) + '\n').encode())
            self.wfile.flush()


class ProbeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-host', default='127.0.0.1')
    parser.add_argument('--probe-port', type=int, default=7811)
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if '--ducks' in rest and rest[rest.index('--ducks') + 1] != '1':
        raise SystemExit('probe_body.py records one duck; use --ducks 1')
    ProbeWorld.probe_address = (own.probe_host, own.probe_port)
    # bootstrap_body.main() would install StartupWorld; install its Body and this World instead.
    sim.Body = bootstrap_body.StartupBody
    sim.World = ProbeWorld
    print('== simulation startup assistance enabled; count balancing only after support ends',
          flush=True)
    sim.main()


if __name__ == '__main__':
    main()
