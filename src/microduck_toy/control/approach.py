"""The approach (plan 1E): B segment at H1, then C segment at H2 asking InchingController.decide.

    eyes = Eyes(next_boxes, ranging.CameraModel.from_config(stage1['camera']), facts['ranging'])
    b = b_segment(robot, eyes, toy, plane_z, stage1)          # turn to the aim point, walk closer
    if b['end'] == 'ready':
        c = c_segment(robot, eyes, toy, plane_z, ctl, stage1)  # 'pick', 'retreat', 'abort'

Only robot.state and camera boxes are used, so this runs unchanged against robotd on the board;
`next_boxes(after_seq) -> (seq, boxes)` is the camera (the arena body's oracle boxes in Stage 1,
YOLO later), boxes as [{'name', 'xyxy', 'cut'}] in the upright image.

B segment (look and move, standing between moves): stand settle_s, take `frames` frames at H1,
range the toy (1D correction table for H1), aim at a point aim_right_of_toy_m right of it. If the
aim is more than turn_tol_deg off, turn in place towards it (wz +-turn_wz_radps until robotd's
odometry yaw has turned the angle less turn_lead_deg); else, if the toy is further than c_start_m,
walk towards it with the heading PI (stage0_facts walking.straight_1m_heading_pi: kp 1, ki 0.25)
on the odometry yaw, stopping walk_slide_m early for the coast. Look again after every move.
C segment: stand settle_s, take `frames` frames at H2 (1D correction table for H2), hand a Look to
decide(), send the pulse it names (move for its seconds, then zero), until it says pick, retreat
or abort.
"""
from __future__ import annotations

import bisect
import math
import time

from microduck_toy.control import ranging
from microduck_toy.control.inching import ABORT, PICK, RETREAT, Look


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class Correction:
    """1D's table: ranged - truth (cm) by ranged forward distance; subtracted, interpolated."""

    def __init__(self, table_cm: dict):
        self.keys = sorted(float(k) for k in table_cm['forward'])
        self.fwd = [table_cm['forward'][f'{k:.2f}'] / 100 for k in self.keys]
        self.left = [table_cm['left'][f'{k:.2f}'] / 100 for k in self.keys]

    def _at(self, values: list[float], x: float) -> float:
        if x <= self.keys[0]:
            return values[0]
        if x >= self.keys[-1]:
            return values[-1]
        i = bisect.bisect_right(self.keys, x)
        x0, x1 = self.keys[i - 1], self.keys[i]
        w = (x - x0) / (x1 - x0)
        return values[i - 1] * (1 - w) + values[i] * w

    def apply(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return x - self._at(self.fwd, x), y - self._at(self.left, x)


class Eyes:
    def __init__(self, next_boxes, cam: ranging.CameraModel, ranging_facts: dict):
        self.next_boxes = next_boxes
        self.cam = cam
        self.corrections = {head: Correction(ranging_facts[head]['correction_table_cm'])
                            for head in ('H1', 'H2')}
        self.seq = -1

    def look(self, robot, head: str, toy: str, plane_z: float, frames: int) -> dict:
        """`frames` new frames; the toy's corrected (forward, left) averaged over those that saw it."""
        self.seq, _ = self.next_boxes(-1)          # only frames rendered after this call
        points, raw = [], []
        for _ in range(frames):
            self.seq, boxes = self.next_boxes(self.seq)
            state = robot.snapshot()
            box = next((b for b in boxes if b['name'] == toy), None)
            if box is None:
                continue
            hit = ranging.ground_point(self.cam, state, box['xyxy'], plane_z)
            if hit is None:
                continue
            raw.append(hit)
            points.append(self.corrections[head].apply(hit))
        toy_at = None
        if points:
            toy_at = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
        return {'toy': toy_at, 'seen': len(points), 'frames': frames, 'raw': raw}


def odom_of(state: dict) -> tuple[float, float, float]:
    o = state['odom']
    return float(o['position'][0]), float(o['position'][1]), float(o['yaw'])


def send_pulse(robot, pulse) -> None:
    robot.move(pulse.vx_mps, 0.0, pulse.wz_radps)
    time.sleep(pulse.seconds)
    robot.move(0.0, 0.0, 0.0)


def turn(robot, angle: float, cfg: dict) -> dict:
    """Turn in place by about `angle` (rad, left > 0) on the odometry yaw."""
    lead = math.radians(cfg['turn_lead_deg'])
    goal = max(0.0, abs(angle) - lead)
    _, _, yaw0 = odom_of(robot.snapshot())
    wz = math.copysign(cfg['turn_wz_radps'], angle)
    robot.move(0.0, 0.0, wz)
    timeout = goal / max(0.3, abs(wz) * 0.6) + 1.0
    try:
        robot.wait(lambda s: abs(wrap(odom_of(s)[2] - yaw0)) >= goal, timeout, 'turn')
        reached = True
    except TimeoutError:
        reached = False
    robot.move(0.0, 0.0, 0.0)
    return {'move': 'turn', 'angle_deg': math.degrees(angle), 'wz': wz, 'reached': reached,
            'turned_deg': math.degrees(wrap(odom_of(robot.snapshot())[2] - yaw0))}


def walk(robot, distance: float, bearing: float, cfg: dict) -> dict:
    """Walk about `distance` m towards `bearing` (rad, from the current heading) with the
    heading PI on the odometry yaw; stop walk_slide_m early, robotd coasts the rest."""
    x0, y0, yaw0 = odom_of(robot.snapshot())
    ref = yaw0 + bearing
    ux, uy = math.cos(ref), math.sin(ref)
    goal = max(0.0, distance - cfg['walk_slide_m'])
    kp, ki, cap = cfg['heading_kp'], cfg['heading_ki'], cfg['heading_max_radps']
    integral, last_t, travelled = 0.0, None, 0.0
    deadline = time.monotonic() + goal / 0.08 + 2.0
    while travelled < goal and time.monotonic() < deadline:
        state = robot.next_states(1)[0]
        x, y, yaw = odom_of(state)
        travelled = (x - x0) * ux + (y - y0) * uy
        t = float(state['t'])
        dt = 0.02 if last_t is None else max(1e-3, t - last_t)
        last_t = t
        err = wrap(ref - yaw)
        integral = max(-cap / max(ki, 1e-9), min(cap / max(ki, 1e-9), integral + err * dt))
        wz = max(-cap, min(cap, kp * err + ki * integral))
        robot.move(cfg['walk_vx_mps'], 0.0, wz)
    robot.move(0.0, 0.0, 0.0)
    return {'move': 'walk', 'distance_m': distance, 'bearing_deg': math.degrees(bearing),
            'commanded_until_m': travelled, 'reached': travelled >= goal}


def b_segment(robot, eyes: Eyes, toy: str, plane_z: float, stage1: dict, aim: float | None = None) -> dict:
    cfg, inch = stage1['approach'], stage1['inching']
    aim = inch['aim_right_of_toy_m'] if aim is None else aim
    robot.head(cfg['b_head'])
    steps = []
    for _ in range(int(cfg['b_max_steps'])):
        time.sleep(inch['settle_s'])
        seen = eyes.look(robot, cfg['b_head'], toy, plane_z, int(inch['frames']))
        step = {'toy': seen['toy'], 'seen': seen['seen'], 'odom': odom_of(robot.snapshot())}
        steps.append(step)
        if seen['toy'] is None:
            return {'end': 'unseen', 'steps': steps}
        x, y = seen['toy']
        bearing = math.atan2(y - aim, x)
        step['bearing_deg'] = math.degrees(bearing)
        if abs(bearing) > math.radians(cfg['turn_tol_deg']):
            step.update(turn(robot, bearing, cfg))
        elif x > cfg['c_start_m'] + cfg['walk_min_m']:
            step.update(walk(robot, x - cfg['c_start_m'], bearing, cfg))
        else:
            robot.head(cfg['c_head'])
            return {'end': 'ready', 'steps': steps}
    robot.head(cfg['c_head'])
    return {'end': 'b_budget', 'steps': steps}


def c_segment(robot, eyes: Eyes, toy: str, plane_z: float, ctl, stage1: dict) -> dict:
    cfg, inch = stage1['approach'], stage1['inching']
    robot.head(cfg['c_head'])
    ctl.reset()
    t0 = time.monotonic()
    falls0 = len(robot.falls)
    steps = []
    for _ in range(int(inch['sim']['max_decisions'])):
        time.sleep(inch['settle_s'])
        fallen = len(robot.falls) > falls0 or bool(robot.snapshot()['safety'].get('fallen'))
        if fallen:
            seen = {'toy': None, 'seen': 0, 'frames': int(inch['frames'])}
        else:
            seen = eyes.look(robot, cfg['c_head'], toy, plane_z, int(inch['frames']))
        look = Look(toy=seen['toy'], seen=seen['seen'], frames=seen['frames'],
                    elapsed_s=round(time.monotonic() - t0, 3), fallen=fallen)
        action = ctl.decide(look)
        steps.append({'toy': look.toy, 'seen': look.seen, 'elapsed_s': look.elapsed_s,
                      'fallen': fallen, 'action': action, 'odom': odom_of(robot.snapshot())})
        if action in (PICK, RETREAT, ABORT):
            return {'end': action, 'steps': steps,
                    'pulses': sum(s['action'] in ctl.pulses for s in steps)}
        if action not in ctl.pulses:
            raise ValueError(f'decide() returned {action!r}, not one of {sorted(ctl.actions())}')
        send_pulse(robot, ctl.pulses[action])
    return {'end': 'runaway', 'steps': steps, 'pulses': sum(s['action'] in ctl.pulses for s in steps)}
