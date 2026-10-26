"""The head ToF as an emergency stop (decision 5A: ToF only stops the robot, it does not map).

    tof = TofReader(('127.0.0.1', 7801), robot, cfg)    # polls {"op": "tof"} at cfg['hz']
    tof.start(); ...; reason = tof.stop_reason()         # None, or why the robot must stop now

The sim body answers {"op": "tof"} on its robot link (third_party/microduck_rl/src/mjlab_microduck/
sim/body_server.py:429) with one 8x8 frame: distance_mm and status per zone, 45 deg square field,
4 m, row 0 the top, column 0 the sensor's left (sim/tof.py). A real robot reads the same frames from
tofd; on the board the brain asks the Mac's sim body directly. Each zone's return is put in the level
trunk frame with the sensor pose robot.state gives at that tick (frames.tof, from the same head FK
upstream's kinematics/src/tof.rs reprojects with) and the trunk height (odom z):

    obstacle zone   status valid, slant range >= self_blind_m (nearer = the robot's own beak or the
                    toy in it; 3P: 0.08, the lab's hold_blind_m 0.12 hid a wall from about 0.19 m
                    on, the sensor being 6.5-8.5 cm ahead of the trunk), height above the floor
                    >= min_height_m (so the floor and the 1 cm tray rim are not obstacles),
                    ahead of the trunk, horizontal distance from the trunk axis <= stop_range_m,
                    and the trunk tilted no more than max_tilt_deg (not while falling)
    stop            obstacle zones in `frames` consecutive frames (15 Hz)

stop_range_m: stopping from 0.3 m/s slides 3.9-4.9 cm (stage0_facts stop_from_0.3mps_mac) plus the
toes and margin; plan 3D starts at 0.22 m and 3D measures it.
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time

import numpy as np

from microduck_toy.control import ranging

ROWS = COLS = 8
FOV_DEG = 45.0
STATUS_VALID = 5


def beams() -> np.ndarray:
    """Unit beam directions in the sensor frame (+x forward, +y left, +z up), row-major,
    as kinematics/src/tof.rs Reprojector::new builds them."""
    half = math.radians(FOV_DEG / 2 - FOV_DEG / COLS / 2)
    step = 2 * half / (COLS - 1)
    out = np.zeros((ROWS * COLS, 3))
    for i in range(ROWS * COLS):
        el = half - (i // COLS) * step
        az = half - (i % COLS) * step
        out[i] = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    return out


BEAMS = beams()


def points(frame: dict, state: dict) -> list[dict]:
    """Every valid zone's return in the level trunk frame: x forward, y left, z above the floor."""
    pose = state['frames']['tof']
    level = ranging.level_matrix(state)
    rot = level @ ranging.quat_matrix(pose['quat'])
    origin = level @ np.asarray(pose['pos'], float)
    trunk_z = float(state['odom']['position'][2])
    out = []
    for i, (d_mm, st) in enumerate(zip(frame['distance_mm'], frame['status'])):
        if st != STATUS_VALID:
            continue
        d = d_mm / 1000.0
        p = origin + rot @ (BEAMS[i] * d)
        out.append({'zone': i, 'slant_m': d, 'x': float(p[0]), 'y': float(p[1]),
                    'z': float(p[2] + trunk_z), 'range_m': float(math.hypot(p[0], p[1]))})
    return out


def tilt_deg(state: dict) -> float:
    """The trunk's tilt from upright (roll and pitch together), degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(ranging.level_matrix(state)[2, 2])))))


def obstacles(frame: dict, state: dict, cfg: dict) -> list[dict]:
    """Obstacle zones ahead of the trunk. None while the trunk is tilted past max_tilt_deg: a robot
    going over sees the floor and itself at odd angles (cloud 3D stopped mid-fall on a return
    0.12 m behind the trunk, 0.26 m up)."""
    if tilt_deg(state) > float(cfg['max_tilt_deg']):
        return []
    return [p for p in points(frame, state)
            if p['slant_m'] >= cfg['self_blind_m'] and p['z'] >= cfg['min_height_m']
            and p['x'] > 0.0 and p['range_m'] <= cfg['stop_range_m']]


class TofReader(threading.Thread):
    """Polls the sim body's ToF at cfg['hz'] and keeps the stop verdict of the latest frames."""

    def __init__(self, address: tuple[str, int], robot, cfg: dict, log=None):
        super().__init__(name='tof', daemon=True)
        self.address, self.robot, self.cfg, self.log = address, robot, cfg, log
        self.lock = threading.Lock()
        self.streak = 0
        self.last: dict | None = None
        self.frames = 0
        self.error: str | None = None
        self.stopped = threading.Event()
        self.armed = False

    def run(self) -> None:
        period = 1.0 / float(self.cfg['hz'])
        try:
            with socket.create_connection(self.address, timeout=2.0) as sock:
                stream = sock.makefile('rwb')
                while not self.stopped.is_set():
                    t0 = time.monotonic()
                    stream.write(b'{"op": "tof"}\n')
                    stream.flush()
                    frame = json.loads(stream.readline())
                    if 'error' in frame:
                        raise RuntimeError(frame['error'])
                    try:
                        state = self.robot.snapshot()
                    except TimeoutError:
                        state = None
                    hits = obstacles(frame, state, self.cfg) if state else []
                    with self.lock:
                        self.frames += 1
                        self.streak = self.streak + 1 if hits else 0
                        self.last = {'mono': t0, 'hits': hits[:6], 'n_hits': len(hits),
                                     'streak': self.streak}
                    self.stopped.wait(max(0.0, period - (time.monotonic() - t0)))
        except Exception as error:
            with self.lock:
                self.error = f'{type(error).__name__}: {error}'
            if self.log is not None:
                self.log('tof_error', error=self.error)

    def arm(self) -> None:
        """Count only frames from now on (a stop is judged on the walk that is starting)."""
        with self.lock:
            self.streak = 0
            self.armed = True

    def stop_reason(self) -> dict | None:
        with self.lock:
            if self.last is not None and self.streak >= int(self.cfg['frames']):
                hit = min(self.last['hits'], key=lambda p: p['range_m'])
                return {'range_m': round(hit['range_m'], 3), 'z_m': round(hit['z'], 3),
                        'point': [round(hit['x'], 3), round(hit['y'], 3)],
                        'zones': self.last['n_hits'], 'streak': self.streak}
            return None

    def close(self) -> None:
        self.stopped.set()
        self.join(1.0)
