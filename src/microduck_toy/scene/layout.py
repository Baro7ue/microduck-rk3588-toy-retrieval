"""Seeded layouts for the Stage 1 field: the same seed always gives the same layout.

A layout is where the robot starts and three toy slots: one for the target and one for each
distractor. Positions depend only on the seed; the target slot takes whichever target the episode
asks for, so one layout serves all three targets (20 layouts x 3 targets = the 60 formal episodes,
plan 1A). The other two targets stay parked outside the field.

Constraints (configs/control/stage1.json "layout"): toys at least wall_clear_m from the walls,
tray_clear_m from the tray centre, toy_gap_m from each other and start_clear_m from the robot's
start; the robot at least robot_wall_clear_m from the walls and tray_clear_m + half a tray side from
the tray centre. Headings are uniform.
"""
from __future__ import annotations

import math

import numpy as np

from microduck_toy.scene import toys as toy_lib

MAX_TRIES = 10000


def _inside(rng, half: float, clear: float) -> tuple[float, float]:
    lim = half - clear
    return float(rng.uniform(-lim, lim)), float(rng.uniform(-lim, lim))


def _far(p, q, d) -> bool:
    return math.hypot(p[0] - q[0], p[1] - q[1]) >= d


def make_layout(seed: int, layout_cfg: dict, arena_cfg: dict) -> dict:
    rng = np.random.default_rng(int(seed))
    half = float(arena_cfg['half_size_m'])
    tray = tuple(float(v) for v in arena_cfg['tray']['centre_m'])
    tray_side = float(arena_cfg['tray']['size_m'])
    c = {k: float(v) for k, v in layout_cfg.items() if k.endswith('_m')}
    for _ in range(MAX_TRIES):
        robot = _inside(rng, half, c['robot_wall_clear_m'])
        if not _far(robot, tray, c['tray_clear_m'] + tray_side / 2):
            continue
        robot_yaw = float(rng.uniform(-math.pi, math.pi))
        slots = []
        for _slot in range(3):
            for _ in range(MAX_TRIES):
                p = _inside(rng, half, c['wall_clear_m'])
                if (_far(p, tray, c['tray_clear_m']) and _far(p, robot, c['start_clear_m'])
                        and all(_far(p, q, c['toy_gap_m']) for q in slots)):
                    slots.append(p)
                    break
            else:
                break
        if len(slots) == 3:
            yaws = [float(rng.uniform(-math.pi, math.pi)) for _ in slots]
            names = ['target', *(toy_lib.SHORT[n] for n in toy_lib.distractors())]
            r4 = lambda v: round(float(v), 4)  # noqa: E731  (0.1 mm / 0.006 deg)
            return {'seed': int(seed), 'robot': [r4(robot[0]), r4(robot[1]), r4(robot_yaw)],
                    'slots': {name: [r4(p[0]), r4(p[1]), r4(yaw)]
                              for name, p, yaw in zip(names, slots, yaws)}}
    raise RuntimeError(f'seed {seed}: no layout met the constraints in {MAX_TRIES} tries')


def episode(layout: dict, target: str) -> dict:
    """The probe's layout op arguments for one episode: robot pose and the three toys placed."""
    short = toy_lib.SHORT[toy_lib.resolve(target)]
    if toy_lib.resolve(target) not in toy_lib.targets():
        raise ValueError(f'{target!r} is not a target toy')
    toys = {short: layout['slots']['target']}
    toys.update({k: v for k, v in layout['slots'].items() if k != 'target'})
    return {'robot': layout['robot'], 'toys': toys}


def violations(layout: dict, layout_cfg: dict, arena_cfg: dict) -> list[str]:
    """Every constraint a layout breaks (empty for a good one); for tests and the 1A check."""
    half = float(arena_cfg['half_size_m'])
    tray = tuple(float(v) for v in arena_cfg['tray']['centre_m'])
    tray_side = float(arena_cfg['tray']['size_m'])
    c = {k: float(v) for k, v in layout_cfg.items() if k.endswith('_m')}
    out = []
    robot = layout['robot'][:2]
    if max(abs(robot[0]), abs(robot[1])) > half - c['robot_wall_clear_m'] + 1e-4:
        out.append('robot near a wall')
    if not _far(robot, tray, c['tray_clear_m'] + tray_side / 2 - 1e-4):
        out.append('robot near the tray')
    pts = [v[:2] for v in layout['slots'].values()]
    for name, p in zip(layout['slots'], pts):
        if max(abs(p[0]), abs(p[1])) > half - c['wall_clear_m'] + 1e-4:
            out.append(f'{name} near a wall')
        if not _far(p, tray, c['tray_clear_m'] - 1e-4):
            out.append(f'{name} near the tray')
        if not _far(p, robot, c['start_clear_m'] - 1e-4):
            out.append(f'{name} near the start')
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _far(pts[i], pts[j], c['toy_gap_m'] - 1e-4):
                out.append(f'slots {i} and {j} too close')
    return out


def formal_layouts(layout_cfg: dict, arena_cfg: dict) -> list[dict]:
    base, count = int(layout_cfg['formal_seed_base']), int(layout_cfg['formal_seeds'])
    return [make_layout(base + i, layout_cfg, arena_cfg) for i in range(count)]
