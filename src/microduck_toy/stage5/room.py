"""Stage 5 room demo: the room config and its toy layout (not a formal result).

    doc = load()                                  # configs/demo/room.json
    s1 = merged(doc, 'stage1')                    # configs/control/stage1.json with the room over it
    lay = layout(doc, seed)                       # {'robot': [x, y, yaw], 'toys': {'bus': [...], 'bus_2': [...], ...}}

The room keeps the frozen detector and brain; only the field size, the tray position, the search's
relocate spots and the time limit change (configs/demo/room.json), plus walls, furniture and copies
of the toys in the sim body (scene/arena.py). The board brain imports load() and deep_merge() only
(brain.load_configs), so nothing here imports mujoco at module level.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random

from microduck_toy.paths import repo_root

FROZEN = {'stage1': 'configs/control/stage1.json', 'stage3': 'configs/control/stage3.json'}


def path() -> Path:
    return repo_root() / 'configs' / 'demo' / 'room.json'


def load(p: str | Path | None = None) -> dict:
    return json.loads(Path(p or path()).read_text())


def deep_merge(base: dict, over: dict) -> dict:
    """Dicts merge key by key; anything else in `over` replaces. Neither input changes."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def merged(doc: dict, name: str) -> dict:
    base = json.loads((repo_root() / FROZEN[name]).read_text())
    return deep_merge(base, doc.get(name, {}))


def toy_names(doc: dict) -> list[str]:
    """Short names of every toy in the room: bus, henry, ..., then bus_2, henry_2, croc_2, croc_3, ..."""
    from microduck_toy.scene import toys as toy_lib     # imports mujoco: not on the board brain's path
    names = toy_lib.register_copies(doc['stage1']['arena'].get('copies'))
    return [toy_lib.SHORT[n] for n in names]


def layout(doc: dict, seed: int | None = None, tries: int = 20000) -> dict:
    """Every toy at a random spot and yaw in the room's toy field, keeping the demo's gaps."""
    d = doc['demo']
    seed = int(d['seed'] if seed is None else seed)
    rng = random.Random(seed)
    tray = doc['stage1']['arena']['tray']['centre_m']
    robot = [float(v) for v in d['robot']]
    lim, gap = float(d['toy_field_m']), float(d['toy_gap_m'])
    placed: dict[str, list[float]] = {}
    for name in toy_names(doc):
        for _ in range(tries):
            x, y = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
            if math.dist((x, y), tray) < float(d['tray_clear_m']):
                continue
            if math.dist((x, y), robot[:2]) < float(d['start_clear_m']):
                continue
            if any(math.dist((x, y), p[:2]) < gap for p in placed.values()):
                continue
            placed[name] = [round(x, 4), round(y, 4), round(rng.uniform(-math.pi, math.pi), 4)]
            break
        else:
            raise RuntimeError(f'seed {seed}: no room left for {name}')
    return {'seed': seed, 'robot': robot, 'toys': placed}
