"""The five Stage 1 toys: where their meshes are, and the box each one gets in the simulator.

sim/assets/toys/toys.json holds the facts (scale, mass, the rotation that lays the ketchup bottle
down, the long axis). Everything else is computed from the scaled mesh here, once:

    toy frame   x = the long axis, z = up, origin = the centre of the scaled mesh's bounding box
    mesh        placed in that frame by `mesh_quat` and `mesh_pos`
    collision   one box with half sizes `half`; the toy rests on the floor at z = half[2]

The box is the plan's "simple collision shape" (final plan, section 4). A toy's yaw in the world is
the direction its long axis points.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_toy.paths import repo_root


def assets_dir() -> Path:
    return repo_root() / 'sim' / 'assets' / 'toys'


@dataclass(frozen=True)
class Toy:
    name: str
    role: str            # 'target' or 'distractor'
    label: str
    scale: float
    mass_kg: float
    mesh: Path           # model.obj (unscaled; the scale goes on the MuJoCo mesh)
    texture: Path
    mesh_quat: tuple     # mesh -> toy frame rotation, scalar first
    mesh_pos: tuple      # mesh origin in the toy frame, metres
    half: tuple          # box half sizes in the toy frame, metres

    @property
    def rest_z(self) -> float:
        return self.half[2]


def mesh_vertices(path: Path) -> np.ndarray:
    rows = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith('v ')]
    return np.array(rows, dtype=float)


def yaw_matrix(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mat_to_quat(m: np.ndarray) -> tuple:
    """Rotation matrix to a unit quaternion (w, x, y, z)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, dtype=float).flatten())
    return tuple(float(v) for v in q)


def _build(row: dict, folder: Path) -> Toy:
    scale = float(row['scale'])
    # Mesh -> toy frame: lay down (base_rotation), then turn the long axis onto +x.
    rot = yaw_matrix(-math.radians(row['axis_deg'])) @ np.array(row['base_rotation'], float)
    verts = (mesh_vertices(folder / 'model.obj') * scale) @ rot.T
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2
    return Toy(name=row['name'], role=row['role'], label=row['label'], scale=scale,
               mass_kg=float(row['mass_g']) / 1000.0, mesh=folder / 'model.obj',
               texture=folder / 'texture.png', mesh_quat=mat_to_quat(rot),
               mesh_pos=tuple(float(v) for v in -centre),
               half=tuple(float(v) for v in (hi - lo) / 2))


@lru_cache(maxsize=None)
def catalogue(folder: str | None = None) -> dict[str, Toy]:
    """All five toys by name, targets first (the order of toys.json)."""
    root = Path(folder) if folder else assets_dir()
    doc = json.loads((root / 'toys.json').read_text())
    return {row['name']: _build(row, root / row['name']) for row in doc['toys']}


def targets() -> list[str]:
    return [name for name, toy in catalogue().items() if toy.role == 'target']


def distractors() -> list[str]:
    return [name for name, toy in catalogue().items() if toy.role == 'distractor']


SHORT = {
    'SCHOOL_BUS': 'bus',
    'Thomas_Friends_Woodan_Railway_Henry': 'henry',
    'FOOD_BEVERAGE_SET__2': 'ketchup',
    'My_First_Wiggle_Crocodile': 'croc',
    'Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj': 'thomas',
}
LONG = {v: k for k, v in SHORT.items()}


def resolve(name: str) -> str:
    """Accept the short name (bus, henry, ketchup, croc, thomas) or the GSO name."""
    return LONG.get(name, name)
