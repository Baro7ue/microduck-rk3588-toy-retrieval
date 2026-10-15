"""Kinematic renders of the Stage 1 field: the robot posed (no physics, no robotd), toys placed,
the head camera rendered as the detector sees it, plus the segmentation boxes.

    scene = ArenaRender(rl_root, cfg['arena'], floor='grid')
    scene.pose('H2')                                # STAND keyframe + head offsets, at the origin
    scene.place('bus', 0.20, 0.015, yaw=0.5)        # toys at rest; others stay parked
    rgb, boxes = scene.view()                       # upright 360x640 RGB, oracle boxes

Used by the Mac/cloud render parity check (1A) and, later, for detector training data. The
camera turn and the upright rotation are the same as the body server's (sim/camera.py turn,
np.rot90 k=3), and the boxes come from the same function as the arena body's oracle.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_toy.scene import arena
from microduck_toy.scene import toys as toy_lib

HEADS = {'H0': (0.0, 0.0), 'H1': (0.0, 0.6), 'H2': (0.0, 1.0), 'H3': (-0.4, 0.8)}
RAW_W, RAW_H = 640, 360


def turned_camera(model: mujoco.MjModel, name: str = 'head_camera') -> int:
    """The camera id, turned 180 degrees about its right axis as mjlab_microduck/sim/camera.py
    does (the MJCF camera faces backwards)."""
    cam = model.camera(name).id
    fixed = np.zeros(4)
    mujoco.mju_mulQuat(fixed, model.cam_quat[cam], np.array([0.0, 1.0, 0.0, 0.0]))
    model.cam_quat[cam] = fixed
    return cam


def boxes_from_seg(seg: np.ndarray, owners: dict[int, str]) -> list[dict]:
    """Boxes in the upright image (360 wide x 640 high) from a raw 640x360 segmentation render.
    owners: geom id -> name. Same output as the arena body's oracle."""
    ids, kinds = seg[:, :, 0], seg[:, :, 1]
    height, width = ids.shape
    geom = kinds == int(mujoco.mjtObj.mjOBJ_GEOM)
    found: dict[str, list] = {}
    for gid, owner in owners.items():
        mask = geom & (ids == gid)
        if not mask.any():
            continue
        vs, us = np.nonzero(mask)
        found.setdefault(owner, []).append((us.min(), us.max(), vs.min(), vs.max(), len(us)))
    boxes = []
    for owner, parts in found.items():
        u0 = min(p[0] for p in parts); u1 = max(p[1] for p in parts)
        v0 = min(p[2] for p in parts); v1 = max(p[3] for p in parts)
        # Upright (np.rot90 k=3): column = (height - 1) - v, row = u.
        boxes.append({'name': owner, 'conf': 1.0,
                      'xyxy': [int(height - 1 - v1), int(u0), int(height - v0), int(u1) + 1],
                      'area_px': int(sum(p[4] for p in parts)),
                      'cut': bool(u0 <= 0 or v0 <= 0 or u1 >= width - 1 or v1 >= height - 1)})
    return boxes


def owners(model: mujoco.MjModel, ids: arena.ArenaIds) -> dict[int, str]:
    out = {toy['visual']: toy_lib.SHORT[name] for name, toy in ids.toys.items()}
    out.update({g: 'tray' for g in ids.tray_geoms})
    return out


def upright(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, 3))


class ArenaRender:
    def __init__(self, rl_root: Path, arena_cfg: dict, floor: str = 'grid'):
        self.model = arena.build_spec(arena_cfg, Path(rl_root), floor).compile()
        self.data = mujoco.MjData(self.model)
        self.ids = arena.ArenaIds(self.model)
        self.owners = owners(self.model, self.ids)
        self.cam = turned_camera(self.model)
        self.key = self.model.key('STAND').id
        self.rgb = mujoco.Renderer(self.model, height=RAW_H, width=RAW_W)
        self.seg = mujoco.Renderer(self.model, height=RAW_H, width=RAW_W)
        self.seg.enable_segmentation_rendering()
        self.pose('H0')

    def pose(self, head: str | tuple[float, float], x: float = 0.0, y: float = 0.0,
             yaw: float = 0.0) -> None:
        """STAND keyframe at (x, y, yaw), head offsets added to the keyframe angles (robotd's
        head command is an offset from HOME), every toy parked."""
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.key)
        neck, head_pitch = HEADS[head] if isinstance(head, str) else head
        d.qpos[m.joint('neck_pitch').qposadr[0]] += neck
        d.qpos[m.joint('head_pitch').qposadr[0]] += head_pitch
        d.qpos[0:2] = [x, y]
        d.qpos[3:7] = arena.yaw_quat(yaw)
        for name in self.ids.toys:
            arena.park_toy(m, d, self.ids, name)
        mujoco.mj_forward(m, d)

    def seg_upright(self) -> np.ndarray:
        """The last view's segmentation, upright: geom id per pixel, -1 where no geom."""
        ids = np.where(self.last_seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
                       self.last_seg[:, :, 0], -1)
        return upright(ids)

    def place(self, name: str, x: float, y: float, yaw: float = 0.0) -> None:
        arena.place_toy(self.model, self.data, self.ids, toy_lib.resolve(name), x, y, yaw)
        mujoco.mj_forward(self.model, self.data)

    def view(self, with_rgb: bool = True) -> tuple[np.ndarray | None, list[dict]]:
        self.seg.update_scene(self.data, camera=self.cam)
        self.last_seg = self.seg.render().copy()
        boxes = boxes_from_seg(self.last_seg, self.owners)
        rgb = None
        if with_rgb:
            self.rgb.update_scene(self.data, camera=self.cam)
            rgb = upright(self.rgb.render())
        return rgb, boxes

    def trunk_frame_point(self, forward: float, left: float) -> tuple[float, float]:
        """World (x, y) of a point given in the posed trunk's heading frame."""
        d = self.data
        w, qx, qy, qz = (float(v) for v in d.qpos[3:7])
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, s = math.cos(yaw), math.sin(yaw)
        return float(d.qpos[0] + c * forward - s * left), float(d.qpos[1] + s * forward + c * left)
