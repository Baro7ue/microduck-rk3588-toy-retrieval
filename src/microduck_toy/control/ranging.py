"""From a box in the head camera's upright image to a point on the floor, using only robot.state.

    cam = CameraModel.from_config(stage1['camera'])
    fwd, left = ground_point(cam, state, box_xyxy, plane_z=toy_half_height)

What robot.state gives (robotd duck-ipc-proto, RobotState):
    frames.camera   camera pose in the trunk frame at this tick's measured head joints, OpenCV axes
                    (+x right, +y down, +z along the optical axis), from the same head FK robot.look
                    uses (kinematics/src/head.rs)
    imu.quat        trunk orientation (trunk -> IMU world, scalar first); only its tilt is used
    odom.position   trunk position; z is the trunk's height above the ground (contact odometry)

The answer is in the level trunk frame: origin under the trunk, x along the trunk's heading on the
floor, y to its left. The ray through the box centre is cut with the horizontal plane z = plane_z
(half the toy's height: the box centre is about the middle of the toy).

The image the detector sees is upright, 360 wide x 640 high (np.rot90(raw, 3) of the 640x360
frame). `frame` says which image the OpenCV axes of frames.camera belong to: 'upright' or 'raw';
checked against the simulator in Stage 1 (1D).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int = 360            # upright image
    height: int = 640
    focal_px: float = 180.0 / math.tan(math.radians(22.5))
    frame: str = 'upright'      # which image frames.camera's OpenCV axes describe

    @classmethod
    def from_config(cls, cfg: dict) -> 'CameraModel':
        raw_w, raw_h = int(cfg['raw_width']), int(cfg['raw_height'])
        focal = (raw_h / 2) / math.tan(math.radians(float(cfg['fovy_deg']) / 2))
        return cls(width=raw_h, height=raw_w, focal_px=focal, frame=cfg.get('frame', 'upright'))

    def pixel_ray(self, col: float, row: float) -> np.ndarray:
        """Ray (OpenCV camera axes) through an upright-image pixel position (col, row)."""
        if self.frame == 'upright':
            x, y, cx, cy = col, row, self.width / 2, self.height / 2
        elif self.frame == 'raw':
            # upright (col, row) = raw (u = row, v = width - col): raw is height wide, width high
            x, y = row, self.width - col
            cx, cy = self.height / 2, self.width / 2
        else:
            raise ValueError(f'unknown frame {self.frame!r}')
        return np.array([(x - cx) / self.focal_px, (y - cy) / self.focal_px, 1.0])


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def level_matrix(state: dict) -> np.ndarray:
    """Trunk -> level frame (the trunk's tilt only, heading removed)."""
    imu = state.get('imu') or {}
    if 'quat' in imu:
        rot = quat_matrix(imu['quat'])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        c, s = math.cos(-yaw), math.sin(-yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot
    # Gravity only (older robotd): the smallest rotation taking trunk gravity to (0, 0, -1).
    g = np.asarray(state['safety']['gravity'], float)
    g = g / np.linalg.norm(g)
    down = np.array([0.0, 0.0, -1.0])
    axis = np.cross(g, down)
    s, c = np.linalg.norm(axis), float(g @ down)
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(math.atan2(s, c)) * kx + (1 - c) * (kx @ kx)


def camera_in_level(state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Camera position (m) and rotation (OpenCV axes -> level frame), trunk height included."""
    cam = state['frames']['camera']
    level = level_matrix(state)
    trunk_z = float(state['odom']['position'][2])
    pos = level @ np.asarray(cam['pos'], float) + np.array([0.0, 0.0, trunk_z])
    return pos, level @ quat_matrix(cam['quat'])


def ground_point(cam: CameraModel, state: dict, xyxy, plane_z: float,
                 point: str = 'centre') -> tuple[float, float] | None:
    """(forward, left) in metres of the box's point on the plane z = plane_z, or None when the
    ray does not come down to it."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    col = (x0 + x1) / 2
    row = (y0 + y1) / 2 if point == 'centre' else y1
    origin, rot = camera_in_level(state)
    ray = rot @ cam.pixel_ray(col, row)
    if ray[2] >= -1e-6:
        return None
    t = (plane_z - origin[2]) / ray[2]
    if t <= 0:
        return None
    hit = origin + t * ray
    return float(hit[0]), float(hit[1])


def project(cam: CameraModel, state: dict, point_level) -> tuple[float, float] | None:
    """Where a level-frame point lands in the upright image (col, row); None behind the camera.
    The inverse of ground_point, for checks."""
    origin, rot = camera_in_level(state)
    v = rot.T @ (np.asarray(point_level, float) - origin)
    if v[2] <= 1e-6:
        return None
    x = cam.focal_px * v[0] / v[2]
    y = cam.focal_px * v[1] / v[2]
    if cam.frame == 'upright':
        return x + cam.width / 2, y + cam.height / 2
    u, vv = x + cam.height / 2, y + cam.width / 2
    return cam.width - vv, u
