"""What the Stage 2 detector is fed: one 320x320 input per camera frame, taken one of two ways.

    window      searching (head 0 to +0.3): a 360x360 square of the upright frame whose top edge sits
                `margin_rows` above the horizon, scaled to 320 (x0.889, no padding). Toys lie on the
                floor, so they are always below the horizon; the rows above it show walls and sky.
    letterbox   approaching (H1): the whole 360x640 frame scaled to 320 high (x0.5) and padded left
                and right with grey 114, as the Stage 0 detector did (yolo_postprocess.letterbox).

    m = window_mapping(horizon_row(rot))      # rot: camera OpenCV axes -> level frame
    m = letterbox_mapping()
    img320 = to_input(frame, m)               # frame: upright 360x640x3 uint8
    box320 = to_input_box(box, m); box = to_frame_box(box320, m)

A far toy is 1.78 times longer in the window input than in the letterbox input (0.889 / 0.5), for
the same single inference (plan section 2). The horizon row comes from the camera pose: robot.state
on the robot (ranging.camera_in_level), the MuJoCo camera in the renderer.

The frame is upright, 360 wide x 640 high: np.rot90(raw, 3) of the camera's 640x360 image, focal
length 434.6 px (fovy 45 deg across the raw 360 px).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

FRAME_W, FRAME_H = 360, 640
INPUT = 320
WINDOW = 360
MARGIN_ROWS = 30
PAD_VALUE = 114
FOCAL_PX = 180.0 / math.tan(math.radians(22.5))


@dataclass(frozen=True)
class Mapping:
    """Frame -> input: input = (frame - (x0, y0)) * gain + (pad_x, pad_y)."""
    kind: str          # 'window' or 'letterbox'
    x0: float
    y0: float
    gain: float
    pad_x: float
    pad_y: float

    def as_dict(self) -> dict:
        return {'kind': self.kind, 'x0': self.x0, 'y0': self.y0, 'gain': self.gain,
                'pad_x': self.pad_x, 'pad_y': self.pad_y}

    @classmethod
    def from_dict(cls, d: dict) -> 'Mapping':
        return cls(d['kind'], float(d['x0']), float(d['y0']), float(d['gain']),
                   float(d['pad_x']), float(d['pad_y']))


def horizon_row(rot_level, focal_px: float = FOCAL_PX) -> float | None:
    """Upright-image row of the horizon straight ahead of the camera. rot_level: camera OpenCV axes
    (+x right, +y down, +z forward) -> level frame (z up). None if the camera looks straight
    up or down."""
    rot = np.asarray(rot_level, float)
    ahead = rot[:, 2].copy()
    ahead[2] = 0.0
    n = float(np.linalg.norm(ahead))
    if n < 1e-6:
        return None
    v = rot.T @ (ahead / n)
    if v[2] <= 1e-6:
        return None
    return FRAME_H / 2 + focal_px * v[1] / v[2]


def window_top(horizon: float, margin_rows: int = MARGIN_ROWS) -> int:
    """Top row of the window: `margin_rows` above the horizon, kept inside the frame."""
    return int(min(max(round(horizon - margin_rows), 0), FRAME_H - WINDOW))


def window_mapping(horizon: float, margin_rows: int = MARGIN_ROWS) -> Mapping:
    return Mapping('window', 0.0, float(window_top(horizon, margin_rows)), INPUT / WINDOW, 0.0, 0.0)


def letterbox_mapping() -> Mapping:
    # Same arithmetic as yolo_postprocess.letterbox for a 360x640 frame: 180x320, 70 px each side.
    gain = min(INPUT / FRAME_H, INPUT / FRAME_W)
    new_w = round(FRAME_W * gain)
    pad_x = round((INPUT - new_w) / 2 - 0.1)
    return Mapping('letterbox', 0.0, 0.0, gain, float(pad_x), 0.0)


def to_input(frame: np.ndarray, m: Mapping) -> np.ndarray:
    """The 320x320 uint8 image the detector sees (same channel order as `frame`)."""
    if frame.shape[:2] != (FRAME_H, FRAME_W):
        raise ValueError(f'expected an upright {FRAME_W}x{FRAME_H} frame, got {frame.shape}')
    if m.kind == 'window':
        y0 = int(m.y0)
        crop = np.ascontiguousarray(frame[y0:y0 + WINDOW, :])
        return cv2.resize(crop, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
    if m.kind == 'letterbox':
        new_w, new_h = round(FRAME_W * m.gain), round(FRAME_H * m.gain)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        left = int(m.pad_x)
        right = INPUT - new_w - left
        return cv2.copyMakeBorder(small, 0, INPUT - new_h, left, right, cv2.BORDER_CONSTANT,
                                  value=(PAD_VALUE,) * 3)
    raise ValueError(f'unknown mapping {m.kind!r}')


def to_input_box(xyxy, m: Mapping, clip: bool = True) -> list[float] | None:
    """A frame box in input pixels; with clip, cut to the input (None if nothing is left)."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    out = [(x0 - m.x0) * m.gain + m.pad_x, (y0 - m.y0) * m.gain + m.pad_y,
           (x1 - m.x0) * m.gain + m.pad_x, (y1 - m.y0) * m.gain + m.pad_y]
    if not clip:
        return out
    lo_x, hi_x = (m.pad_x, INPUT - m.pad_x) if m.kind == 'letterbox' else (0.0, float(INPUT))
    out = [min(max(out[0], lo_x), hi_x), min(max(out[1], 0.0), INPUT),
           min(max(out[2], lo_x), hi_x), min(max(out[3], 0.0), INPUT)]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def to_frame_box(xyxy, m: Mapping) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    return [(x0 - m.pad_x) / m.gain + m.x0, (y0 - m.pad_y) / m.gain + m.y0,
            (x1 - m.pad_x) / m.gain + m.x0, (y1 - m.pad_y) / m.gain + m.y0]


def tensor(img320: np.ndarray, bgr: bool = True) -> np.ndarray:
    """[1, 3, 320, 320] float32 RGB in 0-1, the ONNX input (from BGR by default, as cv2 reads)."""
    rgb = img320[:, :, ::-1] if bgr else img320
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
