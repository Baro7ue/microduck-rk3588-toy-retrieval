"""The camera link's colour round trip, so rendered training images match what the board decodes.

The sim body sends each 640x360 render as packed UYVY (upstream mjlab_microduck/sim/camera.py:41-60:
BT.601, chroma averaged over each horizontal pixel pair of the RAW image). The board and the Mac
decode it with full-range BT.601 (camera_stream.save_rotated_png, d5_detect). Chroma is shared
along raw rows, which are columns of the upright frame, so the round trip runs on the raw image.
"""
from __future__ import annotations

import numpy as np

_Y = np.array([0.299, 0.587, 0.114])
_U = np.array([-0.168736, -0.331264, 0.5])
_V = np.array([0.5, -0.418688, -0.081312])


def encode(raw_rgb: np.ndarray) -> np.ndarray:
    """RGB (H x W x 3, W even) -> packed UYVY as (H, W/2, 4) uint8, upstream's arithmetic."""
    frame = raw_rgb.astype(np.float32)
    luma = frame @ _Y
    u = frame @ _U + 128.0
    v = frame @ _V + 128.0
    packed = np.empty((frame.shape[0], frame.shape[1] // 2, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((u[:, 0::2] + u[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 1] = np.clip(luma[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((v[:, 0::2] + v[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 3] = np.clip(luma[:, 1::2], 0, 255)
    return packed


def decode(pairs: np.ndarray) -> np.ndarray:
    """Packed UYVY (H, W/2, 4) -> RGB uint8 (H, W, 3), the board's full-range BT.601 decode."""
    h, w2 = pairs.shape[:2]
    y = np.empty((h, w2 * 2), dtype=np.float32)
    y[:, 0::2] = pairs[:, :, 1]
    y[:, 1::2] = pairs[:, :, 3]
    u = np.repeat(pairs[:, :, 0].astype(np.float32) - 128, 2, axis=1)
    v = np.repeat(pairs[:, :, 2].astype(np.float32) - 128, 2, axis=1)
    rgb = np.stack((y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u), axis=2)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def round_trip_upright(upright_rgb: np.ndarray) -> np.ndarray:
    """An upright render as the board would see it: back to raw, UYVY, decode, upright again."""
    raw = np.ascontiguousarray(np.rot90(upright_rgb, 1))
    return np.ascontiguousarray(np.rot90(decode(encode(raw)), 3))
