"""Colour-threshold detector: the Stage 1 cross-check of the oracle box, and the Stage 2 baseline.

    det = ColourDetector(stage1['colour'])
    boxes = det.detect(rgb_upright, names=['SCHOOL_BUS'])

HSV in OpenCV's ranges (H 0-180, S and V 0-255), computed with numpy so it runs wherever the
control code runs. For each toy: pixels inside any of its ranges, then the largest 4-connected blob
on a 2x-downsampled mask (so one stray pixel on the floor does not stretch the box), then that blob's
box at full resolution. Same box format as the oracle and YOLO: xyxy in the upright 360x640 image.
"""
from __future__ import annotations

import numpy as np

from microduck_toy.scene import toys as toy_lib


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    some = d > 1e-6
    is_r = some & (mx == r)
    is_g = some & (mx == g) & ~is_r
    is_b = some & ~is_r & ~is_g
    h[is_r] = ((g - b)[is_r] / d[is_r]) % 6.0
    h[is_g] = (b - r)[is_g] / d[is_g] + 2.0
    h[is_b] = (r - g)[is_b] / d[is_b] + 4.0
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    return np.stack([h * 30.0, s * 255.0, mx * 255.0], axis=-1)


def largest_blob(mask: np.ndarray) -> np.ndarray | None:
    """Boolean mask of the largest 4-connected component, or None if the mask is empty."""
    if not mask.any():
        return None
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    best, best_size, current = 0, 0, 0
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        current += 1
        labels[start] = current
        stack, size = [start], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        if size > best_size:
            best, best_size = current, size
    return labels == best


class ColourDetector:
    def __init__(self, cfg: dict):
        self.ranges = {toy_lib.resolve(k): [(np.array(lo, float), np.array(hi, float))
                                            for lo, hi in v]
                       for k, v in cfg['ranges_hsv'].items()}
        self.min_area = int(cfg.get('min_area_px', 12))

    def mask(self, hsv: np.ndarray, name: str) -> np.ndarray:
        out = np.zeros(hsv.shape[:2], dtype=bool)
        for lo, hi in self.ranges[toy_lib.resolve(name)]:
            out |= np.all((hsv >= lo) & (hsv <= hi), axis=-1)
        return out

    def detect(self, rgb: np.ndarray, names: list[str] | None = None) -> list[dict]:
        hsv = rgb_to_hsv(rgb)
        height, width = rgb.shape[:2]
        boxes = []
        for name in names or list(self.ranges):
            full = self.mask(hsv, name)
            if int(full.sum()) < self.min_area:
                continue
            small = full[0::2, 0::2] | full[1::2, 0::2] | full[0::2, 1::2] | full[1::2, 1::2]
            blob = largest_blob(small)
            if blob is None:
                continue
            ys, xs = np.nonzero(blob)
            y0, y1, x0, x1 = 2 * ys.min(), 2 * ys.max() + 2, 2 * xs.min(), 2 * xs.max() + 2
            inside = full[y0:y1, x0:x1]
            if int(inside.sum()) < self.min_area:
                continue
            ry, rx = np.nonzero(inside)
            box = [int(x0 + rx.min()), int(y0 + ry.min()), int(x0 + rx.max()) + 1,
                   int(y0 + ry.max()) + 1]
            boxes.append({'name': toy_lib.SHORT[toy_lib.resolve(name)], 'conf': 1.0, 'xyxy': box,
                          'area_px': int(inside.sum()),
                          'cut': bool(box[0] <= 0 or box[1] <= 0 or box[2] >= width
                                      or box[3] >= height)})
        return boxes
