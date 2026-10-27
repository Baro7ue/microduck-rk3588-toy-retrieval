"""Robot's-eye pictures: the camera frame the brain decided on, with the YOLO boxes it got.

The brain saves one at each decision that rests on the picture (Brain.shot): the spin frame that
found a candidate, the last frame of each standing look (named by its result), the last frame of
each B-segment look, the tray look and the RECOVER looks; at most MAX_PER_EPISODE an episode, as
episodes/<label>/frames/NN_<tag>_s<seq>.jpg. Drawn in the upright 360x640 picture the detector
works in: each box in its toy's colour (thick for the target) with class and confidence, the search
window's band when the frame went through the window, and a title line. English labels, so the
pictures go into the report as they are.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

COLOURS = {'bus': '#e0b000', 'henry': '#2e8b57', 'ketchup': '#d62728', 'croc': '#7a8f2a',
           'thomas': '#1f6fb4', 'tray': '#9a9a9a'}
MAX_PER_EPISODE = 24
TITLE_H = 18


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                     # Pillow < 10.1: the fixed bitmap font
        return ImageFont.load_default()


def draw(rgb: np.ndarray, boxes: list[dict], title: str, target: str | None = None,
         window_y0: int | None = None, window_h: int = 360):
    """The picture with its boxes, as a PIL image (title band above the frame)."""
    from PIL import Image, ImageDraw
    h, w = rgb.shape[:2]
    img = Image.new('RGB', (w, h + TITLE_H), (0, 0, 0))
    img.paste(Image.fromarray(rgb), (0, TITLE_H))
    d = ImageDraw.Draw(img)
    small = _font(11)
    d.text((3, 3), title, fill=(255, 255, 255), font=small)
    if window_y0 is not None:
        top, bottom = TITLE_H + window_y0, TITLE_H + min(window_y0 + window_h, h) - 1
        for y in (top, bottom):
            for x in range(0, w, 12):                       # dashed: the window is not a detection
                d.line([(x, y), (min(x + 6, w - 1), y)], fill=(255, 255, 255), width=1)
    for box in boxes:
        x0, y0, x1, y1 = box['xyxy']
        colour = COLOURS.get(box['cls'], '#ff00ff')
        thick = 3 if box['cls'] == target else 2
        d.rectangle([x0, TITLE_H + y0, x1, TITLE_H + y1], outline=colour, width=thick)
        label = f"{box['cls']} {box['conf']:.2f}"
        tx, ty = x0, max(TITLE_H, TITLE_H + y0 - 13)
        tw = d.textlength(label, font=small) if hasattr(d, 'textlength') else 7 * len(label)
        d.rectangle([tx, ty, tx + tw + 4, ty + 12], fill=colour)
        d.text((tx + 2, ty), label, fill=(0, 0, 0) if box['cls'] in ('bus', 'tray') else (255, 255, 255),
               font=small)
    return img


def save(path: Path, rgb: np.ndarray, boxes: list[dict], title: str, target: str | None = None,
         window_y0: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draw(rgb, boxes, title, target, window_y0).save(path, quality=85)
    return path
