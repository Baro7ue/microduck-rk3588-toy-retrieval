"""Label audit sheet (plan 2A): random dataset frames with their label boxes drawn, as one PNG.

    python -m microduck_toy.stage2.sheet --data runs/<UTC>_s2a_dataset --split train --n 100 \
        --out sheet.png [--seed 0] [--cols 10]

Each tile is the 320x320 input as the detector sees it, boxes coloured by class with the class
name and the camera distance (m); the caption under it gives the frame id and the view.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLOURS = {'bus': (0, 200, 255), 'henry': (0, 160, 0), 'ketchup': (0, 0, 230),
           'croc': (120, 255, 120), 'thomas': (255, 80, 0), 'tray': (200, 200, 200)}


def tile(data: Path, split: str, meta: dict, scale: int = 1) -> np.ndarray:
    img = cv2.imread(str(data / 'images' / split / f"{meta['id']}.png"))
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for o in meta['objects']:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in o['xyxy_input'])
        colour = COLOURS[o['name']]
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 1)
        cv2.putText(img, f"{o['name']} {o['dist_cam_m']:.1f}", (x0, max(9, y0 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32 * scale, colour, 1, cv2.LINE_AA)
    cap = np.full((16, img.shape[1], 3), 255, np.uint8)
    cv2.putText(cap, f"{meta['id'][-6:]} {meta['view']} {meta['floor']}", (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([img, cap])


def sheet(data: Path, split: str, n: int, seed: int, cols: int) -> np.ndarray:
    rows = [json.loads(line) for line in (data / 'meta' / f'{split}.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False))
    tiles = [tile(data, split, rows[i]) for i in pick]
    h, w = tiles[0].shape[:2]
    blank = np.full((h, w, 3), 255, np.uint8)
    tiles += [blank] * (-len(tiles) % cols)
    grid = [np.hstack([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255)
                       for t in tiles[r:r + cols]]) for r in range(0, len(tiles), cols)]
    return np.vstack(grid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cols', type=int, default=10)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cv2.imwrite(str(args.out), sheet(args.data, args.split, args.n, args.seed, args.cols))
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
