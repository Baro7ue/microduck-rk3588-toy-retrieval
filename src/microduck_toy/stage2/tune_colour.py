"""Tune the colour-threshold baseline on train frames only (plan 2B); writes the ranges as JSON.

    python -m microduck_toy.stage2.tune_colour --data runs/<UTC>_s2a_dataset --frames 400 \
        --out runs/<UTC>_s2b_colour/colour.json

Per colour class, coordinate descent from the start values in configs/vision/stage2.json
(baselines.colour): hue bounds, minimum saturation and value, minimum blob size; the score is the
class's AP50 on the first --frames train frames. The other toys stay in the frames, so a range
that also catches the green crocodile is paid for as false Henry boxes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from microduck_toy.paths import repo_root
from microduck_toy.stage2 import evaluate as ev
from microduck_toy.vision import detect

HUE_STEPS = (1, 2, 4, 8)
SV_STEPS = (10, 20, 40)
AREAS = (2, 3, 4, 6, 8, 12)


def load(data: Path, n: int) -> list:
    rows = [json.loads(x) for x in (data / 'meta' / 'train.jsonl').read_text().splitlines()][:n]
    out = []
    for m in rows:
        img = cv2.imread(str(data / 'images' / 'train' / f"{m['id']}.png"))
        out.append((m, detect.ColourBoxes.hsv(img)))
    return out


def score(frames: list, name: str, cls_cfg: dict, half: float) -> float:
    det = detect.ColourBoxes({'classes': {name: cls_cfg}, 'score_half_px': half})
    pairs = []
    n_gt = 0
    for m, hsv in frames:
        gts = [o['xyxy_input'] for o in m['objects'] if o['name'] == name]
        ign = [o['xyxy_input'] for o in m['dropped'] if o['name'] == name and o.get('xyxy_input')]
        n_gt += len(gts)
        pairs += ev.match_class(gts, ign, det.from_hsv(hsv), ev.IOU)
    return ev.ap_101(np.array([p[1] for p in pairs]), np.array([p[0] for p in pairs]), n_gt)


def candidates(c: dict):
    """Neighbours of one class setting: each hue bound, shared S and V minimum, blob size."""
    for i, (lo, hi) in enumerate(c['ranges_hsv']):
        for j, side in ((0, lo), (1, hi)):
            if (j == 0 and side[0] == 0) or (j == 1 and side[0] == 180):
                continue  # the red range's fixed ends
            for step in HUE_STEPS:
                for sign in (-1, 1):
                    h = side[0] + sign * step
                    if not 0 <= h <= 180:
                        continue
                    n = copy.deepcopy(c)
                    n['ranges_hsv'][i][j][0] = h
                    if n['ranges_hsv'][i][0][0] <= n['ranges_hsv'][i][1][0]:
                        yield n
    for k in (1, 2):
        for step in SV_STEPS:
            for sign in (-1, 1):
                v = c['ranges_hsv'][0][0][k] + sign * step
                if 0 <= v <= 250:
                    n = copy.deepcopy(c)
                    for r in n['ranges_hsv']:
                        r[0][k] = v
                    yield n
    for a in AREAS:
        if a != c['min_area_px']:
            n = copy.deepcopy(c)
            n['min_area_px'] = a
            yield n


def tune(frames: list, name: str, start: dict, half: float, rounds: int = 6) -> tuple[dict, float, float]:
    best = copy.deepcopy(start)
    first = best_ap = score(frames, name, best, half)
    for _ in range(rounds):
        improved = False
        for cand in list(candidates(best)):
            ap = score(frames, name, cand, half)
            if ap > best_ap + 1e-4:
                best, best_ap, improved = cand, ap, True
        if not improved:
            break
    return best, first, best_ap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=400)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = json.loads((repo_root() / 'configs/vision/stage2.json').read_text())
    col = cfg['baselines']['colour']
    frames = load(args.data, args.frames)
    out = {'data': str(args.data), 'split': 'train', 'frames': len(frames),
           'score_half_px': col['score_half_px'], 'classes': {}, 'ap50_train': {}}
    for name, start in col['classes'].items():
        best, first, best_ap = tune(frames, name, start, col['score_half_px'])
        out['classes'][name] = best
        out['ap50_train'][name] = {'start': round(first, 4), 'tuned': round(best_ap, 4)}
        print(name, out['ap50_train'][name], json.dumps(best), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
