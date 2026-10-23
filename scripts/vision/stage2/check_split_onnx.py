"""Plan 2D acceptance: the split ONNX + decode_split gives the same boxes as Ultralytics' own ONNX.

    PYTHONPATH=src python scripts/vision/stage2/check_split_onnx.py \
        --split runs/<UTC>_s2d_export/yolo11n_toys_split_320.onnx \
        --single runs/<UTC>_s2d_export/yolo11n_toys_single_320.onnx \
        --images runs/<UTC>_s2a_dataset/images/test --n 300 --out runs/<UTC>_s2d_export/split_check.json

Both files come from the same best.pt (export_split_onnx.py). The single file is decoded with
decode_single, which 2P already matched box for box against Ultralytics; checking the single file
against Ultralytics again (check_against_ultralytics.py) closes the chain. Boxes pair by class and
overlap (check_tools.pair_boxes); every pair must be within 0.01 px at each corner and 1e-4 in
score. A box left without a partner is an NMS tie, not a mismatch, when its IoU with a
higher-scoring box of its class is within 1e-4 of the NMS threshold (0.44999994 vs 0.45: float
rounding decides which side keeps it). Checked at conf 0.25 (detector) and 0.001 (what the
evaluator reads).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from microduck_toy.stage2.check_tools import nms_tie, pair_boxes
from microduck_toy.vision import detect

BOX_PX = 0.01
CONF = 1e-4


def compare(a, b, nms_iou: float) -> dict:
    r = pair_boxes(a, b)
    alone = [(i, a) for i in r['unpaired_a']] + [(j, b) for j in r['unpaired_b']]
    ties = sum(nms_tie(i, side, nms_iou) for i, side in alone)
    bad_pairs = r['box_px'] > BOX_PX or r['conf'] > CONF
    return {'pairs': r['pairs'], 'box_px': r['box_px'], 'conf': r['conf'], 'nms_ties': ties,
            'unpaired': len(alone) - ties + (1 if bad_pairs else 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--single', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    split, single = detect.OnnxYoloSplit(args.split), detect.OnnxYolo(args.single)
    if split.names != single.names:
        raise SystemExit(f'class names differ: {split.names} vs {single.names}')
    paths = sorted(args.images.glob('*.png'))[:args.n]
    out = {'split': str(args.split), 'single': str(args.single), 'images': str(args.images),
           'frames': len(paths), 'tolerance': {'box_px': BOX_PX, 'conf': CONF}, 'by_conf': {}}
    for conf in (0.25, 0.001):
        tot = {'pairs': 0, 'box_px': 0.0, 'conf': 0.0, 'nms_ties': 0, 'unpaired': 0}
        bad = []
        for p in paths:
            img = cv2.imread(str(p))
            a = detect.decode_split(split.raw(img), conf, split.iou)
            b = detect.decode_single(single.raw(img), conf, single.iou)
            r = compare(a, b, split.iou)
            for k in ('pairs', 'nms_ties', 'unpaired'):
                tot[k] += r[k]
            tot['box_px'], tot['conf'] = max(tot['box_px'], r['box_px']), max(tot['conf'], r['conf'])
            if r['unpaired'] or r['nms_ties']:
                bad.append({'image': p.name, **r})
        out['by_conf'][str(conf)] = {**tot, 'frames_with_unpaired_or_ties': bad[:10]}
    out['pass'] = all(v['unpaired'] == 0 for v in out['by_conf'].values())
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({c: {k: v[k] for k in ('pairs', 'box_px', 'conf', 'nms_ties', 'unpaired')}
                      for c, v in out['by_conf'].items()}), 'pass' if out['pass'] else 'FAIL')
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
