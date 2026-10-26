"""Plan 2P acceptance: our decoding + NMS on an Ultralytics ONNX gives the same boxes as Ultralytics.

    PYTHONPATH=src python scripts/vision/stage2/check_against_ultralytics.py \
        --model models/yolo/yolo11n_coco/yolo11n_coco_fp32_320.onnx \
        --images runs/<UTC>_s2a_dataset/images/test --n 200 --out runs/<UTC>_s2p_check.json

Needs the ultralytics package (same version as the export). Both sides see the same 320x320 input
(our dataset images are already the detector input, so Ultralytics' letterbox does nothing). A box
pair matches when the class is the same, IoU >= 0.99 and the scores differ by <= 1e-4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from microduck_toy.stage2.evaluate import iou
from microduck_toy.vision import detect


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    import ultralytics
    from ultralytics import YOLO
    theirs_model = YOLO(str(args.model), task='detect')
    ours = detect.OnnxYolo(args.model, iou=args.iou)
    paths = sorted(args.images.glob('*.png'))[:args.n]
    frames, worst_iou, worst_conf, bad = 0, 1.0, 0.0, []
    n_ours = n_theirs = 0
    for p in paths:
        img = cv2.imread(str(p))
        mine = ours(img, conf=args.conf)
        r = theirs_model.predict(img, imgsz=320, conf=args.conf, iou=args.iou, verbose=False,
                                 device='cpu')[0]
        theirs = [{'cls': r.names[int(c)], 'conf': float(s), 'xyxy': b.tolist()}
                  for b, s, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy())]
        n_ours, n_theirs = n_ours + len(mine), n_theirs + len(theirs)
        used = set()
        for m in mine:
            best, best_iou = None, 0.0
            for j, t in enumerate(theirs):
                if j not in used and t['cls'] == m['cls'] and iou(m['xyxy'], t['xyxy']) > best_iou:
                    best, best_iou = j, iou(m['xyxy'], t['xyxy'])
            if best is None:
                bad.append({'image': p.name, 'ours_only': m})
                continue
            used.add(best)
            worst_iou = min(worst_iou, best_iou)
            worst_conf = max(worst_conf, abs(m['conf'] - theirs[best]['conf']))
            if best_iou < 0.99 or abs(m['conf'] - theirs[best]['conf']) > 1e-4:
                bad.append({'image': p.name, 'ours': m, 'theirs': theirs[best]})
        bad += [{'image': p.name, 'theirs_only': t} for j, t in enumerate(theirs) if j not in used]
        frames += 1
    out = {'model': str(args.model), 'images': str(args.images), 'frames': frames,
           'boxes_ours': n_ours, 'boxes_theirs': n_theirs, 'worst_iou': round(worst_iou, 6),
           'worst_conf_diff': worst_conf, 'mismatches': bad[:20], 'n_mismatches': len(bad),
           'ultralytics': ultralytics.__version__, 'pass': not bad}
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print({k: v for k, v in out.items() if k != 'mismatches'})
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
