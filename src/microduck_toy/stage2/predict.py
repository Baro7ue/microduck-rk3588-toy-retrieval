"""Run one detector over a dataset split and write its boxes for stage2.evaluate (plans 2B, 2E).

    python -m microduck_toy.stage2.predict --data runs/<UTC>_s2a_dataset --split test \
        --detector onnx --model models/yolo/yolo11n_coco/yolo11n_coco_fp32_320.onnx \
        --out runs/<UTC>_s2b_coco/test.jsonl
    python -m microduck_toy.stage2.predict --data ... --split test --detector colour --out ...
    python -m microduck_toy.stage2.predict --data ... --split test --detector onnx-split \
        --model runs/<UTC>_s2d_export/yolo11n_toys_split_320.onnx --out ...
    python -m microduck_toy.stage2.predict --data ... --split test --detector rknn --model x.rknn ...

--detector rknn runs on the board (RKNNLite). The cloud simulator's predictions come from
convert_rknn.py, which calls run() here with the model it has just built.

One JSON line per frame: {"id", "dets": [{"cls", "conf", "xyxy"}], "ms"}. Boxes are kept down to
conf 0.001 (Ultralytics' val threshold) so AP sees the whole ranking; ms is the detector call on
this machine, not a deployment number (that is 2F).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from microduck_toy.paths import repo_root
from microduck_toy.vision import detect

AP_CONF = 0.001


def make(kind: str, model: Path | None, cfg: dict):
    if kind == 'onnx':
        return detect.OnnxYolo(model, iou=float(cfg['detect']['iou']))
    if kind == 'onnx-split':
        return detect.OnnxYoloSplit(model, iou=float(cfg['detect']['iou']))
    if kind == 'rknn':
        return detect.RknnYolo.board(model, cfg['classes'], iou=float(cfg['detect']['iou']))
    if kind == 'colour':
        return detect.ColourBoxes(cfg['baselines']['colour'])
    raise ValueError(kind)


def run(data: Path, split: str, det, out: Path, only: set | None = None) -> dict:
    ids = [json.loads(x)['id'] for x in (data / 'meta' / f'{split}.jsonl').read_text().splitlines()]
    if only is not None:
        ids = [i for i in ids if i in only]
    n_det = 0
    with open(out, 'w') as f:
        for fid in ids:
            img = cv2.imread(str(data / 'images' / split / f'{fid}.png'))
            t0 = time.perf_counter()
            dets = det(img, conf=AP_CONF)
            ms = (time.perf_counter() - t0) * 1000
            n_det += len(dets)
            for d in dets:
                d['conf'] = round(d['conf'], 5)
                d['xyxy'] = [round(v, 2) for v in d['xyxy']]
            f.write(json.dumps({'id': fid, 'dets': dets, 'ms': round(ms, 2)}) + '\n')
    return {'frames': len(ids), 'boxes': n_det}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--detector', choices=('onnx', 'onnx-split', 'rknn', 'colour'), required=True)
    ap.add_argument('--model', type=Path)
    ap.add_argument('--ids', type=Path, help='only these frame ids (one per line)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = json.loads((repo_root() / 'configs/vision/stage2.json').read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    only = set(args.ids.read_text().split()) if args.ids else None
    print(run(args.data, args.split, make(args.detector, args.model, cfg), args.out, only))
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
