"""Plan 2D steps 3-4: convert the split ONNX to RKNN (FP16 or INT8) for RK3588 and check it in
rknn-toolkit2's simulator; optionally write the simulator's boxes for stage2.evaluate.

    PYTHONPATH=src $RKNN_PY scripts/vision/stage2/convert_rknn.py \
        --onnx runs/<UTC>_s2d_export/yolo11n_toys_split_320.onnx \
        --data runs/<UTC>_s2a_dataset --mode int8 [--algo mmse] [--predict test,test_far]

$RKNN_PY is a Python 3.11 with rknn-toolkit2 2.3.2 (x86 Linux only). Writes a new
runs/<UTC>_s2d_rknn_<mode>[_<algo>]/ with
  yolo11n_toys_<mode>.rknn   the file the board loads
  calib.txt                  INT8 only: 300 train frames that show at least one target toy, drawn
                             per (view, floor) in proportion to the pool (plan 2D step 3)
  accuracy/                  accuracy_analysis on one search and one approach frame: per-layer
                             cosine of the simulator against float (plan 2D step 4)
  sim_check.json             simulator against ONNX Runtime on --check-n test frames: cosine per
                             output pooled over the frames (a frame with no object near a cell
                             has an all-but-zero cls map, whose own cosine means nothing), max
                             error per output, boxes at conf 0.25
  pred_<split>.jsonl         --predict: the simulator's boxes, the format of stage2.predict
  conversion.json            sha256 of the ONNX and the RKNN, config, versions, the above summaries
  toolkit.log                the toolkit's own log
  check*.onnx                the toolkit's intermediate graphs (it writes them to the working
                             directory, so the script runs it from this folder)

Config as in rknn_model_zoo examples/yolo11/python/convert.py: mean 0, std 255, target rk3588.
INT8 is the toolkit default w8a8 per channel; --algo picks normal (default), mmse or kl_divergence
(the plan's INT8 fallback order). The toolkit reads the calibration PNGs as RGB, like the model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import onnx

from microduck_toy.paths import repo_root
from microduck_toy.stage2 import predict
from microduck_toy.stage2.check_tools import pair_boxes
from microduck_toy.vision import detect

CALIB_N = 300
CALIB_SEED = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(stage: str, code) -> None:
    if code != 0:
        raise RuntimeError(f'{stage} failed: return code {code}')


def calibration(data: Path, targets: list[str], n: int = CALIB_N, seed: int = CALIB_SEED):
    """n train frames with at least one target toy, (view, floor) strata in proportion."""
    metas = [json.loads(x) for x in (data / 'meta' / 'train.jsonl').read_text().splitlines()]
    pool = defaultdict(list)
    for m in metas:
        if any(o['name'] in targets for o in m['objects']):
            pool[(m['view'], m['floor'])].append(m['id'])
    total = sum(len(v) for v in pool.values())
    keys = sorted(pool)
    exact = {k: n * len(pool[k]) / total for k in keys}
    quota = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: -(exact[k] - quota[k]))[:n - sum(quota.values())]:
        quota[k] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for k in keys:
        chosen += sorted(rng.choice(pool[k], quota[k], replace=False).tolist())
    return chosen, {f'{k[0]}/{k[1]}': {'pool': len(pool[k]), 'chosen': quota[k]} for k in keys}


def sim_check(sim: detect.RknnYolo, ort: detect.OnnxYoloSplit, paths: list[Path], conf: float):
    names = [o.name for o in ort.session.get_outputs()]
    dots = {k: np.zeros(3) for k in names}   # a.b, a.a, b.b summed over the frames
    err = {k: 0.0 for k in names}
    boxes = {'onnx': 0, 'rknn': 0, 'pairs': 0, 'worst_box_px': 0.0, 'worst_conf': 0.0}
    for p in paths:
        img = cv2.imread(str(p))
        a, b = ort.raw(img), sim.raw(img)
        for k, x, y in zip(names, a, b):
            x, y = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
            dots[k] += (x @ y, x @ x, y @ y)
            err[k] = max(err[k], float(np.abs(x - y).max()))
        da = detect.decode_split(a, conf, ort.iou)
        db = detect.decode_split(b, conf, sim.iou)
        r = pair_boxes(da, db)
        boxes['onnx'] += len(da[0])
        boxes['rknn'] += len(db[0])
        boxes['pairs'] += r['pairs']
        boxes['worst_box_px'] = max(boxes['worst_box_px'], r['box_px'])
        boxes['worst_conf'] = max(boxes['worst_conf'], r['conf'])
    return {'frames': len(paths), 'conf': conf,
            'cosine': {k: round(float(d[0] / np.sqrt(d[1] * d[2] + 1e-24)), 6)
                       for k, d in dots.items()},
            'max_abs_err': {k: round(v, 5) for k, v in err.items()}, 'boxes': boxes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--onnx', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--mode', choices=('fp16', 'int8'), required=True)
    ap.add_argument('--algo', choices=('normal', 'mmse', 'kl_divergence'), default='normal')
    ap.add_argument('--check-n', type=int, default=100)
    ap.add_argument('--predict', default='', help='comma-separated splits, e.g. test,test_far')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    args.onnx, args.data = args.onnx.resolve(), args.data.resolve()
    from rknn.api import RKNN

    root = repo_root()
    cfg = json.loads((root / 'configs/vision/stage2.json').read_text())
    ort = detect.OnnxYoloSplit(args.onnx, iou=float(cfg['detect']['iou']))
    if ort.names != cfg['classes']:
        raise SystemExit(f'ONNX class names {ort.names} differ from the config {cfg["classes"]}')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    tag = args.mode + ('' if args.algo == 'normal' or args.mode == 'fp16' else f'_{args.algo}')
    folder = (args.out or root / 'runs' / f'{stamp}_s2d_rknn_{tag}').resolve()
    folder.mkdir(parents=True)
    rknn_path = folder / f'yolo11n_toys_{tag}.rknn'
    config = {'mean_values': [[0, 0, 0]], 'std_values': [[255, 255, 255]],
              'target_platform': 'rk3588'}
    calib_info = None
    if args.mode == 'int8':
        config.update(quantized_dtype='w8a8', quantized_method='channel',
                      quantized_algorithm=args.algo)
        ids, calib_info = calibration(args.data, cfg['targets'])
        (folder / 'calib.txt').write_text(
            ''.join(f'{(args.data / "images" / "train" / f"{i}.png").resolve()}\n' for i in ids))
        calib_info = {'frames': len(ids), 'seed': CALIB_SEED, 'strata': calib_info,
                      'sha256': sha256(folder / 'calib.txt')}

    cwd = Path.cwd()
    os.chdir(folder)
    rknn = RKNN(verbose=True, verbose_file=str(folder / 'toolkit.log'))
    try:
        rknn.config(**config)
        check('load_onnx', rknn.load_onnx(model=str(args.onnx)))
        check('build', rknn.build(do_quantization=args.mode == 'int8',
                                  dataset=str(folder / 'calib.txt') if calib_info else None))
        check('export_rknn', rknn.export_rknn(str(rknn_path)))
        meta = [json.loads(x) for x in (args.data / 'meta' / 'test.jsonl').read_text().splitlines()]
        test_png = lambda m: args.data / 'images' / 'test' / f'{m["id"]}.png'
        probe = [test_png(next(m for m in meta if m['view'] == v)) for v in ('search', 'approach')]
        check('accuracy_analysis', rknn.accuracy_analysis(
            inputs=[str(probe[0])], output_dir=str(folder / 'accuracy' / 'search')))
        check('accuracy_analysis', rknn.accuracy_analysis(
            inputs=[str(probe[1])], output_dir=str(folder / 'accuracy' / 'approach')))
        check('init_runtime', rknn.init_runtime())
        sim = detect.RknnYolo(rknn, cfg['classes'], iou=float(cfg['detect']['iou']))
        checked = sim_check(sim, ort, [test_png(m) for m in meta[:args.check_n]],
                            float(cfg['detect']['conf']))
        (folder / 'sim_check.json').write_text(json.dumps(checked, indent=1) + '\n')
        print(json.dumps(checked['cosine']), json.dumps(checked['boxes']), flush=True)
        preds = {}
        for split in filter(None, args.predict.split(',')):
            preds[split] = predict.run(args.data, split, sim, folder / f'pred_{split}.jsonl')
            print(split, preds[split], flush=True)
    finally:
        rknn.release()
        os.chdir(cwd)

    onnx_model = onnx.load(str(args.onnx))
    report = {
        'stamp': stamp, 'mode': args.mode, 'algo': args.algo if args.mode == 'int8' else None,
        'onnx': str(args.onnx), 'onnx_sha256': sha256(args.onnx),
        'source_pt_sha256': {p.key: p.value for p in onnx_model.metadata_props}.get('source_pt_sha256'),
        'rknn': rknn_path.name, 'rknn_sha256': sha256(rknn_path),
        'rknn_bytes': rknn_path.stat().st_size, 'config': config, 'calibration': calib_info,
        'names': cfg['classes'], 'sim_check': checked, 'predictions': preds,
        'scope': 'cloud simulator; the board run (init_runtime + inference on the NPU) is plan 2E/2F',
        'python': sys.version.split()[0],
        'versions': {n: version(n) for n in ('rknn-toolkit2', 'onnx', 'onnxruntime', 'numpy', 'torch')},
    }
    (folder / 'conversion.json').write_text(json.dumps(report, indent=1) + '\n')
    print(folder)
    return 0


if __name__ == '__main__':
    sys.exit(main())
