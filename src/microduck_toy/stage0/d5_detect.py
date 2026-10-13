"""Stage 0 d5: run YOLO11n on the cloud's rendered toy frames and check that the numbers match.

The cloud rendered the five chosen toys from the robot's head camera (plan 0d) and saved every frame
exactly as the detector sees it (upright BGR, 360 wide x 640 high, PNG). This runs the same detector
path on those frames on the Mac (CPU, FP32 ONNX) or on the board (NPU, FP16 RKNN) and compares the
result with the cloud's own run of this module:

    run      --frames PACK --backend cpu|npu --model FILE --output-dir DIR [--distances 0.5,1.2]
    compare  --frames PACK --result DIR
    subset   --frames PACK --distances 0.5 --output-dir DIR     (a smaller pack to send to the board)

PACK = runs/<UTC>_d5_cloud_pack/: pack.json, views.jsonl (one line per frame: toy, head pose,
distance, heading, true box, ground band), frames/*.png and cloud/ (the cloud's `run` output).
Every frame is detected whole and as three horizon-band tiles (screen.py in the cloud), all 80 COCO
classes, confidence 0.25, IoU 0.45.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np

from microduck_toy.vision.yolo_postprocess import (validate_model, letterbox, decode_predictions,
                                                   class_aware_nms, restore_boxes)

ONNX_SHA256 = '121662c804b9aa906563235ed2909815f2c23bc6e4052e651e01468817551826'
CONFIDENCE, IOU = 0.25, 0.45
TILE_W, TILE_X = 180, (0, 90, 180)
MODES = ('whole', 'band')
# Pass rules: the Mac CPU must give the cloud's counts exactly and the same boxes; the board's FP16
# may move a score across 0.25, so its on-toy counts may differ by 2 frames in 40 (5 pp).
MATCH_IOU, CPU_CONF_TOL, CPU_AGREE, NPU_COUNT_TOL = 0.9, 0.01, 0.95, 2


class CpuBackend:
    def __init__(self, model_path: Path):
        import onnxruntime as ort
        data = model_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != ONNX_SHA256:
            raise ValueError(f'{model_path} is not the project YOLO11n ONNX ({ONNX_SHA256[:8]}…)')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(data, sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.model = validate_model(self.session)
        self.info = {'backend': 'cpu', 'model': str(model_path), 'sha256': ONNX_SHA256,
                     'onnxruntime': version('onnxruntime'), 'threads': 2}

    def raw(self, tensor: np.ndarray) -> np.ndarray:
        raw, = self.session.run([self.model['output_name']], {self.model['input_name']: tensor})
        return raw

    def close(self) -> None:
        pass


class NpuBackend:
    def __init__(self, model_path: Path):
        from rknnlite.api import RKNNLite
        from microduck_toy.vision.rknn_runner import RKNN_SHA256
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        conversion = json.loads((model_path.parent / 'conversion.json').read_text())
        if digest != RKNN_SHA256 or conversion['rknn_sha256'] != digest \
                or conversion['onnx_sha256'] != ONNX_SHA256 or conversion['do_quantization']:
            raise ValueError('RKNN model must be the verified rk3588 FP16 conversion of the ONNX')
        self.runtime = RKNNLite()
        if self.runtime.load_rknn(str(model_path)) != 0 or self.runtime.init_runtime() != 0:
            self.close()
            raise RuntimeError(f'cannot load {model_path} on the NPU')
        self.info = {'backend': 'npu', 'model': str(model_path), 'sha256': digest,
                     'rknn-toolkit-lite2': version('rknn-toolkit-lite2'),
                     'sdk': repr(self.runtime.get_sdk_version())}

    def raw(self, tensor: np.ndarray) -> np.ndarray:
        from microduck_toy.vision.cpu_npu_compare import npu_pixels, validate_output
        return validate_output(self.runtime.inference(
            inputs=[npu_pixels(tensor)], data_format=['nhwc'], inputs_pass_through=[0]))

    def close(self) -> None:
        if getattr(self, 'runtime', None) is not None:
            runtime, self.runtime = self.runtime, None
            runtime.release()


def detect(backend, image: np.ndarray, names: list[str]) -> list[dict]:
    tensor, gain, padding = letterbox(image)
    boxes, scores, classes = decode_predictions(backend.raw(tensor), CONFIDENCE,
                                                class_ids=tuple(range(len(names))))
    keep = class_aware_nms(boxes, scores, classes, IOU)
    boxes = restore_boxes(boxes[keep], image.shape[:2], gain, padding)
    return [{'cls': names[int(cls)], 'conf': float(score), 'xyxy': [float(x) for x in box]}
            for box, score, cls in zip(boxes, scores[keep], classes[keep])
            if box[2] > box[0] and box[3] > box[1]]


def detect_band(backend, image: np.ndarray, band, names: list[str]) -> list[dict]:
    """Three overlapping 180-px-wide tiles of the ground band; boxes back in frame coordinates."""
    top, bottom = band
    found = []
    for x0 in TILE_X:
        tile = np.ascontiguousarray(image[top:bottom, x0:x0 + TILE_W])
        for det in detect(backend, tile, names):
            b = det['xyxy']
            det['xyxy'] = [b[0] + x0, b[1] + top, b[2] + x0, b[3] + top]
            found.append(det)
    return found


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def on_toy(det: dict, gt) -> bool:
    """IoU >= 0.3 with the true box, or centre inside it (grown by max(3 px, 25%)) and not over 4x
    its area; the rule the cloud screening used."""
    if not gt:
        return False
    box = det['xyxy']
    if iou(box, gt) >= 0.3:
        return True
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    mx, my = max(3, 0.25 * w), max(3, 0.25 * h)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    area = (box[2] - box[0]) * (box[3] - box[1])
    return gt[0] - mx <= cx <= gt[2] + mx and gt[1] - my <= cy <= gt[3] + my and area <= 4 * w * h


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_distances(text: str | None):
    return tuple(float(x) for x in text.split(',')) if text else None


def run(args) -> int:
    pack = json.loads((args.frames / 'pack.json').read_text())
    views = read_jsonl(args.frames / 'views.jsonl')
    wanted = parse_distances(args.distances)
    if wanted:
        views = [v for v in views if v['d'] in wanted]
    if not views:
        print('没有要检测的画面（检查 --distances）。', file=sys.stderr)
        return 1
    backend = (CpuBackend if args.backend == 'cpu' else NpuBackend)(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with open(args.output_dir / 'views.jsonl', 'w') as out:
            for i, view in enumerate(views, 1):
                path = args.frames / view['frame']
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None or image.shape != (640, 360, 3):
                    raise ValueError(f'{path}: expected a 360x640 BGR PNG')
                out.write(json.dumps({'frame': view['frame'],
                                      'whole': detect(backend, image, pack['names']),
                                      'band': detect_band(backend, image, view['band'], pack['names'])})
                          + '\n')
                if i % 20 == 0:
                    print(f'{i}/{len(views)}', flush=True)
    finally:
        backend.close()
    summary = {'pack': pack['name'], 'frames': len(views), 'distances': sorted({v['d'] for v in views}),
               'seconds': round(time.perf_counter() - started, 1), **backend.info,
               'host': platform.node(), 'machine': platform.machine(),
               'python': platform.python_version(), 'numpy': np.__version__,
               'opencv': cv2.__version__, 'confidence': CONFIDENCE, 'iou': IOU}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'检测完成：{len(views)} 张画面，{summary["seconds"]} s，结果在 {args.output_dir}')
    return 0


def match(expected: list[dict], actual: list[dict]) -> dict:
    """Greedy one-to-one matching of same-class boxes at IoU >= 0.9."""
    left = list(actual)
    pairs = []
    for det in sorted(expected, key=lambda d: -d['conf']):
        best = max((x for x in left if x['cls'] == det['cls']),
                   key=lambda x: iou(x['xyxy'], det['xyxy']), default=None)
        if best is not None and iou(best['xyxy'], det['xyxy']) >= MATCH_IOU:
            left.remove(best)
            pairs.append(abs(best['conf'] - det['conf']))
    return {'matched': len(pairs), 'only_expected': len(expected) - len(pairs),
            'only_actual': len(left), 'max_dconf': max(pairs, default=0.0)}


def compare(args) -> int:
    views = {v['frame']: v for v in read_jsonl(args.frames / 'views.jsonl')}
    cloud = {r['frame']: r for r in read_jsonl(args.frames / 'cloud/views.jsonl')}
    here = read_jsonl(args.result / 'views.jsonl')
    backend = json.loads((args.result / 'summary.json').read_text())['backend']
    counts, agree = {}, {'matched': 0, 'only_expected': 0, 'only_actual': 0, 'max_dconf': 0.0}
    for row in here:
        view, ref = views[row['frame']], cloud[row['frame']]
        for mode in MODES:
            key = (view['d'], mode)
            c = counts.setdefault(key, {'n': 0, 'cloud_any': 0, 'any': 0, 'cloud_hit': 0, 'hit': 0})
            c['n'] += 1
            for prefix, dets in (('cloud_', ref[mode]), ('', row[mode])):
                mine = [x for x in dets if on_toy(x, view['gt'])]
                c[prefix + 'any'] += bool(mine)
                c[prefix + 'hit'] += any(x['cls'] == view['coco'] for x in mine)
            m = match(ref[mode], row[mode])
            for k in ('matched', 'only_expected', 'only_actual'):
                agree[k] += m[k]
            agree['max_dconf'] = max(agree['max_dconf'], m['max_dconf'])
    total = agree['matched'] + agree['only_expected'] + agree['only_actual']
    agree['agreement'] = agree['matched'] / total if total else 1.0
    tol = 0 if backend == 'cpu' else NPU_COUNT_TOL
    counts_ok = all(abs(c['any'] - c['cloud_any']) <= tol and abs(c['hit'] - c['cloud_hit']) <= tol
                    for c in counts.values())
    boxes_ok = backend != 'cpu' or (agree['agreement'] >= CPU_AGREE
                                    and agree['max_dconf'] <= CPU_CONF_TOL)
    result = {'backend': backend, 'pass': counts_ok and boxes_ok, 'counts_ok': counts_ok,
              'boxes_ok': boxes_ok, 'count_tolerance': tol, 'detections': agree,
              'by_distance': [{'d': d, 'mode': mode, **c} for (d, mode), c in sorted(counts.items())]}
    (args.result / 'compare.json').write_text(json.dumps(result, indent=2) + '\n')
    print('距离   方式    画面  任意类别在玩具上（云端 / 这里）  叫对类别（云端 / 这里）')
    for (d, mode), c in sorted(counts.items()):
        print(f'{d:4.1f} m  {"整幅" if mode == "whole" else "裁剪"}  {c["n"]:4d}   '
              f'{c["cloud_any"]:3d} / {c["any"]:<3d}                       {c["cloud_hit"]:3d} / {c["hit"]}')
    print(f'检测框逐个对照：一致 {agree["matched"]}，只有云端有 {agree["only_expected"]}，'
          f'只有这里有 {agree["only_actual"]}，一致比例 {agree["agreement"]:.1%}，'
          f'置信度最大差 {agree["max_dconf"]:.4f}')
    print(f'结论：{"一致" if result["pass"] else "不一致"}（{backend}；计数允许差 {tol}）'
          f'，详细见 {args.result / "compare.json"}')
    return 0 if result['pass'] else 2


def subset(args) -> int:
    wanted = parse_distances(args.distances)
    views = [v for v in read_jsonl(args.frames / 'views.jsonl') if v['d'] in wanted]
    if not views:
        print('这个距离没有画面。', file=sys.stderr)
        return 1
    (args.output_dir / 'frames').mkdir(parents=True)
    (args.output_dir / 'cloud').mkdir()
    shutil.copy(args.frames / 'pack.json', args.output_dir / 'pack.json')
    frames = {v['frame'] for v in views}
    with open(args.output_dir / 'views.jsonl', 'w') as out:
        for v in views:
            shutil.copy(args.frames / v['frame'], args.output_dir / v['frame'])
            out.write(json.dumps(v) + '\n')
    with open(args.output_dir / 'cloud/views.jsonl', 'w') as out:
        for r in read_jsonl(args.frames / 'cloud/views.jsonl'):
            if r['frame'] in frames:
                out.write(json.dumps(r) + '\n')
    print(f'{len(views)} 张画面 → {args.output_dir}')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--frames', type=Path, required=True)
    r.add_argument('--backend', choices=('cpu', 'npu'), required=True)
    r.add_argument('--model', type=Path, required=True)
    r.add_argument('--output-dir', type=Path, required=True)
    r.add_argument('--distances', help='comma list, e.g. 0.5,1.2 (default: every frame)')
    c = sub.add_parser('compare')
    c.add_argument('--frames', type=Path, required=True)
    c.add_argument('--result', type=Path, required=True)
    s = sub.add_parser('subset')
    s.add_argument('--frames', type=Path, required=True)
    s.add_argument('--distances', required=True)
    s.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    return {'run': run, 'compare': compare, 'subset': subset}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
