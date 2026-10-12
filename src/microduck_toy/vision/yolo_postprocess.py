# Moved 2026-10-11 from work/detect_yolo11_image.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Offline COCO YOLO11n-320 CPU detection; reads one image and writes local results."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
import onnxruntime as ort


def validate_model(session) -> dict:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 320, 320] or inputs[0].type != 'tensor(float)':
        raise ValueError('expected one float32 input with fixed shape [1,3,320,320]')
    if len(outputs) != 1 or outputs[0].shape != [1, 84, 2100] or outputs[0].type != 'tensor(float)':
        raise ValueError('expected one raw float32 output with fixed shape [1,84,2100]')
    metadata = session.get_modelmeta().custom_metadata_map
    encoded = metadata.get('names', '')
    try:
        names = json.loads(encoded)
    except json.JSONDecodeError:
        names = ast.literal_eval(encoded)
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or any(isinstance(k, bool) or not isinstance(k, (str, int))
                                         for k in names):
        raise ValueError('model names metadata must describe 80 COCO classes')
    names = {int(k): v for k, v in names.items()}
    if set(names) != set(range(80)) or not all(isinstance(v, str) for v in names.values()) \
            or names[0] != 'person' or names[32] != 'sports ball':
        raise ValueError('expected 80 COCO names, person at 0 and sports ball at 32')
    if metadata.get('task', 'detect') != 'detect':
        raise ValueError('model metadata task must be detect')
    return {'input_name': inputs[0].name, 'input_shape': inputs[0].shape,
            'output_name': outputs[0].name, 'output_shape': outputs[0].shape,
            'tensor_type': 'float32', 'names': names, 'metadata': metadata}


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image.shape[:2]
    gain = min(320 / height, 320 / width)
    resized = (round(width * gain), round(height * gain))
    dw, dh = (320 - resized[0]) / 2, (320 - resized[1]) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    if (width, height) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor / 255.0, gain, (left, top)


def decode_predictions(raw: np.ndarray, confidence: float, class_ids=(0, 32)):
    if raw.shape != (1, 84, 2100) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('runtime output must be finite float32 [1,84,2100]')
    candidates = raw[0].T
    classes = candidates[:, 4:].argmax(axis=1)
    scores = candidates[np.arange(len(candidates)), 4 + classes]
    chosen = (scores > confidence) & np.isin(classes, class_ids)
    xywh = candidates[chosen, :4]
    boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2,
                            xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    return boxes, scores[chosen], classes[chosen]


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray,
                    classes: np.ndarray, threshold: float) -> np.ndarray:
    pending = np.argsort(-scores, kind='stable')
    keep = []
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    while len(pending):
        current, remaining = pending[0], pending[1:]
        keep.append(current)
        overlap = np.maximum(np.minimum(boxes[current, 2:], boxes[remaining, 2:]) -
                             np.maximum(boxes[current, :2], boxes[remaining, :2]), 0)
        intersection = overlap[:, 0] * overlap[:, 1]
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        pending = remaining[~((classes[remaining] == classes[current]) & (iou > threshold))]
    return np.asarray(keep, dtype=np.int64)


def restore_boxes(boxes: np.ndarray, original_shape: tuple[int, int],
                  gain: float, padding: tuple[int, int]) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])
    return boxes


def detect(model_path: Path, image_path: Path, output_dir: Path,
           confidence: float, iou: float) -> dict:
    if not np.isfinite([confidence, iou]).all() or not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError('confidence and iou must be finite values from 0 to 1')
    model_bytes, image_bytes = model_path.read_bytes(), image_path.read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot decode image: {image_path}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=['CPUExecutionProvider'])
    model = validate_model(session)
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError(f'expected CPUExecutionProvider only, got {session.get_providers()}')
    tensor, gain, padding = letterbox(image)
    feed = {model['input_name']: tensor}
    for _ in range(5):
        session.run([model['output_name']], feed)
    durations = []
    for _ in range(10):
        started = time.perf_counter_ns()
        raw, = session.run([model['output_name']], feed)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    boxes, scores, classes = decode_predictions(raw, confidence)
    keep = class_aware_nms(boxes, scores, classes, iou)
    boxes = restore_boxes(boxes[keep], image.shape[:2], gain, padding)
    detections = []
    for box, score, class_id in zip(boxes, scores[keep], classes[keep]):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append({'class_id': int(class_id), 'class_name': model['names'][int(class_id)],
                           'confidence': float(score), 'xyxy': box.tolist()})
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = (0, 180, 0) if class_id == 0 else (255, 160, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{model["names"][int(class_id)]} {score:.3f}',
                    (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    report = {
        'schema_version': 1, 'scope': 'offline single-image CPU detection; no body or robot RPC',
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'opencv': cv2.__version__, 'onnxruntime': ort.__version__},
        'model': {'path': str(model_path.resolve()), 'sha256': hashlib.sha256(model_bytes).hexdigest(), **model},
        'image': {'path': str(image_path.resolve()), 'sha256': hashlib.sha256(image_bytes).hexdigest(),
                  'decoded_bgr_shape': list(image.shape)},
        'preprocessing': {'gain': gain, 'padding_left_top': list(padding), 'fill': 114,
                          'resize': 'cv2.INTER_LINEAR', 'layout': 'RGB NCHW float32 /255'},
        'providers': session.get_providers(), 'intra_op_threads': 2,
        'confidence_threshold': confidence, 'iou_threshold': iou, 'detections': detections,
        'session_run_ms': {'warmup_count': 5, 'measured_count': 10, 'samples': durations,
                           'p50': float(np.median(durations)), 'p95': float(np.percentile(durations, 95)),
                           'max': max(durations),
                           'scope': 'session.run only; excludes preprocessing/postprocessing; not a real-time test'},
    }
    if not cv2.imwrite(str(output_dir / 'annotated.jpg'), image):
        raise OSError('failed to write annotated.jpg')
    (output_dir / 'detection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8')
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = detect(args.model, args.image, args.output_dir, args.confidence, args.iou)
    except Exception:
        error = traceback.format_exc()
        (args.output_dir / 'error.txt').write_text(error, encoding='utf-8')
        print(error, file=sys.stderr, end='', flush=True)
        return 1
    print(f'检测到 {len(report["detections"])} 个 person/sports ball 目标。', flush=True)
    print(args.output_dir.resolve() / 'detection.json', flush=True)
    print(args.output_dir.resolve() / 'annotated.jpg', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
