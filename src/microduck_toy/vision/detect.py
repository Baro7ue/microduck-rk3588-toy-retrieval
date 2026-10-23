"""Stage 2 detectors: one 320x320 input image in, boxes in input pixels out.

    det = OnnxYolo('model.onnx')              # Ultralytics single-output ONNX [1, 4+nc, 2100]
    det = ColourBoxes(cfg['colour_baseline'])  # HSV threshold + connected components (plan 2B)
    for d in det(img320_bgr, conf=0.25):       # {'cls': 'bus', 'conf': 0.91, 'xyxy': [x0, y0, x1, y1]}
        frame_box = views.to_frame_box(d['xyxy'], mapping)

The input is what views.to_input makes (window or letterbox, BGR, uint8). Class names come from the
model's own metadata (COCO names for the pretrained model, our 6 names after fine-tuning). The
split-output decoder for RKNN (plan 2D) is added here when the export exists.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import numpy as np

from microduck_toy.vision import views
from microduck_toy.vision.yolo_postprocess import class_aware_nms

MAX_CANDIDATES = 1000  # pre-NMS cap, highest scores first (Ultralytics caps at 30000)
MAX_DET = 100          # per image after NMS (Ultralytics val uses 300; we never have > 10 objects)


def decode_single(raw: np.ndarray, conf: float, iou: float, max_det: int = MAX_DET):
    """Ultralytics single output [1, 4+nc, N] (cx, cy, w, h, class scores) -> boxes, scores, classes.

    One class per anchor (the best one), class-aware NMS, boxes as xyxy in input pixels."""
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] < 5 or not np.isfinite(raw).all():
        raise ValueError(f'expected finite [1, 4+nc, N], got {raw.shape}')
    cand = raw[0].T
    classes = cand[:, 4:].argmax(axis=1)
    scores = cand[np.arange(len(cand)), 4 + classes]
    keep = np.nonzero(scores > conf)[0]
    if len(keep) > MAX_CANDIDATES:
        keep = keep[np.argsort(-scores[keep], kind='stable')[:MAX_CANDIDATES]]
    xywh = cand[keep, :4].astype(np.float64)
    boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    scores, classes = scores[keep].astype(np.float64), classes[keep]
    order = class_aware_nms(boxes, scores, classes, iou)[:max_det]
    boxes = boxes[order].clip(0, views.INPUT)
    return boxes, scores[order], classes[order]


def model_names(session) -> list[str]:
    encoded = session.get_modelmeta().custom_metadata_map.get('names', '')
    try:
        names = json.loads(encoded)
    except json.JSONDecodeError:
        names = ast.literal_eval(encoded)
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


class OnnxYolo:
    """CPU ONNX Runtime, one fixed [1,3,320,320] float input, one [1, 4+nc, 2100] output."""

    def __init__(self, path: Path, threads: int = 2, iou: float = 0.45):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=['CPUExecutionProvider'])
        inp, = self.session.get_inputs()
        out, = self.session.get_outputs()
        if list(inp.shape) != [1, 3, views.INPUT, views.INPUT]:
            raise ValueError(f'expected input [1,3,320,320], got {inp.shape}')
        self.input, self.output = inp.name, out.name
        self.names = model_names(self.session)
        if out.shape[1] != 4 + len(self.names):
            raise ValueError(f'output {out.shape} does not match {len(self.names)} class names')
        self.iou = iou

    def raw(self, img320: np.ndarray) -> np.ndarray:
        return self.session.run([self.output], {self.input: views.tensor(img320)})[0]

    def __call__(self, img320: np.ndarray, conf: float = 0.25) -> list[dict]:
        boxes, scores, classes = decode_single(self.raw(img320), conf, self.iou)
        return [{'cls': self.names[int(c)], 'conf': float(s), 'xyxy': [float(v) for v in b]}
                for b, s, c in zip(boxes, scores, classes)]


class ColourBoxes:
    """Colour-threshold baseline: for each colour class, pixels inside its HSV ranges, then every
    8-connected blob of at least min_area_px pixels becomes one box.

    HSV is OpenCV's 8-bit convention (H 0-180, S and V 0-255), as in Stage 1's colour.py. The score
    grows with the blob's pixel count, area / (area + score_half_px), so that bigger blobs rank
    first when AP is computed; it is not a probability."""

    def __init__(self, cfg: dict):
        self.classes = {name: {'ranges': [(np.array(lo, np.uint8), np.array(hi, np.uint8))
                                          for lo, hi in c['ranges_hsv']],
                               'min_area': int(c['min_area_px'])}
                        for name, c in cfg['classes'].items()}
        self.half = float(cfg.get('score_half_px', 20.0))
        self.names = list(self.classes)

    @staticmethod
    def hsv(img320: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img320, cv2.COLOR_BGR2HSV)

    def mask(self, hsv: np.ndarray, name: str) -> np.ndarray:
        out = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self.classes[name]['ranges']:
            out |= cv2.inRange(hsv, lo, hi)
        return out

    def from_hsv(self, hsv: np.ndarray, conf: float = 0.0) -> list[dict]:
        dets = []
        for name, c in self.classes.items():
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(self.mask(hsv, name), connectivity=8)
            for x, y, w, h, area in stats[1:]:
                score = area / (area + self.half)
                if area >= c['min_area'] and score > conf:
                    dets.append({'cls': name, 'conf': float(score),
                                 'xyxy': [float(x), float(y), float(x + w), float(y + h)]})
        return sorted(dets, key=lambda d: -d['conf'])[:MAX_DET]

    def __call__(self, img320: np.ndarray, conf: float = 0.25) -> list[dict]:
        return self.from_hsv(self.hsv(img320), conf)
