"""Stage 2 detectors: one 320x320 input image in, boxes in input pixels out.

    det = OnnxYolo('model.onnx')              # Ultralytics single-output ONNX [1, 4+nc, 2100]
    det = OnnxYoloSplit('model_split.onnx')   # our 9-output export (export_split_onnx.py, plan 2D)
    det = RknnYolo.board('model.rknn', names) # the same 9 outputs on the board's NPU (RKNNLite)
    det = ColourBoxes(cfg['colour_baseline'])  # HSV threshold + connected components (plan 2B)
    for d in det(img320_bgr, conf=0.25):       # {'cls': 'bus', 'conf': 0.91, 'xyxy': [x0, y0, x1, y1]}
        frame_box = views.to_frame_box(d['xyxy'], mapping)

The input is what views.to_input makes (window or letterbox, BGR, uint8). Class names come from the
model's own metadata (COCO names for the pretrained model, our 6 names after fine-tuning).
decode_split is the one post-processing for the split ONNX on the CPU and for both RKNN files on
the NPU (same 9 arrays); the board's C++ benchmark (plan 2H) does the same steps.
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
STRIDES = (8, 16, 32)  # split outputs come in this order: box, cls, sum for each stride
REG_MAX = 16           # DFL bins per box side
SUM_MARGIN = 0.01      # a cell is skipped when its score sum < conf - SUM_MARGIN (INT8 rounding)


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


def decode_split(outs, conf: float, iou: float, max_det: int = MAX_DET):
    """Split outputs (box [1,64,H,W], cls [1,nc,H,W], sum [1,1,H,W] for stride 8, 16, 32) ->
    boxes, scores, classes; the same result as decode_single on Ultralytics' own export.

    1. skip cells whose score sum is below the threshold (the max class score can't be above it);
    2. best class per cell, keep it when its score > conf;
    3. DFL: per box side, softmax over the 16 bins, the expected bin = distance in grid cells;
    4. anchor = cell centre (x + 0.5, y + 0.5); x0 = (ax - left) * stride, x1 = (ax + right) * stride;
    5. class-aware NMS, boxes clipped to the 320 input."""
    if len(outs) != 3 * len(STRIDES):
        raise ValueError(f'expected {3 * len(STRIDES)} outputs, got {len(outs)}')
    boxes, scores, classes = [], [], []
    for k, stride in enumerate(STRIDES):
        box, cls, ssum = (np.asarray(o, np.float32) for o in outs[3 * k:3 * k + 3])
        h, w = cls.shape[2:]
        if box.shape != (1, 4 * REG_MAX, h, w) or ssum.shape != (1, 1, h, w) or h * stride != views.INPUT:
            raise ValueError(f'stride {stride}: box {box.shape} cls {cls.shape} sum {ssum.shape}')
        ys, xs = np.nonzero(ssum[0, 0] >= conf - SUM_MARGIN)
        if not len(ys):
            continue
        cell = cls[0][:, ys, xs]                                  # [nc, n]
        best = cell.argmax(axis=0)
        score = cell[best, np.arange(len(best))]
        ok = score > conf
        ys, xs, best, score = ys[ok], xs[ok], best[ok], score[ok]
        if not len(ys):
            continue
        logits = box[0][:, ys, xs].reshape(4, REG_MAX, -1).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        dist = (prob * np.arange(REG_MAX)[None, :, None]).sum(axis=1)  # [4, n]: left top right bottom
        ax, ay = xs + 0.5, ys + 0.5
        boxes.append(np.stack((ax - dist[0], ay - dist[1], ax + dist[2], ay + dist[3]), 1) * stride)
        scores.append(score.astype(np.float64))
        classes.append(best)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    boxes, scores, classes = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    if len(scores) > MAX_CANDIDATES:
        top = np.argsort(-scores, kind='stable')[:MAX_CANDIDATES]
        boxes, scores, classes = boxes[top], scores[top], classes[top]
    order = class_aware_nms(boxes, scores, classes, iou)[:max_det]
    return boxes[order].clip(0, views.INPUT), scores[order], classes[order]


def to_dicts(names, boxes, scores, classes) -> list[dict]:
    return [{'cls': names[int(c)], 'conf': float(s), 'xyxy': [float(v) for v in b]}
            for b, s, c in zip(boxes, scores, classes)]


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
        return to_dicts(self.names, *decode_single(self.raw(img320), conf, self.iou))


class OnnxYoloSplit:
    """CPU ONNX Runtime on the split export: one [1,3,320,320] float input, 9 outputs."""

    def __init__(self, path: Path, threads: int = 2, iou: float = 0.45):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=['CPUExecutionProvider'])
        inp, = self.session.get_inputs()
        if list(inp.shape) != [1, 3, views.INPUT, views.INPUT]:
            raise ValueError(f'expected input [1,3,320,320], got {inp.shape}')
        self.input = inp.name
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.names = model_names(self.session)
        if len(self.outputs) != 3 * len(STRIDES):
            raise ValueError(f'expected 9 outputs, got {self.outputs}')
        self.iou = iou

    def raw(self, img320: np.ndarray) -> list[np.ndarray]:
        return self.session.run(self.outputs, {self.input: views.tensor(img320)})

    def __call__(self, img320: np.ndarray, conf: float = 0.25) -> list[dict]:
        return to_dicts(self.names, *decode_split(self.raw(img320), conf, self.iou))


class RknnYolo:
    """The split model on the NPU. runtime is anything with RKNN's inference(): RKNNLite on the
    board (RknnYolo.board), or rknn-toolkit2's simulator in the process that built the model
    (convert_rknn.py; the simulator cannot load a finished .rknn). The input is RGB uint8 NHWC;
    mean 0 / std 255 were compiled into the model, as in Stage 0 (cpu_npu_compare.npu_pixels).
    An .rknn file carries no class names, so they come from configs/vision/stage2.json."""

    def __init__(self, runtime, names: list[str], iou: float = 0.45):
        self.runtime, self.names, self.iou = runtime, list(names), iou

    @classmethod
    def board(cls, path: Path, names: list[str], iou: float = 0.45) -> 'RknnYolo':
        from rknnlite.api import RKNNLite
        runtime = RKNNLite()
        if runtime.load_rknn(str(path)) != 0 or runtime.init_runtime() != 0:
            runtime.release()
            raise RuntimeError(f'cannot load {path} on the NPU')
        return cls(runtime, names, iou)

    def raw(self, img320: np.ndarray) -> list[np.ndarray]:
        rgb = np.ascontiguousarray(img320[None, :, :, ::-1])
        outs = self.runtime.inference(inputs=[rgb], data_format=['nhwc'])
        if outs is None or len(outs) != 3 * len(STRIDES):
            raise RuntimeError(f'RKNN returned {None if outs is None else len(outs)} outputs')
        return outs

    def __call__(self, img320: np.ndarray, conf: float = 0.25) -> list[dict]:
        return to_dicts(self.names, *decode_split(self.raw(img320), conf, self.iou))

    def close(self) -> None:
        if self.runtime is not None:
            runtime, self.runtime = self.runtime, None
            runtime.release()


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
