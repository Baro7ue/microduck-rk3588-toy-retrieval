# Moved 2026-10-11 from work/probe_board_yolo11_npu_camera.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Bounded camera-only YOLO11n-320 RKNN inference; no body or robot RPC."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import cv2
import numpy as np

from microduck_toy.paths import board_root
from microduck_toy.vision.cpu_npu_compare import npu_pixels, validate_output
from microduck_toy.vision.yolo_postprocess import (letterbox, decode_predictions, class_aware_nms,
                                 restore_boxes)
from microduck_toy.vision.yolo_camera_probe import decode_frame, main as run_camera

ONNX_SHA256 = '121662c804b9aa906563235ed2909815f2c23bc6e4052e651e01468817551826'
RKNN_SHA256 = '72b202f6b6cebb12f60bd1ae2dceffa706ba19cc66c93955e483645ef8c26a02'


class NPUCameraDetector:
    def __init__(self, model_path: Path, include_orange=False):
        self.class_ids = (0, 32, 49) if include_orange else (0, 32)
        model_path = Path(model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        conversion = json.loads((model_path.parent / 'conversion.json').read_text())
        if digest != RKNN_SHA256 or conversion['rknn_sha256'] != digest \
                or conversion['onnx_sha256'] != ONNX_SHA256 \
                or conversion['status'] != 'converted' \
                or conversion['target_platform'] != 'rk3588' \
                or conversion['do_quantization'] is not False \
                or conversion['mean_values'] != [[0, 0, 0]] \
                or conversion['std_values'] != [[255, 255, 255]]:
            raise ValueError('RKNN model must match the verified rk3588 nonquantized std255 conversion')
        self.model = {'path': str(model_path.resolve()), 'sha256': digest,
                      'onnx_sha256': ONNX_SHA256, 'conversion': conversion,
                      'input_shape': [1, 320, 320, 3], 'input_dtype': 'uint8',
                      'input_layout': 'RGB NHWC uint8; compiler applies std255',
                      'output_shape': [1, 84, 2100], 'output_dtype': 'float32',
                      'names': {0: 'person', 32: 'sports ball'}}
        if include_orange:
            self.model['names'][49] = 'orange'
        from rknnlite.api import RKNNLite
        self.runtime = RKNNLite()
        try:
            code = self.runtime.load_rknn(str(model_path))
            if code != 0:
                raise RuntimeError(f'load_rknn returned {code}')
            code = self.runtime.init_runtime()
            if code != 0:
                raise RuntimeError(f'init_runtime returned {code}')
            self.runtime_info = {'backend': 'npu', 'runtime': 'RKNNLite',
                                 'package': 'rknn-toolkit-lite2',
                                 'version': version('rknn-toolkit-lite2'),
                                 'sdk_version': repr(self.runtime.get_sdk_version())}
            zero = np.zeros((1, 320, 320, 3), dtype=np.uint8)
            for _ in range(5):
                validate_output(self.runtime.inference(
                    inputs=[zero], data_format=['nhwc'], inputs_pass_through=[0]))
        except BaseException:
            self.close()
            raise

    def process(self, payload):
        began = time.perf_counter_ns()
        image = decode_frame(payload)
        tensor, gain, padding = letterbox(image)
        # The shared CPU letterbox returns /255 floats; RKNN's compiler normalizes uint8.
        pixels = npu_pixels(tensor)
        infer_started = time.perf_counter_ns()
        result = self.runtime.inference(inputs=[pixels], data_format=['nhwc'],
                                        inputs_pass_through=[0])
        inference_ms = (time.perf_counter_ns() - infer_started) / 1_000_000
        raw = validate_output(result)
        boxes, scores, classes = decode_predictions(raw, 0.25, class_ids=self.class_ids)
        keep = class_aware_nms(boxes, scores, classes, 0.45)
        boxes = restore_boxes(boxes[keep], image.shape[:2], gain, padding)
        detections = []
        for box, score, class_id in zip(boxes, scores[keep], classes[keep]):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            name = self.model['names'][int(class_id)]
            detections.append({'class_id': int(class_id), 'class_name': name,
                               'confidence': float(score), 'xyxy': box.tolist()})
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(image, f'{name} {score:.3f}', (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        return {'inference_call_ms': inference_ms,
                'complete_processing_ms': (time.perf_counter_ns() - began) / 1_000_000,
                'detections': detections}, image

    def close(self):
        if self.runtime is not None:
            runtime, self.runtime = self.runtime, None
            runtime.release()


def main(duration=30.0, address=('192.168.50.1', 7901), root=None, model_path=None,
         directory=None):
    model_path = Path(model_path or board_root() /
                      'yolo11n-npu-camera/models/yolo11n-rk3588-fp.rknn')
    return run_camera(duration=duration, address=address, root=root, model_path=model_path,
                      directory=directory, detector_factory=NPUCameraDetector, backend='npu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=None)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(main(duration=args.duration, model_path=args.model, directory=args.output_dir))
