# Moved 2026-10-11 from work/probe_board_yolo11_camera.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Bounded camera-only detection using one latest complete UYVY frame; defaults to CPU."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import onnxruntime as ort

from microduck_toy.paths import board_root
from microduck_toy.vision.camera_stream import FrameWorker, FrameProtocolError, WIDTH, HEIGHT, interrupt
from microduck_toy.vision.yolo_postprocess import (validate_model, letterbox, decode_predictions,
                                 class_aware_nms, restore_boxes)


class LatestFrameSlot:
    """Publish metadata and immutable payload as a pair; never queue old frames."""
    def __init__(self):
        self.condition = threading.Condition()
        self.latest = None

    def publish(self, record, payload):
        with self.condition:
            self.latest = (record.copy(), payload)
            self.condition.notify()

    def take_after(self, receiver_index, timeout):
        with self.condition:
            if not self.condition.wait_for(
                    lambda: self.latest is not None and self.latest[0]['index'] > receiver_index,
                    timeout):
                return None
            return self.latest


def decode_frame(payload: bytes, width=WIDTH, height=HEIGHT) -> np.ndarray:
    """Simulator UYVY is full-range BT.601; OpenCV's limited-range YUV conversion is unsuitable."""
    if width <= 0 or width % 2 or height <= 0 or len(payload) != width * height * 2:
        raise FrameProtocolError('UYVY 负载长度或图像尺寸不匹配')
    pairs = np.frombuffer(payload, dtype=np.uint8).reshape(height, width // 2, 4)
    y = np.empty((height, width), dtype=np.float32)
    y[:, 0::2], y[:, 1::2] = pairs[:, :, 1], pairs[:, :, 3]
    u = np.repeat(pairs[:, :, 0].astype(np.float32) - 128, 2, axis=1)
    v = np.repeat(pairs[:, :, 2].astype(np.float32) - 128, 2, axis=1)
    bgr = np.stack((y + 1.772 * u, y - 0.344136 * u - 0.714136 * v,
                    y + 1.402 * v), axis=2)
    pixels = np.clip(np.rint(bgr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.rot90(pixels, k=-1))


class CameraDetector:
    def __init__(self, model_path: Path, include_orange=False):
        self.class_ids = (0, 32, 49) if include_orange else (0, 32)
        data = model_path.read_bytes()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(data, sess_options=options,
                                             providers=['CPUExecutionProvider'])
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('仅允许 CPUExecutionProvider')
        self.model = {'path': str(model_path.resolve()),
                      'sha256': hashlib.sha256(data).hexdigest(), **validate_model(self.session)}
        zero = np.zeros((1, 3, 320, 320), dtype=np.float32)
        for _ in range(5):
            self.session.run([self.model['output_name']], {self.model['input_name']: zero})

    def process(self, payload):
        began = time.perf_counter_ns()
        image = decode_frame(payload)
        tensor, gain, padding = letterbox(image)
        infer_started = time.perf_counter_ns()
        raw, = self.session.run([self.model['output_name']], {self.model['input_name']: tensor})
        inference_ms = (time.perf_counter_ns() - infer_started) / 1_000_000
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
            cv2.putText(image, f'{name} {score:.3f}', (x1, max(18, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        return {'session_run_ms': inference_ms,
                'complete_processing_ms': (time.perf_counter_ns() - began) / 1_000_000,
                'detections': detections}, image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'无法保存标框图片：{path}')


def main(duration=30.0, address=('192.168.50.1', 7901), root=None, model_path=None,
         directory=None, detector_factory=None, backend='cpu') -> int:
    if not math.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError('duration必须为0到60之间的有限秒数')
    if backend not in ('cpu', 'npu'):
        raise ValueError('backend必须为cpu或npu')
    if model_path is None:
        raise ValueError('model_path必须给出（板上模型文件）')
    inference_field = 'session_run_ms' if backend == 'cpu' else 'inference_call_ms'
    inference_name = 'session.run' if backend == 'cpu' else 'RKNNLite.inference'
    root = Path(root or board_root() / 'run')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    directory = Path(directory) if directory is not None else root / f'camera_{backend}_{stamp}'
    directory.mkdir(parents=True, exist_ok=False)
    model_path = Path(model_path)
    report = {'schema_version': 1, 'scope': f'camera-only {backend.upper()} detection; no body or robot RPC',
              'outcome': 'not_started', 'requested_duration_s': duration, 'address': list(address),
              'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                           'opencv': cv2.__version__, 'onnxruntime': ort.__version__},
              'confidence_threshold': 0.25, 'iou_threshold': 0.45,
              'warmup_count': 5, 'received_frame_count': 0, 'processed_frame_count': 0,
              'skipped_receiver_frames': 0,
              'definitions': {
                  'transport_fps': 'Complete payload count / receiver connection-to-stop elapsed; '
                                   'no source timestamps or frame IDs, not unique rendered FPS.',
                  'processing_throughput_fps': 'Processed frame count / connection-to-stop elapsed; '
                                               'includes waiting and evidence writes.',
                  'receive_to_process_wait_ms': 'Local wait from complete payload receipt to processing start; '
                                               'not source frame age or network latency.',
                  inference_field: f'Only synchronous {inference_name}; excludes decode/preprocess/postprocess; '
                                   'includes runtime/input/output overhead, not pure device compute.',
                  'complete_processing_ms': f'UYVY decode, rotation, preprocessing, {inference_name}, '
                                            'postprocessing and annotation; excludes JPEG/JSON writes.',
                  'skipped_receiver_frames': 'Locally received complete frames bypassed before consuming '
                                             'a newer frame; not source camera dropped frames.',
                  'cpu_window': 'Process CPU time / connection-to-stop wall time; '
                                '100% = one full core; includes receiver thread and frame processing, '
                                'excludes model loading/warmup and final image/summary writes.'}}
    if backend == 'cpu':
        report['intra_op_threads'] = 2
    worker = None
    detector = None
    records = []
    last_index = 0
    last_image = None
    status = 1
    print(f'证据目录：{directory}', flush=True)
    (directory / 'frames.jsonl').touch()
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        detector = (detector_factory or CameraDetector)(model_path)
        report['model'] = detector.model
        if backend == 'cpu':
            report['providers'] = detector.session.get_providers()
        else:
            report['runtime'] = detector.runtime_info
        slot = LatestFrameSlot()
        worker = FrameWorker(address, directory, on_frame=slot.publish)
        worker.start()
        next_print = 5.0
        with (directory / 'detections.jsonl').open('w') as stream:
            while True:
                worker.raise_if_failed()
                connected = worker.started_monotonic_s
                if connected is None:
                    time.sleep(0.01)
                    continue
                remaining = connected + duration - time.monotonic()
                if remaining <= 0:
                    if not records:
                        raise RuntimeError('本轮没有完成任何完整帧检测')
                    report['outcome'], status = 'completed', 0
                    break
                latest = slot.take_after(last_index, min(0.1, remaining))
                worker.raise_if_failed()
                if latest is None or time.monotonic() >= connected + duration:
                    continue
                frame, payload = latest
                processing_started = time.monotonic()
                detection, image = detector.process(payload)
                skipped = frame['index'] - last_index - 1
                record = {'receiver_index': frame['index'], 'payload_sha256': frame['payload_sha256'],
                          'received_complete_monotonic_s': frame['arrived_monotonic_s'],
                          'processing_started_monotonic_s': processing_started,
                          'processed_monotonic_s': time.monotonic(),
                          'receive_to_process_wait_ms': 1000 * (processing_started - frame['arrived_monotonic_s']),
                          'skipped_receiver_frames': skipped, **detection}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                if last_image is None:
                    save_image(directory / 'first_annotated.jpg', image)
                last_image, last_index = image, frame['index']
                elapsed = time.monotonic() - connected
                if elapsed >= next_print:
                    print(f'{elapsed:.1f}秒 | received={len(worker.records)} | '
                          f'processed={len(records)} | targets={len(detection["detections"])}', flush=True)
                    next_print += 5.0
    except KeyboardInterrupt:
        report['outcome'], status = 'interrupted', 130
        report['error'] = traceback.format_exc()
    except Exception:
        report['outcome'], status = 'failed', 1
        report['error'] = traceback.format_exc()
    finally:
        if worker is not None:
            try:
                worker.stop()
                worker.raise_if_failed()
            except Exception:
                report['error'] = report.get('error', '') + traceback.format_exc()
                report['outcome'], status = 'failed', 1
        finished, cpu_finished = time.monotonic(), time.process_time()
        if detector is not None and backend == 'npu':
            try:
                detector.close()
            except Exception:
                report['error'] = report.get('error', '') + traceback.format_exc()
                report['outcome'], status = 'failed', 1
        if worker is not None and worker.started_monotonic_s is not None:
            elapsed = finished - worker.started_monotonic_s
            cpu = cpu_finished - worker.started_process_time_s
            report.update(transport=worker.summary([]), received_frame_count=len(worker.records),
                          elapsed_from_connection_s=elapsed,
                          processing_throughput_fps=len(records) / elapsed,
                          transport_fps=worker.summary([])['transport_fps'],
                          cpu_window={'process_time_s': cpu, 'wall_elapsed_s': elapsed,
                                      'cpu_percent_one_core': 100 * cpu / elapsed},
                          unprocessed_tail_receiver_frames=(len(worker.records) -
                              (records[-1]['receiver_index'] if records else 0)))
        report['processed_frame_count'] = len(records)
        report['skipped_receiver_frames'] = sum(r['skipped_receiver_frames'] for r in records)
        report['timing_ms'] = {
            field: {'count': len(records), 'p50': float(np.median([r[field] for r in records])),
                    'p95': float(np.percentile([r[field] for r in records], 95)),
                    'max': max(r[field] for r in records)}
            for field in (inference_field, 'complete_processing_ms') if records}
        try:
            if last_image is not None:
                save_image(directory / 'last_annotated.jpg', last_image)
        except Exception:
            report['error'] = report.get('error', '') + traceback.format_exc()
            report['outcome'], status = 'failed', 1
        if 'error' in report:
            (directory / 'error.txt').write_text(report['error'])
            print(report['error'], file=sys.stderr, flush=True)
        (directory / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        throughput = report.get('processing_throughput_fps', 0.0)
        print(f'结果：{report["outcome"]}；received={report["received_frame_count"]}；'
              f'processed={len(records)}；处理吞吐={throughput:.3f}fps；证据目录：{directory}', flush=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(duration=args.duration, model_path=args.model, directory=args.output_dir))
