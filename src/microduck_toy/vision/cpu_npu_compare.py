# Moved 2026-10-11 from work/compare_yolo11_cpu_npu.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Compare the same YOLO11n-320 image on board CPU and NPU; no robot control."""
import argparse
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import time
import traceback

import cv2
import numpy as np
import onnxruntime as ort

from microduck_toy.vision.yolo_postprocess import (letterbox, validate_model, decode_predictions,
                                 class_aware_nms, restore_boxes)


def npu_pixels(tensor):
    return np.ascontiguousarray(np.rint(tensor.transpose(0, 2, 3, 1) * 255), dtype=np.uint8)


def validate_output(result):
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise ValueError("Expected exactly one runtime output")
    raw = result[0]
    if not isinstance(raw, np.ndarray) or raw.shape != (1, 84, 2100) \
            or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError(f"Unexpected output: shape={getattr(raw, 'shape', None)}, "
                         f"dtype={getattr(raw, 'dtype', None)}; expected finite float32 [1,84,2100]")
    return raw


def measure(call):
    for _ in range(5):
        validate_output(call())
    samples = []
    for _ in range(10):
        started = time.perf_counter_ns()
        result = call()
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        raw = validate_output(result)
        samples.append(elapsed)
    return raw, {"samples_ms": samples, "p50_ms": float(np.median(samples)),
                 "p95_ms": float(np.percentile(samples, 95)), "warmup": 5, "count": 10}


def annotate(raw, image, gain, padding, destination):
    boxes, scores, classes = decode_predictions(raw, 0.25)
    keep = class_aware_nms(boxes, scores, classes, 0.45)
    boxes = restore_boxes(boxes[keep], image.shape[:2], gain, padding)
    canvas = image.copy()
    detections = []
    for box, score, class_id in zip(boxes, scores[keep], classes[keep]):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = "person" if class_id == 0 else "sports ball"
        detections.append({"class_id": int(class_id), "class_name": label,
                           "confidence": float(score), "xyxy": box.tolist()})
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(canvas, f"{label} {score:.3f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Cannot save {destination}")
    return detections


def main():
    from rknnlite.api import RKNNLite

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="bundle with models/{conversion.json,yolo11n.onnx,yolo11n-rk3588-fp.rknn}")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)
    root = args.root
    conversion = json.loads((root / "models/conversion.json").read_text())
    onnx_path = root / "models/yolo11n.onnx"
    rknn_path = root / "models/yolo11n-rk3588-fp.rknn"
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in (("onnx_sha256", onnx_path), ("rknn_sha256", rknn_path))}
    if hashes["onnx_sha256"] != "121662c804b9aa906563235ed2909815f2c23bc6e4052e651e01468817551826" \
            or any(conversion[name] != value for name, value in hashes.items()) \
            or conversion["status"] != "converted" or conversion["do_quantization"]:
        raise ValueError("Models do not match the nonquantized conversion record")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    cpu = ort.InferenceSession(str(onnx_path), sess_options=options,
                               providers=["CPUExecutionProvider"])
    model = validate_model(cpu)
    report = {"scope": "two offline images; not accuracy evaluation or camera FPS",
              "timing_scope": "synchronous session.run/inference wall time; excludes pre/postprocessing and validation",
              "hashes": hashes, "conversion": conversion, "cpu_threads": 2,
              "cpu_providers": cpu.get_providers(), "confidence": 0.25, "iou": 0.45,
              "versions": {name: version(name) for name in
                           ("rknn-toolkit-lite2", "onnxruntime", "numpy")}, "images": {}}
    npu = RKNNLite()
    try:
        code = npu.load_rknn(str(rknn_path))
        if code != 0:
            raise RuntimeError(f"load_rknn returned {code}")
        code = npu.init_runtime()
        if code != 0:
            raise RuntimeError(f"init_runtime returned {code}")
        report["npu_sdk"] = repr(npu.get_sdk_version())
        for name in ("bus.jpg", "camera_floor.png"):
            image_path = root / "images" / name
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot decode {image_path}")
            tensor, gain, padding = letterbox(image)
            pixels = npu_pixels(tensor)
            cpu_raw, cpu_time = measure(lambda: cpu.run([model["output_name"]],
                                                       {model["input_name"]: tensor}))
            npu_raw, npu_time = measure(lambda: npu.inference(
                inputs=[pixels], data_format=["nhwc"], inputs_pass_through=[0]))
            destination = args.output_dir / image_path.stem
            destination.mkdir()
            item = {"image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "gain": gain, "padding_left_top": padding,
                    "cpu_input": "RGB NCHW float32 /255",
                    "npu_input": "RGB NHWC uint8; compiler applies std255",
                    "cpu": {"call_time": cpu_time}, "npu": {"call_time": npu_time}}
            for backend, raw in (("cpu", cpu_raw), ("npu", npu_raw)):
                np.save(destination / f"{backend}_raw.npy", raw)
                item[backend]["detections"] = annotate(raw, image, gain, padding,
                                                       destination / f"{backend}_annotated.jpg")
            (destination / "comparison.json").write_text(json.dumps(item, indent=2) + "\n")
            report["images"][name] = item
            print(f"{name}: CPU {len(item['cpu']['detections'])} targets, "
                  f"p50 {cpu_time['p50_ms']:.2f} ms | NPU {len(item['npu']['detections'])} targets, "
                  f"p50 {npu_time['p50_ms']:.2f} ms", flush=True)
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Comparison completed: {args.output_dir}", flush=True)
    except Exception:
        (args.output_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        npu.release()


if __name__ == "__main__":
    main()
