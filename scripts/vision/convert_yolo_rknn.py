# Copied 2026-10-11 from work/vision-offline/yolo11n-rknn-232/convert_yolo11_rknn.py (original kept in
# archive/2026-10_ball_follow/vision-offline/), unchanged. It runs on the Ubuntu x86 conversion machine from a
# bundle folder with models/yolo11n.onnx next to it; see that folder's README.md for the steps.
"""Convert the existing YOLO11n-320 ONNX to nonquantized RKNN for RK3588."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import sys

import onnx
from rknn.api import RKNN


def check(stage, code):
    if code != 0:
        raise RuntimeError(f"{stage} failed: return code {code}")


def main():
    root = Path(__file__).resolve().parent
    model_path = root / "models/yolo11n.onnx"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != "121662c804b9aa906563235ed2909815f2c23bc6e4052e651e01468817551826":
        raise ValueError("ONNX differs from the existing CPU baseline")
    if version("rknn-toolkit2") != "2.3.2":
        raise ValueError("This experiment requires rknn-toolkit2 2.3.2")
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    if inputs != {"images": [1, 3, 320, 320]} or outputs != {"output0": [1, 84, 2100]}:
        raise ValueError(f"Unexpected model interface: {inputs}, {outputs}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = root / "converted" / stamp
    output_dir.mkdir(parents=True)
    output_path = output_dir / "yolo11n-rk3588-fp.rknn"
    print(f"ONNX: IR {model.ir_version}, input {inputs}, output {outputs}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    rknn = RKNN(verbose=True)
    try:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                    target_platform="rk3588")
        check("load_onnx", rknn.load_onnx(model=str(model_path)))
        check("build", rknn.build(do_quantization=False))
        check("export_rknn", rknn.export_rknn(str(output_path)))
    finally:
        rknn.release()
    if output_path.stat().st_size == 0:
        raise RuntimeError("Exported RKNN file is empty")
    report = {
        "status": "converted", "scope": "offline conversion; board inference not yet tested",
        "onnx_sha256": model_hash, "onnx_ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": inputs, "outputs": outputs, "target_platform": "rk3588",
        "do_quantization": False, "mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]],
        "versions": {name: version(name) for name in
                     ("rknn-toolkit2", "onnx", "onnxruntime", "torch", "numpy")},
        "python": sys.version.split()[0], "rknn_path": str(output_path),
        "rknn_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (output_dir / "conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Conversion completed: {output_path}", flush=True)


if __name__ == "__main__":
    main()
