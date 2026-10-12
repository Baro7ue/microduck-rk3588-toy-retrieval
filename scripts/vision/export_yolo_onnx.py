# Moved 2026-10-11 from work/export_yolo11_cpu.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Export the official YOLO11n checkpoint as static FP32 ONNX for CPU.

Input: models/yolo/yolo11n_coco/yolo11n_coco.pt (or --weights).
Output: <weights stem>.onnx and export.json in a new runs/<UTC stamp>_export_yolo_onnx/;
copy the result into models/ yourself once it is checked. Console log saved by caller.
Uses isolated export tooling; no devices, sockets, or robot processes.
"""
import argparse
import datetime as dt
from functools import partial
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sys

import onnx
import torch
from ultralytics import YOLO

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weights", type=Path, default=root / "models/yolo/yolo11n_coco/yolo11n_coco.pt")
cli = parser.parse_args()
stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
folder = root / "runs" / f"{stamp}_export_yolo_onnx"
folder.mkdir(parents=True)
weights = folder / cli.weights.name
shutil.copy2(cli.weights, weights)
args = dict(format="onnx", imgsz=320, batch=1, device="cpu", dynamic=False,
            half=False, simplify=False, opset=17, nms=False)

# Ultralytics 8.3.200 predates PyTorch 2.9's dynamo=True default.
# Explicitly select the legacy exporter in this process, keeping opset 17.
# No installed package, pinned environment, or repository file is edited.
torch.onnx.export = partial(torch.onnx.export, dynamo=False)
result = Path(YOLO(str(weights)).export(**args))
model = onnx.load(str(result))
onnx.checker.check_model(model)
shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
inputs = {v.name: shape(v) for v in model.graph.input}
outputs = {v.name: shape(v) for v in model.graph.output}
assert inputs == {"images": [1, 3, 320, 320]}, inputs
assert outputs == {"output0": [1, 84, 2100]}, outputs
report = {
    "status": "exported_and_onnx_checker_passed",
    "checkpoint_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    "onnx_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
    "python": sys.version, "arguments": args,
    "torch_exporter": "legacy: explicit dynamo=False",
    "versions": {name: importlib.metadata.version(name) for name in
                 ("ultralytics", "torch", "torchvision", "onnx", "numpy")},
    "inputs": inputs, "outputs": outputs,
    "opsets": {v.domain or "ai.onnx": v.version for v in model.opset_import},
    "metadata": {v.key: v.value for v in model.metadata_props},
}
(folder / "export.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("status", "inputs", "outputs", "onnx_sha256")}, indent=2))
