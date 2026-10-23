"""Plan 2D step 1: export the fine-tuned best.pt as two static FP32 ONNX files.

    PYTHONPATH=src python scripts/vision/stage2/export_split_onnx.py --weights <best.pt>

Writes a new runs/<UTC>_s2d_export/ with
  <stem>_split_320.onnx   9 outputs, the file RKNN is converted from and the CPU detector runs:
                          per stride 8, 16, 32: box [1,64,H,W] (DFL logits, 4 sides x 16 bins),
                          cls [1,nc,H,W] (after sigmoid), sum [1,1,H,W] (clamp(sum of cls, 0, 1))
  <stem>_single_320.onnx  Ultralytics' own export [1,4+nc,2100], kept only to check the split
                          file against (check_split_onnx.py) and against Ultralytics
  export.json             sha256 of every file, versions, input/output shapes

The split head is the one in airockchip/ultralytics_yolo11 (RKOPT_README.md, Detect.forward when
format == 'rknn'): it stops before DFL and box decoding because "post-processing in the model is
not friendly to quantization" and DFL is slow on the NPU; the sum branch lets the decoder skip
empty cells. That fork is not installed here: the same few lines are patched onto the head of our
own model in this process. No installed package or upstream file is edited.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import types

import onnx
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

ROOT = Path(__file__).resolve().parents[3]
STRIDES = (8, 16, 32)
SIZE = 320


def split_forward(self, x):
    """airockchip's Detect.forward for format 'rknn': no DFL, no anchors, no concatenation."""
    y = []
    for i in range(self.nl):
        cls = torch.sigmoid(self.cv3[i](x[i]))
        y += [self.cv2[i](x[i]), cls, torch.clamp(cls.sum(1, keepdim=True), 0, 1)]
    return y


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shapes(model) -> tuple[dict, dict]:
    shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    return ({v.name: shape(v) for v in model.graph.input},
            {v.name: shape(v) for v in model.graph.output})


def export_split(yolo: YOLO, out: Path, source_sha: str) -> dict:
    model = copy.deepcopy(yolo.model).float().eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.fuse()
    head = model.model[-1]
    if not isinstance(head, Detect) or head.reg_max != 16 or tuple(head.stride.tolist()) != STRIDES:
        raise ValueError(f'unexpected head {type(head).__name__} reg_max {head.reg_max} '
                         f'stride {head.stride.tolist()}')
    head.forward = types.MethodType(split_forward, head)
    names = [f'{kind}{s}' for s in STRIDES for kind in ('box', 'cls', 'sum')]
    im = torch.zeros(1, 3, SIZE, SIZE)
    torch.onnx.export(model, im, str(out), opset_version=17, input_names=['images'],
                      output_names=names, do_constant_folding=True, dynamo=False)
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    meta = {'names': json.dumps(dict(enumerate(yolo.names[i] for i in range(len(yolo.names))))),
            'stride': json.dumps(list(STRIDES)), 'imgsz': json.dumps([SIZE, SIZE]),
            'reg_max': '16', 'task': 'detect', 'layout': 'split9 (box, cls, sum) x stride 8/16/32',
            'source_pt_sha256': source_sha,
            'reference': 'airockchip/ultralytics_yolo11 RKOPT_README.md'}
    for k, v in meta.items():
        onnx_model.metadata_props.add(key=k, value=v)
    onnx.save(onnx_model, str(out))
    inputs, outputs = shapes(onnx_model)
    nc = len(yolo.names)
    want = {}
    for s in STRIDES:
        g = SIZE // s
        want.update({f'box{s}': [1, 64, g, g], f'cls{s}': [1, nc, g, g], f'sum{s}': [1, 1, g, g]})
    if inputs != {'images': [1, 3, SIZE, SIZE]} or outputs != want:
        raise ValueError(f'unexpected interface {inputs} {outputs}')
    return {'inputs': inputs, 'outputs': outputs, 'metadata': meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--out', type=Path, help='default: runs/<UTC>_s2d_export')
    args = ap.parse_args(argv)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = args.out or ROOT / 'runs' / f'{stamp}_s2d_export'
    folder.mkdir(parents=True)
    weights = folder / args.weights.name
    shutil.copy2(args.weights, weights)
    stem = 'yolo11n_toys'
    # Ultralytics 8.3.200 predates PyTorch 2.9's dynamo=True default (same as export_yolo_onnx.py).
    torch.onnx.export = partial(torch.onnx.export, dynamo=False)
    yolo = YOLO(str(weights))
    split = folder / f'{stem}_split_{SIZE}.onnx'
    split_info = export_split(yolo, split, sha256(weights))
    single_tmp = Path(YOLO(str(weights)).export(format='onnx', imgsz=SIZE, batch=1, device='cpu',
                                                dynamic=False, half=False, simplify=False,
                                                opset=17, nms=False))
    single = folder / f'{stem}_single_{SIZE}.onnx'
    single_tmp.rename(single)
    _, single_out = shapes(onnx.load(str(single)))
    report = {
        'stamp': stamp, 'weights': str(args.weights), 'weights_sha256': sha256(weights),
        'names': [yolo.names[i] for i in range(len(yolo.names))],
        'split': {'file': split.name, 'sha256': sha256(split), **split_info},
        'single': {'file': single.name, 'sha256': sha256(single), 'outputs': single_out},
        'opset': 17, 'torch_exporter': 'legacy: explicit dynamo=False',
        'python': sys.version.split()[0],
        'versions': {n: importlib.metadata.version(n)
                     for n in ('ultralytics', 'torch', 'onnx', 'numpy')},
    }
    (folder / 'export.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'folder': str(folder), 'split': report['split']['sha256'],
                      'single': report['single']['sha256']}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
