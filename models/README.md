# models/

- `policies/`: ONNX policies read by robotd (`walk_17500.onnx` is the walking policy in use;
  `velstand.onnx` is the official stand policy). Tracked in git.
- `yolo/yolo11n_coco/`: COCO-pretrained YOLO11n. `yolo11n_coco_fp32_320.onnx` + `export.json`
  (board CPU), `yolo11n_coco_fp16_320.rknn` + `conversion.json` (board NPU; the runner checks the
  hashes in `conversion.json`). The `.pt` weights are local only.
- `checkpoints/`: training checkpoints, local only.

Name pattern: `<model>_<dataset>_<precision>_<input size>.<ext>`. `MANIFEST.tsv` lists every file.
