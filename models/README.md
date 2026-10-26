# models/

- `policies/`: ONNX policies read by robotd (`walk_17500.onnx` is the walking policy in use;
  `velstand.onnx` is the official stand policy). Tracked in git.
- `yolo/yolo11n_coco/`: COCO-pretrained YOLO11n. `yolo11n_coco_fp32_320.onnx` + `export.json`
  (board CPU), `yolo11n_coco_fp16_320.rknn` + `conversion.json` (board NPU; the runner checks the
  hashes in `conversion.json`). The `.pt` weights are local only.
- `yolo/yolo11n_toys/`: YOLO11n fine-tuned on the Stage 2 toy dataset (6 classes), frozen in Stage 2G.
  `yolo11n_toys_split_320.onnx` (9 outputs, board CPU and the RKNN source) + `export.json`,
  `yolo11n_toys_fp16.rknn` and `yolo11n_toys_int8.rknn` (board NPU) + their `conversion_*.json`.
  INT8 is the closed-loop model; hashes and measured numbers are in `configs/vision/stage2_facts.json`.
  The `.pt` weights stay on the Colab Drive. These keep the names the eval pack and launchers use.
- `checkpoints/`: training checkpoints, local only.

Name pattern: `<model>_<dataset>_<precision>_<input size>.<ext>`. `MANIFEST.tsv` lists every file.
