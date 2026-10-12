#!/bin/zsh
# G4: copy a fixed offline bundle, install isolated CPU bindings, detect two images.
# Input: Mac offline bundle, RK3588 SSH alias, existing board Python 3.11 venv.
# Output: board run/G4/yolo11_cpu_TIMESTAMP and copied Mac G4 evidence; console log below.
# No camera, body, motion RPC, or robot processes are started/stopped.
set -o pipefail
RK_VISION_ID=$(date -u +%Y%m%d_%H%M%S)
RK_VISION_ROOT="/Users/ethan/Documents/RK3588 Microduck"

{
  scp -r "$RK_VISION_ROOT/work/vision-offline/yolo11n-cpu" \
    RK3588:/home/linaro/microduck-board/ &&
  ssh -t RK3588 "export RK_VISION_ID=$RK_VISION_ID; "'
    set -o pipefail
    cd "$HOME/microduck-board/yolo11n-cpu" &&
    sha256sum -c SHA256SUMS &&
    "$HOME/rk3588-npu-venv/bin/python" -m pip install \
      --no-index --no-deps --require-hashes --find-links wheelhouse \
      --target "$HOME/microduck-board/vision-cpu-python" \
      -r requirements-lock.txt &&
    export PYTHONPATH="$HOME/microduck-board/vision-cpu-python" &&
    RK_VISION_DIR="$HOME/microduck-board/run/G4/yolo11_cpu_$RK_VISION_ID" &&
    "$HOME/rk3588-npu-venv/bin/python" -u detect_yolo11_image.py \
      --model models/yolo11n.onnx --image images/bus.jpg \
      --output-dir "$RK_VISION_DIR/bus" &&
    "$HOME/rk3588-npu-venv/bin/python" -u detect_yolo11_image.py \
      --model models/yolo11n.onnx --image images/camera_floor.png \
      --output-dir "$RK_VISION_DIR/camera_floor"
  ' &&
  scp -r "RK3588:/home/linaro/microduck-board/run/G4/yolo11_cpu_$RK_VISION_ID" \
    "$RK_VISION_ROOT/task/evidence/G4/"
} 2>&1 | tee "$RK_VISION_ROOT/task/evidence/G4/yolo11_cpu_console_$RK_VISION_ID.log"
