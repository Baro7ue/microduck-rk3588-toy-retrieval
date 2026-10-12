#!/bin/zsh
# G4: copy external camera detector scripts, run one 30s board CPU trial, copy its evidence.
# Requires: Mac camera server on 192.168.50.1:7901; installed isolated CPU packages and model.
# Creates: this run's board and Mac G4 evidence plus Mac console log.
# Uses camera TCP and SSH only; does not start/stop robotd or issue motion commands.
set -o pipefail
RK_CAMERA_ID=$(date -u +%Y%m%d_%H%M%S)
RK_CAMERA_ROOT="/Users/ethan/Documents/RK3588 Microduck"

{
  scp -r "$RK_CAMERA_ROOT/work/vision-offline/yolo11n-camera" \
    RK3588:/home/linaro/microduck-board/ &&
  ssh -t RK3588 "export RK_CAMERA_ID=$RK_CAMERA_ID; "'
    set -o pipefail
    cd "$HOME/microduck-board/yolo11n-camera" &&
    sha256sum -c SHA256SUMS &&
    PYTHONPATH="$HOME/microduck-board/vision-cpu-python" \
      "$HOME/rk3588-npu-venv/bin/python" -u probe_board_yolo11_camera.py \
      --model "$HOME/microduck-board/yolo11n-cpu/models/yolo11n.onnx" \
      --duration 30 \
      --output-dir "$HOME/microduck-board/run/G4/camera_cpu_$RK_CAMERA_ID"
  ' &&
  scp -r "RK3588:/home/linaro/microduck-board/run/G4/camera_cpu_$RK_CAMERA_ID" \
    "$RK_CAMERA_ROOT/task/evidence/G4/"
} 2>&1 | tee "$RK_CAMERA_ROOT/task/evidence/G4/camera_cpu_console_$RK_CAMERA_ID.log"
