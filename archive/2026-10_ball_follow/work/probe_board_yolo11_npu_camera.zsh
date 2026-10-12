#!/bin/zsh
# Copy the prepared NPU camera detector, run for 30 seconds, and retrieve evidence.
set -o pipefail
RK_NPU_CAMERA_ROOT="/Users/ethan/Documents/RK3588 Microduck"
RK_NPU_CAMERA_ID=$(date -u +%Y%m%d_%H%M%S)
RK_NPU_CAMERA_BUNDLE="$RK_NPU_CAMERA_ROOT/work/vision-offline/yolo11n-npu-camera"
RK_NPU_CAMERA_LOG="$RK_NPU_CAMERA_ROOT/task/evidence/G4/camera_npu_console_$RK_NPU_CAMERA_ID.log"

{
  scp -r "$RK_NPU_CAMERA_BUNDLE" RK3588:/home/linaro/microduck-board/ || exit $?
  RK_NPU_CAMERA_STATUS=0
  ssh RK3588 "export RK_NPU_CAMERA_ID=$RK_NPU_CAMERA_ID; "'
    set -o pipefail
    cd "$HOME/microduck-board/yolo11n-npu-camera" &&
    sha256sum -c SHA256SUMS &&
    PYTHONPATH="$HOME/microduck-board/vision-cpu-python" \
      "$HOME/rk3588-npu-venv/bin/python" -u probe_board_yolo11_npu_camera.py \
      --model models/yolo11n-rk3588-fp.rknn --duration 30 \
      --output-dir "$HOME/microduck-board/run/G4/camera_npu_$RK_NPU_CAMERA_ID"
  ' || RK_NPU_CAMERA_STATUS=$?
  RK_NPU_CAMERA_COPY_STATUS=0
  scp -r "RK3588:/home/linaro/microduck-board/run/G4/camera_npu_$RK_NPU_CAMERA_ID" \
    "$RK_NPU_CAMERA_ROOT/task/evidence/G4/" || RK_NPU_CAMERA_COPY_STATUS=$?
  print "实验退出码：$RK_NPU_CAMERA_STATUS；证据回传退出码：$RK_NPU_CAMERA_COPY_STATUS"
  if (( RK_NPU_CAMERA_STATUS != 0 )); then exit $RK_NPU_CAMERA_STATUS; fi
  exit $RK_NPU_CAMERA_COPY_STATUS
} 2>&1 | tee "$RK_NPU_CAMERA_LOG"
