#!/bin/zsh
# Copy the prepared models, compare two offline images, and retrieve results.
set -o pipefail
RK_COMPARE_ROOT="/Users/ethan/Documents/RK3588 Microduck"
RK_COMPARE_ID=$(date -u +%Y%m%d_%H%M%S)
RK_COMPARE_BUNDLE="$RK_COMPARE_ROOT/work/vision-offline/yolo11n-cpu-npu"
RK_COMPARE_LOG="$RK_COMPARE_ROOT/task/evidence/G4/yolo11_cpu_npu_console_$RK_COMPARE_ID.log"

{
  scp -r "$RK_COMPARE_BUNDLE" RK3588:/home/linaro/microduck-board/ || exit $?
  RK_COMPARE_STATUS=0
  ssh RK3588 "export RK_COMPARE_ID=$RK_COMPARE_ID; "'
    set -o pipefail
    cd "$HOME/microduck-board/yolo11n-cpu-npu" &&
    sha256sum -c SHA256SUMS &&
    PYTHONPATH="$HOME/microduck-board/vision-cpu-python" \
      "$HOME/rk3588-npu-venv/bin/python" -u compare_yolo11_cpu_npu.py \
      --output-dir "$HOME/microduck-board/run/G4/yolo11_cpu_npu_$RK_COMPARE_ID"
  ' || RK_COMPARE_STATUS=$?
  RK_COMPARE_COPY_STATUS=0
  scp -r "RK3588:/home/linaro/microduck-board/run/G4/yolo11_cpu_npu_$RK_COMPARE_ID" \
    "$RK_COMPARE_ROOT/task/evidence/G4/" || RK_COMPARE_COPY_STATUS=$?
  print "实验退出码：$RK_COMPARE_STATUS；证据回传退出码：$RK_COMPARE_COPY_STATUS"
  if (( RK_COMPARE_STATUS != 0 )); then exit $RK_COMPARE_STATUS; fi
  exit $RK_COMPARE_COPY_STATUS
} 2>&1 | tee "$RK_COMPARE_LOG"
