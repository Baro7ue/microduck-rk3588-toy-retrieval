#!/bin/zsh
# Reuse enabled standing robotd; approach to the calibrated close-ball visual reference.
set -o pipefail
RK_FOLLOW_NPU_ROOT="/Users/ethan/Documents/RK3588 Microduck"
RK_FOLLOW_NPU_ID=$(date -u +%Y%m%d_%H%M%S)
RK_FOLLOW_NPU_REMOTE="/home/linaro/microduck-board/run/G4/ball_follow_npu_$RK_FOLLOW_NPU_ID"
RK_FOLLOW_NPU_LOG="$RK_FOLLOW_NPU_ROOT/task/evidence/G4/ball_follow_npu_console_$RK_FOLLOW_NPU_ID.log"

{
  print -r -- "本次板端证据目录：$RK_FOLLOW_NPU_REMOTE"
  scp -r "$RK_FOLLOW_NPU_ROOT/work/vision-offline/yolo11n-follow-npu" \
    RK3588:/home/linaro/microduck-board/ || exit $?
  RK_FOLLOW_NPU_STATUS=0
  ssh -t RK3588 "export RK_FOLLOW_NPU_ID=$RK_FOLLOW_NPU_ID; "'
    set -o pipefail
    cd "$HOME/microduck-board/yolo11n-follow-npu" &&
    sha256sum -c SHA256SUMS &&
    PYTHONPATH="$HOME/microduck-board/vision-cpu-python" \
      "$HOME/rk3588-npu-venv/bin/python" -u follow_ball_npu.py \
      --model models/yolo11n-rk3588-fp.rknn --speed 0.3 --stop-area 0.06 --orange-continuation \
      --camera-motion-compensation --body-steering \
      --output-dir "$HOME/microduck-board/run/G4/ball_follow_npu_$RK_FOLLOW_NPU_ID"
  ' || RK_FOLLOW_NPU_STATUS=$?
  RK_FOLLOW_NPU_COPY_STATUS=0
  scp -r "RK3588:$RK_FOLLOW_NPU_REMOTE" "$RK_FOLLOW_NPU_ROOT/task/evidence/G4/" \
    || RK_FOLLOW_NPU_COPY_STATUS=$?
  print -r -- "控制入口退出码：$RK_FOLLOW_NPU_STATUS；证据回传退出码：$RK_FOLLOW_NPU_COPY_STATUS"
  if (( RK_FOLLOW_NPU_COPY_STATUS != 0 )); then
    print -r -- "只补传本次数据：scp -r RK3588:$RK_FOLLOW_NPU_REMOTE \"$RK_FOLLOW_NPU_ROOT/task/evidence/G4/\""
  fi
  if (( RK_FOLLOW_NPU_STATUS != 0 )); then exit $RK_FOLLOW_NPU_STATUS; fi
  exit $RK_FOLLOW_NPU_COPY_STATUS
} 2>&1 | tee "$RK_FOLLOW_NPU_LOG"
