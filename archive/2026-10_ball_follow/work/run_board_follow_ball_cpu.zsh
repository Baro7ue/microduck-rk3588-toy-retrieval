#!/bin/zsh
# Reuse the current body and enabled robotd; run one bounded board CPU approach.
set -o pipefail
RK_FOLLOW_ROOT="/Users/ethan/Documents/RK3588 Microduck"
RK_FOLLOW_SPEED=0.1
RK_FOLLOW_STOP_AREA=0.012
while (( $# != 0 )); do
  if [[ $# -ge 2 && $1 == --speed && ( $2 == 0.1 || $2 == 0.3 ) ]]; then
    RK_FOLLOW_SPEED=$2
  elif [[ $# -ge 2 && $1 == --stop-area && ( $2 == <-> || $2 == <->.<-> || $2 == .<-> ) ]]; then
    RK_FOLLOW_STOP_AREA=$2
  else
    print -u2 -r -- "用法：run_board_follow_ball_cpu.zsh [--speed 0.1|0.3] [--stop-area 0.012]"
    exit 2
  fi
  shift 2
done
RK_FOLLOW_ID=$(date -u +%Y%m%d_%H%M%S)
RK_FOLLOW_REMOTE="/home/linaro/microduck-board/run/G4/ball_follow_$RK_FOLLOW_ID"

{
  print -r -- "本次板端证据目标目录：$RK_FOLLOW_REMOTE"
  scp -r "$RK_FOLLOW_ROOT/work/vision-offline/yolo11n-follow" \
    RK3588:/home/linaro/microduck-board/
  RK_FOLLOW_PRIMARY_STATUS=$?
  if (( RK_FOLLOW_PRIMARY_STATUS == 0 )); then
    ssh -t RK3588 "export RK_FOLLOW_ID=$RK_FOLLOW_ID RK_FOLLOW_SPEED=$RK_FOLLOW_SPEED RK_FOLLOW_STOP_AREA=$RK_FOLLOW_STOP_AREA; "'
      set -o pipefail
      cd "$HOME/microduck-board/yolo11n-follow" &&
      sha256sum -c SHA256SUMS &&
      PYTHONPATH="$HOME/microduck-board/vision-cpu-python" \
        "$HOME/rk3588-npu-venv/bin/python" -u follow_ball_cpu.py \
        --speed "$RK_FOLLOW_SPEED" \
        --stop-area "$RK_FOLLOW_STOP_AREA" \
        --output-dir "$HOME/microduck-board/run/G4/ball_follow_$RK_FOLLOW_ID"
    '
    RK_FOLLOW_PRIMARY_STATUS=$?
  fi
  scp -r "RK3588:$RK_FOLLOW_REMOTE" "$RK_FOLLOW_ROOT/task/evidence/G4/"
  RK_FOLLOW_RECEIVE_STATUS=$?
  print -r -- "控制入口退出码：$RK_FOLLOW_PRIMARY_STATUS；证据回传退出码：$RK_FOLLOW_RECEIVE_STATUS"
  if (( RK_FOLLOW_RECEIVE_STATUS != 0 )); then
    print -r -- "只补传本次目录，不重跑控制："
    print -r -- "scp -r RK3588:$RK_FOLLOW_REMOTE \"$RK_FOLLOW_ROOT/task/evidence/G4/\""
  fi
  (( RK_FOLLOW_PRIMARY_STATUS == 0 )) || exit "$RK_FOLLOW_PRIMARY_STATUS"
  exit "$RK_FOLLOW_RECEIVE_STATUS"
} 2>&1 | tee "$RK_FOLLOW_ROOT/task/evidence/G4/ball_follow_console_$RK_FOLLOW_ID.log"
RK_FOLLOW_PIPE_STATUS=("${pipestatus[@]}")
(( RK_FOLLOW_PIPE_STATUS[1] == 0 )) || exit "${RK_FOLLOW_PIPE_STATUS[1]}"
exit "${RK_FOLLOW_PIPE_STATUS[2]}"
