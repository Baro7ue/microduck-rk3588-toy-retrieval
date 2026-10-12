#!/bin/zsh
set -o pipefail

BOARD_BODY_LOG="/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/mac_body_$(date +%Y%m%d_%H%M%S).log"
print -r -- "$BOARD_BODY_LOG" > "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/current_body_log.txt" &&
cd "/Users/ethan/Documents/RK3588 Microduck/task/microduck_r1" &&
MD_STARTUP_SOCKET=/private/tmp/md-rk-startup-ethan.sock PYTHONUNBUFFERED=1 \
  .venv/bin/mjpython "/Users/ethan/Documents/RK3588 Microduck/work/runtime_bootstrap_body.py" \
  --ducks 1 --host 192.168.50.1 --port 7801 --keyframe HOME \
  2>&1 | tee "$BOARD_BODY_LOG"
