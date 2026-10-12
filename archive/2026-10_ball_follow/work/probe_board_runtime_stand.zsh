#!/bin/zsh
set -o pipefail

BOARD_BODY_LOG=$(cat "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/current_body_log.txt") &&
MD_ROBOT_SOCKET=/private/tmp/md-rk-startup-ethan.sock MD_EVIDENCE_GATE=G3 \
  "/Users/ethan/Documents/RK3588 Microduck/task/microduck_r1/.venv/bin/python" -u \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_stand.py" \
  --keep-enabled \
  --support-log "$BOARD_BODY_LOG" \
  2>&1 | tee "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/board_stand_console_$(date +%Y%m%d_%H%M%S).log"
