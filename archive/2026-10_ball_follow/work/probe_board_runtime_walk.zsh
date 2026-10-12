#!/bin/zsh
set -o pipefail

MD_EVIDENCE_GATE=G3 \
  "/Users/ethan/Documents/RK3588 Microduck/task/microduck_r1/.venv/bin/python" -u \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_walk.py" \
  --socket /private/tmp/md-rk-startup-ethan.sock --duration "${1:-10}" --vx 0.1 --heading-hold \
  2>&1 | tee "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/board_walk_console_$(date +%Y%m%d_%H%M%S).log"
