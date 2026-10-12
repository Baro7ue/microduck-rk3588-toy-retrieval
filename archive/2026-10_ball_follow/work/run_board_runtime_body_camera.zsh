#!/bin/zsh
set -o pipefail

BOARD_CAMERA_LOG="/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/mac_body_camera_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4" &&
cd "/Users/ethan/Documents/RK3588 Microduck/task/microduck_r1" &&
MD_STARTUP_SOCKET=/private/tmp/md-rk-startup-ethan.sock PYTHONUNBUFFERED=1 \
  .venv/bin/mjpython "/Users/ethan/Documents/RK3588 Microduck/work/runtime_bootstrap_body.py" \
  --ducks 1 --host 192.168.50.1 --port 7801 --keyframe HOME \
  --cameras a --frame-port 7901 --camera-fps 15 --headless \
  2>&1 | tee "$BOARD_CAMERA_LOG"
