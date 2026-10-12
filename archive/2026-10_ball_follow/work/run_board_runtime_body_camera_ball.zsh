#!/bin/zsh
set -o pipefail
RK_BALL_ROOT="/Users/ethan/Documents/RK3588 Microduck"
RK_BALL_LOG="$RK_BALL_ROOT/task/evidence/G4/mac_body_camera_ball_$(date +%Y%m%d_%H%M%S).log"
{
  cat "$RK_BALL_ROOT/work/g4-scenes/ball_1m/manifest.json" &&
  cd "$RK_BALL_ROOT/task/microduck_r1" &&
  MD_STARTUP_SOCKET=/private/tmp/md-rk-startup-ethan.sock PYTHONUNBUFFERED=1 \
    .venv/bin/mjpython "$RK_BALL_ROOT/work/runtime_bootstrap_body.py" \
    --scene "$RK_BALL_ROOT/work/g4-scenes/ball_1m/scene_ball_1m.xml" \
    --ducks 1 --host 192.168.50.1 --port 7801 --keyframe HOME \
    --cameras a --frame-port 7901 --camera-fps 15 --headless
} 2>&1 | tee "$RK_BALL_LOG"
