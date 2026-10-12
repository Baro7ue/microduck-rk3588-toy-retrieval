#!/bin/zsh
set -o pipefail

mkdir -p "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4" &&
scp "/Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_visual_load.py" \
  "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_camera_transport.py" \
  RK3588:/home/linaro/microduck-board/ &&
ssh -t RK3588 \
  'python3 -u "$HOME/microduck-board/probe_board_camera_transport.py"' &&
scp -r 'RK3588:/home/linaro/microduck-board/run/G4/camera_transport_*' \
  "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/" &&
"/Users/ethan/Documents/RK3588 Microduck/task/microduck_r1/.venv/bin/python" \
  "/Users/ethan/Documents/RK3588 Microduck/work/render_board_camera_frames.py"
