#!/bin/zsh
# Mac entry: copy the stdlib observer, sample board resources for 11 minutes,
# and retrieve its saved evidence. No motion commands or package installation.
set -o pipefail

scp "/Users/ethan/Documents/RK3588 Microduck/work/measure_board_runtime_resources.py" \
  RK3588:/home/linaro/microduck-board/ &&
ssh -t RK3588 \
  'python3 -u "$HOME/microduck-board/measure_board_runtime_resources.py" --duration 660' &&
scp -r 'RK3588:/home/linaro/microduck-board/run/resources_*' \
  "/Users/ethan/Documents/RK3588 Microduck/task/evidence/G3/"
