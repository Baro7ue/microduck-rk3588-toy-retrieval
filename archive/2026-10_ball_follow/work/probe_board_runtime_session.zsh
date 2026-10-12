#!/bin/zsh
# Mac entry: stand for 10 s after support release, then walk for 60 s with PI.
# Requires the fresh Mac body and board robotd/SSH forward to remain running.
# Existing probes save G3 logs; failed standing prevents forward motion.
set -o pipefail

zsh "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_runtime_stand.zsh" &&
zsh "/Users/ethan/Documents/RK3588 Microduck/work/probe_board_runtime_walk.zsh" 60
