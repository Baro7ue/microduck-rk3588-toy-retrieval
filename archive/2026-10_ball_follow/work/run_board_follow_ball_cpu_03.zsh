#!/bin/zsh
# Fresh A/B first; verify standing and camera, then approach to a fixed 1.2% image area.
RK_FOLLOW_ROOT="/Users/ethan/Documents/RK3588 Microduck"
/bin/zsh "$RK_FOLLOW_ROOT/work/probe_board_stand_with_camera.zsh" &&
  /bin/zsh "$RK_FOLLOW_ROOT/work/run_board_follow_ball_cpu.zsh" --speed 0.3 --stop-area 0.012
