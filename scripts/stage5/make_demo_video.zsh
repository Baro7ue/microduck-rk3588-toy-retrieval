#!/bin/zsh
# The room demo's video, drawn after the run from its logged poses (src/microduck_toy/stage5/replay.py):
#   make_demo_video.zsh runs/<UTC>_s5_room_demo[_board] [--fps 25] [--speed 1] [--from S] [--to S] [--size 1280x720]
# Needs nothing running (and should run with nothing else running: drawing takes the CPU). Output:
# <run>/demo_video.mp4 (follow camera, the robot's own camera, a top-down map, target and state).
set -euo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：make_demo_video.zsh runs/<房间演示的运行目录> [--fps 25] [--speed 1] [--from 秒] [--to 秒] [--size 1280x720]'
(( $# >= 1 )) || md_die "$usage"
run=${1:A}
shift
[[ -f $run/truth.qpos.f64 ]] || md_die "$run 里没有 truth.qpos.f64；只有用 run_arena_body.zsh --room 跑的房间演示才有。"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 writes the mp4
cd "$MD_ROOT"
exec "$MD_PY" -u -m microduck_toy.stage5.replay "$run" "$@"
