#!/bin/zsh
# Run one Stage 3 test on this Mac against a running arena body (with its camera) and robotd s2:
#   run_test.zsh 3p|3ab|3c|3d|3e [--quick] [--parts …] [--episodes 3000:bus,…] [--drop-d 0.21]
#   run_test.zsh 3e --rescore runs/<dir>     (score a finished run again; no robot needed)
#   run_test.zsh layouts                      (write configs/control/stage3_layouts.json again)
#   3p  = the four 3P measurements (held toy in the picture, ToF zones, reach, spin bearings)
#   3ab = the 30 dev episodes until held (3A search and confirm, 3B pick)
#   3c  = tray look + drop at d = 0.19/0.21/0.23 (c_drop), carry from 9 pick spots (c_route)
#   3d  = pushed over once in GOTO/APPROACH/CARRY (d_fall), ToF walks at a wall (d_tof)
#   3e  = the 30 dev episodes until delivered (the Stage 3 gate, >= 21/30)
# Needs, already running:
#   terminal A  scripts/sim/stage1/run_arena_body.zsh --camera
#   terminal B  scripts/sim/stage0/run_robotd.zsh s2
# The brain runs in this process: camera from 127.0.0.1:7901, YOLO FP32 (split ONNX) on the CPU,
# ToF from the sim body's 7801. Ctrl+C stops after saving the finished episodes.
# Output: runs/<UTC>_s3<test>_s2/ (truth.csv, state.jsonl, events.jsonl, trials.jsonl,
#         episodes/<label>/brain.jsonl, summary.json, console.log).
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_test.zsh 3p|3ab|3c|3d|3e [--quick] [--parts …] [--episodes …]  或  run_test.zsh layouts'
(( $# >= 1 )) || md_die "$usage"
case $1 in
  3p|3ab|3c|3d|3e|layouts) test=$1 ;;
  *) md_die "$usage" ;;
esac
shift
(( ! ${@[(I)--target]} )) || md_die "板端回合用 scripts/board/stage3/run_board_brain.zsh。"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 for the detector's letterbox (as Stage 2)
cd "$MD_ROOT"
if [[ $test == layouts ]] || (( ${@[(I)--rescore]} )); then
  exec "$MD_PY" -u -m microduck_toy.stage3.episode $test "$@"
fi
md_port_busy 7811 || md_die "7811 端口没有测试端口；先在终端 A 运行 scripts/sim/stage1/run_arena_body.zsh --camera。"
md_port_busy 7901 || md_die "7901 端口没有头部相机；终端 A 要带 --camera 启动。"
[[ -S $MD_MAC_STATE/robotd.sock ]] || md_die "没有 robotd 控制口；先在终端 B 运行 scripts/sim/stage0/run_robotd.zsh s2。"
log=$(mktemp "${TMPDIR:-/tmp}/s3console.XXXXXX") || exit 1
rc=0
"$MD_PY" -u -m microduck_toy.stage3.episode $test --source stream "$@" 2>&1 | tee "$log" || rc=${pipestatus[1]}
run=$(sed -n 's/^运行目录：//p' "$log" | head -n 1)
if [[ -n $run && -d $run ]]; then
  mv "$log" "$run/console.log"
  print -r -- "退出码 $rc；结果在 $run/summary.json"
else
  rm -f "$log"
fi
exit $rc
