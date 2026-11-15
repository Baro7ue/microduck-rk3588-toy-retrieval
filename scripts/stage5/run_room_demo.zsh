#!/bin/zsh
# Stage 5 room demo (not a formal result): in a 4.4 x 4.4 m room with furniture and 12 toys (two of
# each target, three of each distractor), the frozen brain and YOLO deliver bus, henry and ketchup
# one after another, without a reset (src/microduck_toy/stage5/demo.py, configs/demo/room.json).
#   run_room_demo.zsh [--board] [--seed N] [--targets bus,henry,ketchup] [--backend cpu|fp16|int8]
# Mac (default): the brain runs in this process, YOLO FP32 on the CPU. Needs, already running:
#   terminal A  scripts/sim/stage1/run_arena_body.zsh --camera --room
#   terminal B  scripts/sim/stage0/run_robotd.zsh s2
# --board: the brain runs on the RK3588 (YOLO INT8 on the NPU), one ssh per target, as Stage 3F.
# Needs, already running:
#   terminal A  scripts/sim/stage1/run_arena_body.zsh --camera --room --board
#   terminal B  scripts/board/stage0/run_board_robotd.zsh s2
# The code, the four frozen configs, configs/demo/room.json and the frozen models go to the board as
# a checksummed bundle. Output: runs/<UTC>_s5_room_demo[_board]/ (demo.json, summary.json, truth.csv,
# episodes/<n>_<target>/...). Ctrl+C stops after saving what is finished.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_room_demo.zsh [--board] [--seed N] [--targets bus,henry,ketchup] [--backend cpu|fp16|int8]'
board=0
args=()
while (( $# )); do
  case $1 in
    --board) board=1; shift ;;
    --target|--episodes|--parts) md_die "$usage" ;;
    -h|--help) print -r -- "$usage"; exit 0 ;;
    *) args+=("$1"); shift ;;
  esac
done
[[ -f $MD_ROOT/configs/demo/room.json ]] || md_die "缺 configs/demo/room.json。"
md_port_busy 7811 || md_die "7811 端口没有测试端口；先在终端 A 运行 scripts/sim/stage1/run_arena_body.zsh --camera --room$( (( board )) && print -n ' --board')。"
md_port_busy 7901 || md_die "7901 端口没有头部相机；终端 A 要带 --camera 启动。"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 for the detector's letterbox (as Stage 2)
cd "$MD_ROOT"

if (( ! board )); then
  [[ -S $MD_MAC_STATE/robotd.sock ]] || md_die "没有 robotd 控制口；先在终端 B 运行 scripts/sim/stage0/run_robotd.zsh s2。"
  log=$(mktemp "${TMPDIR:-/tmp}/s5console.XXXXXX") || exit 1
  rc=0
  "$MD_PY" -u -m microduck_toy.stage5.demo --source stream $args 2>&1 | ( trap '' INT; tee "$log" ) || rc=${pipestatus[1]}
  run=$(sed -n 's/^运行目录：//p' "$log" | head -n 1)
  if [[ -n $run && -d $run ]]; then
    mv "$log" "$run/console.log"
    print -r -- "退出码 $rc；结果在 $run/demo.json"
  else
    rm -f "$log"
  fi
  exit $rc
fi

md_stage3_check
ssh "$MD_BOARD_HOST" "test -S ${(q)MD_BOARD_ROOT}/run/robotd.sock" ||
  md_die "板上没有 robotd 控制口；先在终端 B 运行 scripts/board/stage0/run_board_robotd.zsh s2。"
# The Stage 3 bundle (md_stage3_bundle) plus the room config.
demo_bundle() {
  local f
  md_bundle "$1" || return 1
  for f in $MD_STAGE3_FILES configs/demo/room.json; do
    mkdir -p "$1/${f:h}" && cp "$MD_ROOT/$f" "$1/$f" || return 1
  done
  md_sha256sums "$1"
}
run=$(md_run_dir s5_board_bundle) || exit 1
bundle=$run/bundle
demo_bundle "$bundle" || md_die "打包代码失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}

{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || { print -r -- "代码包传到板上失败。"; exit 1; }
  ssh "$MD_BOARD_HOST" "cd ${(q)remote_bundle} && sha256sum -c --quiet SHA256SUMS && mkdir -p ${(q)remote_out}" ||
    { print -r -- "板上校验代码包失败。"; exit 1; }
  print -r -- "代码包已在板上校验：$remote_bundle"
  "$MD_PY" -u -m microduck_toy.stage5.demo --target board --source stream \
    --board-bundle "$remote_bundle" --board-out "$remote_out" --board-py "$MD_BOARD_PY" \
    --board-socket "$MD_BOARD_ROOT/run/robotd.sock" --board-host "$MD_BOARD_HOST" --sim-host "$MD_SIM_HOST" \
    --board-cpus "${MD_BOARD_CPUS-4-6}" --socket "$MD_STARTUP_SOCKET" $args
} 2>&1 | ( trap '' INT; tee "$run/console.log" )     # tee outlives a Ctrl+C, so the save is logged
exit ${pipestatus[1]}
