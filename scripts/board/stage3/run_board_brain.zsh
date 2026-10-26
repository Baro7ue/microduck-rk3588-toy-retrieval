#!/bin/zsh
# Stage 3F on the RK3588: the brain runs on the board, one process per episode (state machine,
# YOLO INT8 on the NPU, robot.state from the board's robotd, ToF from the Mac sim body's 7801);
# this Mac lays out each episode, keeps the truth and judges it.
#   run_board_brain.zsh 3e [--episodes 3000:bus,…] [--quick] [--backend int8|fp16|cpu] [--mac-run runs/<dir>]
# Default episodes: the first 9 of 3E (dev layouts 3000-3002 x bus, henry, ketchup; plan 9).
# --mac-run: the Mac 3E run whose same 9 episodes the board's deliveries are compared with (or
# later: run_test.zsh 3e --rescore runs/<board run> --mac-run runs/<Mac run>).
# Needs, already running:
#   terminal A  scripts/sim/stage1/run_arena_body.zsh --camera --board   (camera and ToF on $MD_SIM_HOST)
#   terminal B  scripts/board/stage0/run_board_robotd.zsh s2
# The code, the four configs and the frozen Stage 2 models go to the board as a checksummed bundle
# ($MD_BOARD_ROOT/bundles/<run>/, checked with sha256sum -c as in 2F). Episode logs come back per
# episode. Output: runs/<UTC>_s3_board_bundle/{bundle/, console.log} and runs/<UTC>_s3e_s2_board/.
# The brain is pinned with taskset -c $MD_BOARD_CPUS (env.sh: the big cores 4-7; empty = not pinned;
# unset, e.g. an env.sh from before the big-core change, also means 4-7).
set -uo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_board_brain.zsh 3e|3ab [--episodes 3000:bus,…] [--quick] [--backend int8|fp16|cpu] [--mac-run runs/<dir>]'
(( $# >= 1 )) || md_die "$usage"
case $1 in
  3e|3ab) test=$1 ;;
  *) md_die "$usage" ;;
esac
shift
args=("$@")
(( ${args[(I)--target]} )) && md_die "$usage"
if (( ! ${args[(I)--episodes]} && ! ${args[(I)--quick]} )); then
  args+=(--episodes 3000:bus,3000:henry,3000:ketchup,3001:bus,3001:henry,3001:ketchup,3002:bus,3002:henry,3002:ketchup)
fi
models=(models/yolo/yolo11n_toys/yolo11n_toys_split_320.onnx models/yolo/yolo11n_toys/yolo11n_toys_fp16.rknn
        models/yolo/yolo11n_toys/yolo11n_toys_int8.rknn)
configs=(configs/control/stage1.json configs/control/stage1_facts.json configs/control/stage3.json
         configs/vision/stage2.json)
for f in $models $configs; do
  [[ -f $MD_ROOT/$f ]] || md_die "缺 $f；先把仓库更新到阶段 3 的提交（PR #2 冻结的模型也要在）。"
done
md_port_busy 7811 || md_die "7811 端口没有测试端口；先在终端 A 运行 scripts/sim/stage1/run_arena_body.zsh --camera --board。"
md_port_busy 7901 || md_die "7901 端口没有头部相机；终端 A 要带 --camera --board 启动。"
ssh "$MD_BOARD_HOST" "test -S ${(q)MD_BOARD_ROOT}/run/robotd.sock" ||
  md_die "板上没有 robotd 控制口；先在终端 B 运行 scripts/board/stage0/run_board_robotd.zsh s2。"

export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"
run=$(md_run_dir s3_board_bundle) || exit 1
bundle=$run/bundle
md_bundle "$bundle" || md_die "打包代码失败。"
for f in $models $configs; do
  mkdir -p "$bundle/${f:h}" && cp "$MD_ROOT/$f" "$bundle/$f" || md_die "复制 $f 失败。"
done
md_sha256sums "$bundle" || md_die "生成 SHA256SUMS 失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}

{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || { print -r -- "代码包传到板上失败。"; exit 1; }
  ssh "$MD_BOARD_HOST" "cd ${(q)remote_bundle} && sha256sum -c --quiet SHA256SUMS && mkdir -p ${(q)remote_out}" ||
    { print -r -- "板上校验代码包失败。"; exit 1; }
  print -r -- "代码包已在板上校验：$remote_bundle"
  cd "$MD_ROOT"
  "$MD_PY" -u -m microduck_toy.stage3.episode $test --target board --source stream \
    --board-bundle "$remote_bundle" --board-out "$remote_out" --board-py "$MD_BOARD_PY" \
    --board-socket "$MD_BOARD_ROOT/run/robotd.sock" --board-host "$MD_BOARD_HOST" --sim-host "$MD_SIM_HOST" \
    --board-cpus "${MD_BOARD_CPUS-4-7}" --socket "$MD_STARTUP_SOCKET" $args
} 2>&1 | tee "$run/console.log"
exit ${pipestatus[1]}
