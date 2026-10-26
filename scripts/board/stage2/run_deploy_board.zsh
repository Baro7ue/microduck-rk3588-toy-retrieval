#!/bin/zsh
# Stage 2 2F on the RK3588: the live detector on the Mac sim's head camera while robotd s2 on the
# board stands and turns in place, once per backend (CPU FP32 ONNX, NPU FP16, NPU INT8), then the
# deployment table on this Mac.
#   run_deploy_board.zsh runs/<UTC>_s2e_pack [--frames N] [--backends cpu,fp16,int8]
# Needs, already running:
#   terminal A  scripts/sim/stage1/run_arena_body.zsh --camera --board   (camera on $MD_SIM_HOST:7901)
#   terminal B  scripts/board/stage0/run_board_robotd.zsh s2
# The toys of the config's layout seed are placed first (test port 7811). On the board, sudo asks
# for the password once, for the NPU load (/sys/kernel/debug/rknpu/load); without it the run goes
# on and the table says the load was not read. Output: runs/<UTC>_s2f_board_deploy/{bundle/,
# board/<backend>/{summary.json, frames.jsonl, resources.csv, events.jsonl}, deploy_table.md,
# deploy_table.json, layout.json, mac_versions.json, console.log}.
# The board Python runs on cores $MD_BOARD_CPUS (env.sh: 4-7, the big cores; MD_BOARD_CPUS= turns
# pinning off, as in the first 2F run, where the NPU backends' preprocess landed on the little cores).
set -uo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_deploy_board.zsh runs/<UTC>_s2e_pack [--frames N] [--backends cpu,fp16,int8]'
(( $# >= 1 )) || md_die "$usage"
pack=${1:A}
shift
frames='' backends=cpu,fp16,int8
while (( $# )); do
  case $1 in
    --frames) (( $# >= 2 )) || md_die "$usage"; frames=$2; shift 2 ;;
    --backends) (( $# >= 2 )) || md_die "$usage"; backends=$2; shift 2 ;;
    *) md_die "$usage" ;;
  esac
done
[[ -f $pack/pack.json && -f $pack/stage2.json ]] || md_die "$pack 不是阶段 2 评估包（缺 pack.json 或 stage2.json）。"
typeset -A model=(cpu yolo11n_toys_split_320.onnx fp16 yolo11n_toys_fp16.rknn int8 yolo11n_toys_int8.rknn)
for b in ${(s:,:)backends}; do
  [[ -n ${model[$b]:-} ]] || md_die "未知的档 $b（只有 cpu、fp16、int8）。"
  [[ -f $pack/models/${model[$b]} ]] || md_die "评估包里缺 models/${model[$b]}。"
done
[[ -z $frames || $frames == <-> ]] || md_die "$usage"
md_port_busy 7901 || md_die "7901 端口没有头部相机；先在终端 A 运行 scripts/sim/stage1/run_arena_body.zsh --camera --board。"
md_port_busy 7811 || md_die "7811 端口没有测试端口；终端 A 的仿真身体没在运行。"
ssh "$MD_BOARD_HOST" "test -S ${(q)MD_BOARD_ROOT}/run/robotd.sock" ||
  md_die "板上没有 robotd 控制口；先在终端 B 运行 scripts/board/stage0/run_board_robotd.zsh s2。"

export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 (as run_d5_mac.zsh)
run=$(md_run_dir s2f_board_deploy) || exit 1
bundle=$run/bundle
md_bundle "$bundle" || md_die "打包代码失败。"
mkdir -p "$bundle/configs/vision" "$bundle/configs/control" "$bundle/models"
cp "$pack/stage2.json" "$bundle/configs/vision/stage2.json" &&
  cp "$MD_ROOT/configs/control/stage1.json" "$bundle/configs/control/stage1.json" || md_die "复制配置失败。"
for b in ${(s:,:)backends}; do
  cp "$pack/models/${model[$b]}" "$bundle/models/" || md_die "复制模型失败。"
done
md_sha256sums "$bundle" || md_die "生成 SHA256SUMS 失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}
pin=${MD_BOARD_CPUS:+taskset -c ${(q)MD_BOARD_CPUS} }
remote="set -o pipefail
cd ${(q)remote_bundle} &&
sha256sum -c --quiet SHA256SUMS &&
mkdir -p ${(q)remote_out} || exit 1
sudo -n true 2>/dev/null || {
  echo '读 NPU 负载要 sudo：输入板子 linaro 用户的密码（不想输就连按三次回车，只是不记录 NPU 负载）。'
  sudo -v || echo '没有 sudo：这次不记录 NPU 负载。'
}"
for b in ${(s:,:)backends}; do
  remote+="
echo '== $b'
MD_ROOT=\"\$PWD\" PYTHONPATH=\"\$PWD\" $pin${(q)MD_BOARD_PY} -u -m microduck_toy.stage2.deploy \\
  --backend $b --model models/${model[$b]} --camera ${(q)MD_SIM_HOST}:7901 \\
  --socket ${(q)MD_BOARD_ROOT}/run/robotd.sock ${frames:+--frames $frames} --out ${(q)remote_out}/$b || exit 1"
done

{
  cd "$MD_ROOT"
  "$MD_PY" -m microduck_toy.stage2.deploy --place --config "$pack/stage2.json" > "$run/layout.json" ||
    { print -r -- "摆放玩具失败。"; exit 1; }
  "$MD_MJPY" -c 'import json, mujoco, platform; print(json.dumps({"mujoco": mujoco.__version__, "python": platform.python_version(), "mac": platform.mac_ver()[0]}))' \
    > "$run/mac_versions.json" || print -r -- "（没取到 Mac 端版本号）"
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || exit $?
  board_status=0
  ssh -t "$MD_BOARD_HOST" "$remote" || board_status=$?
  copy_status=0
  scp -rq "$MD_BOARD_HOST:$remote_out" "$run/board" || copy_status=$?
  print -r -- "板上退出码：$board_status；结果回传退出码：$copy_status；本地目录：$run"
  (( copy_status == 0 )) || exit 1
  "$MD_PY" -m microduck_toy.stage2.deploy --table "$run" --pack "$pack" || exit 1
  (( board_status == 0 )) || exit 1
  print -r -- "完成；部署表在 $run/deploy_table.md"
} 2>&1 | tee "$run/console.log"
exit ${pipestatus[1]}
