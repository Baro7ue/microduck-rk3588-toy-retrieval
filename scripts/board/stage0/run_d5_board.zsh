#!/bin/zsh
# Stage 0 d5 on the RK3588: YOLO11n FP16 RKNN on the NPU over the cloud's toy frames at one distance,
# then compare with the cloud's own run on this Mac. No robot, no camera; SSH and scp only.
#   run_d5_board.zsh runs/<UTC>_d5_cloud_pack [--distance 0.5]
# The code, the model and the frames go as a checksummed bundle to $MD_BOARD_ROOT/bundles/<run>/.
# Output: runs/<UTC>_d5_board_npu/{bundle/, board/ (views.jsonl, summary.json, compare.json), console.log}.
set -uo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_d5_board.zsh runs/<UTC>_d5_cloud_pack [--distance 0.5]'
(( $# >= 1 )) || md_die "$usage"
pack=${1:A}
shift
distance=0.5
while (( $# )); do
  case $1 in
    --distance) (( $# >= 2 )) || md_die "$usage"; distance=$2; shift 2 ;;
    *) md_die "$usage" ;;
  esac
done
[[ -f $pack/pack.json && -f $pack/cloud/views.jsonl ]] || md_die "$pack 不是云端画面包（缺 pack.json 或 cloud/views.jsonl）。"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"

run=$(md_run_dir d5_board_npu) || exit 1
bundle=$run/bundle
md_bundle "$bundle" || md_die "打包代码失败。"
mkdir -p "$bundle/models"
for file in yolo11n_coco_fp16_320.rknn conversion.json; do
  cp "$MD_MODELS/yolo/yolo11n_coco/$file" "$bundle/models/" || md_die "缺少模型文件 $file。"
done
"$MD_PY" -m microduck_toy.stage0.d5_detect subset --frames "$pack" --distances "$distance" \
  --output-dir "$bundle/pack" || md_die "挑选 $distance m 的画面失败。"
md_sha256sums "$bundle" || md_die "生成 SHA256SUMS 失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}
remote="set -o pipefail
cd ${(q)remote_bundle} &&
sha256sum -c --quiet SHA256SUMS &&
PYTHONPATH=\"\$PWD\" ${(q)MD_BOARD_PY} -u -m microduck_toy.stage0.d5_detect run --frames pack \\
  --backend npu --model models/yolo11n_coco_fp16_320.rknn --output-dir ${(q)remote_out}"

{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || exit $?
  probe_status=0
  ssh "$MD_BOARD_HOST" "$remote" || probe_status=$?
  copy_status=0
  scp -rq "$MD_BOARD_HOST:$remote_out" "$run/board" || copy_status=$?
  print -r -- "板上退出码：$probe_status；结果回传退出码：$copy_status；本地目录：$run"
  (( probe_status == 0 && copy_status == 0 )) || exit 1
  cd "$MD_ROOT" && "$MD_PY" -m microduck_toy.stage0.d5_detect compare --frames "$bundle/pack" \
    --result "$run/board"
} 2>&1 | tee "$run/console.log"
exit ${pipestatus[1]}
