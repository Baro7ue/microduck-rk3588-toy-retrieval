#!/bin/zsh
# Run YOLO11n on the board against the Mac head camera ($MD_SIM_HOST:7901) and copy the evidence back.
# Default: CPU, FP32 ONNX. --npu: RKNN FP16 on the NPU. Uses camera TCP and SSH only; no motion.
# The code and model are sent as a checksummed bundle to $MD_BOARD_ROOT/bundles/<run>/.
# Output: runs/<UTC>_board_yolo_<cpu|npu>/{console.log, board/}.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：probe_board_yolo_camera.zsh [--npu] [--duration 秒]'
backend=cpu duration=30
while (( $# )); do
  case $1 in
    --npu) backend=npu; shift ;;
    --duration) (( $# >= 2 )) || md_die "$usage"; duration=$2; shift 2 ;;
    *) md_die "$usage" ;;
  esac
done
models=$MD_MODELS/yolo/yolo11n_coco
if [[ $backend == npu ]]; then
  module=microduck_toy.vision.rknn_runner
  files=(yolo11n_coco_fp16_320.rknn conversion.json)
else
  module=microduck_toy.vision.yolo_camera_probe
  files=(yolo11n_coco_fp32_320.onnx export.json)
fi

run=$(md_run_dir board_yolo_$backend) || exit 1
bundle=$run/bundle
md_bundle "$bundle" || md_die "打包代码失败。"
mkdir -p "$bundle/models"
for file in $files; do
  cp "$models/$file" "$bundle/models/" || md_die "缺少模型文件 $models/$file。"
done
md_sha256sums "$bundle" || md_die "生成 SHA256SUMS 失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}
remote="set -o pipefail
cd ${(q)remote_bundle} &&
sha256sum -c --quiet SHA256SUMS &&
PYTHONPATH=\"\$PWD\" ${(q)MD_BOARD_PY} -u -m $module \\
  --model models/$files[1] --duration ${(q)duration} --output-dir ${(q)remote_out}"

{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || exit $?
  probe_status=0
  ssh -t "$MD_BOARD_HOST" "$remote" || probe_status=$?
  copy_status=0
  scp -rq "$MD_BOARD_HOST:$remote_out" "$run/board" || copy_status=$?
  print -r -- "实验退出码：$probe_status；证据回传退出码：$copy_status；本地目录：$run"
  (( probe_status == 0 )) || exit $probe_status
  exit $copy_status
} 2>&1 | tee "$run/console.log"
