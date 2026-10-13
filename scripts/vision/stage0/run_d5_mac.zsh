#!/bin/zsh
# Stage 0 d5 on this Mac: YOLO11n FP32 ONNX on the CPU over the cloud's toy frames, then compare
# with the cloud's own run (same frames, same code).
#   run_d5_mac.zsh runs/<UTC>_d5_cloud_pack
# Output: runs/<UTC>_d5_mac_cpu/{views.jsonl, summary.json, compare.json}.
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

(( $# == 1 )) || md_die '用法：run_d5_mac.zsh runs/<UTC>_d5_cloud_pack'
pack=${1:A}
[[ -f $pack/pack.json && -f $pack/cloud/views.jsonl ]] || md_die "$pack 不是云端画面包（缺 pack.json 或 cloud/views.jsonl）。"
run=$(md_run_dir d5_mac_cpu) || exit 1
cd "$MD_ROOT"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"
"$MD_PY" -u -m microduck_toy.stage0.d5_detect run --frames "$pack" --backend cpu \
  --model "$MD_MODELS/yolo/yolo11n_coco/yolo11n_coco_fp32_320.onnx" --output-dir "$run"
exec "$MD_PY" -u -m microduck_toy.stage0.d5_detect compare --frames "$pack" --result "$run"
