#!/bin/zsh
# Stage 2 2E on this Mac: the parity check. Renders the pack's 200 parity views again on this Mac,
# runs the split ONNX on the CPU over them and over the cloud's own frames, and compares both with
# the cloud's boxes (same box count per frame, scores within 0.01).
#   run_parity_mac.zsh runs/<UTC>_s2e_pack
# Output: runs/<UTC>_s2e_mac_parity/{render/, mac_cloud_frames.jsonl, mac_mac_frames.jsonl,
#         same_frames.json, mac_render.json, console.log}.
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

(( $# == 1 )) || md_die '用法：run_parity_mac.zsh runs/<UTC>_s2e_pack'
pack=${1:A}
[[ -f $pack/pack.json && -f $pack/parity_ids.txt ]] || md_die "$pack 不是阶段 2 评估包（缺 pack.json 或 parity_ids.txt）。"
cmp -s "$pack/stage2.json" "$MD_ROOT/configs/vision/stage2.json" ||
  md_die "configs/vision/stage2.json 和评估包里的不一样；先把仓库更新到评估包对应的提交。"
model=$pack/models/yolo11n_toys_split_320.onnx
[[ -f $model ]] || md_die "评估包里缺 $model。"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 (as run_d5_mac.zsh)
run=$(md_run_dir s2e_mac_parity) || exit 1
cd "$MD_ROOT"
same=0 render=0
{
  print -r -- "1/3 在本机重新渲染 200 个视角"
  "$MD_PY" -u -m microduck_toy.stage2.datagen --out "$run/render" --splits test --count 200
  print -r -- "2/3 本机 CPU 检测：云端的画面，本机的画面"
  "$MD_PY" -u -m microduck_toy.stage2.predict --data "$pack/data" --split test \
    --ids "$pack/parity_ids.txt" --detector onnx-split --model "$model" --out "$run/mac_cloud_frames.jsonl"
  "$MD_PY" -u -m microduck_toy.stage2.predict --data "$run/render" --split test \
    --detector onnx-split --model "$model" --out "$run/mac_mac_frames.jsonl"
  print -r -- "3/3 和云端的框比较"
  "$MD_PY" -m microduck_toy.stage2.parity --data "$pack/data" --split test --ids "$pack/parity_ids.txt" \
    --ref "$pack/cloud/fp32_test.jsonl" --new "$run/mac_cloud_frames.jsonl" --rule mac \
    --out "$run/same_frames.json" || same=$?
  "$MD_PY" -m microduck_toy.stage2.parity --data "$pack/data" --split test --ids "$pack/parity_ids.txt" \
    --ref "$pack/cloud/fp32_test.jsonl" --new "$run/mac_mac_frames.jsonl" --rule mac \
    --ref-images "$pack/data/images/test" --new-images "$run/render/images/test" \
    --out "$run/mac_render.json" || render=$?
  print -r -- "同一批画面：$( (( same == 0 )) && print 通过 || print 不通过 )；本机重新渲染：$( (( render == 0 )) && print 通过 || print 不通过 )。结果在 $run"
} 2>&1 | tee "$run/console.log"
exit ${pipestatus[1]}
