#!/bin/zsh
# Stage 2 2E on the RK3588: both RKNN files (FP16, INT8) on the NPU over the pack's test, test_far
# and test_walk frames, then the scores on this Mac. No robot, no camera; SSH and scp only.
#   run_eval_board.zsh runs/<UTC>_s2e_pack
# The code, the config, the two models and the frames go as a checksummed bundle to
# $MD_BOARD_ROOT/bundles/<run>/. Output: runs/<UTC>_s2e_board_npu/{bundle/,
# board/npu_<fp16|int8>_<split>.jsonl, eval_<fp16|int8>_<test|test_far|test_walk>.json,
# compare_<fp16|int8>_<fp32|sim>.json, console.log}.
set -uo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

(( $# == 1 )) || md_die '用法：run_eval_board.zsh runs/<UTC>_s2e_pack'
pack=${1:A}
[[ -f $pack/pack.json && -d $pack/data/meta ]] || md_die "$pack 不是阶段 2 评估包（缺 pack.json 或 data/meta）。"
for m in fp16 int8; do
  [[ -f $pack/models/yolo11n_toys_$m.rknn ]] || md_die "评估包里缺 models/yolo11n_toys_$m.rknn。"
done

export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 (as run_d5_mac.zsh)
run=$(md_run_dir s2e_board_npu) || exit 1
bundle=$run/bundle
md_bundle "$bundle" || md_die "打包代码失败。"
mkdir -p "$bundle/configs/vision" "$bundle/models"
cp "$pack/stage2.json" "$bundle/configs/vision/stage2.json" &&
  cp "$pack"/models/yolo11n_toys_{fp16,int8}.rknn "$bundle/models/" &&
  cp -R "$pack/data" "$bundle/data" || md_die "复制模型或画面失败。"
md_sha256sums "$bundle" || md_die "生成 SHA256SUMS 失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}
remote="set -o pipefail
cd ${(q)remote_bundle} &&
sha256sum -c --quiet SHA256SUMS &&
mkdir -p ${(q)remote_out} || exit 1
for m in fp16 int8; do for s in test test_far test_walk; do
  echo \"NPU \$m \$s\"
  MD_ROOT=\"\$PWD\" PYTHONPATH=\"\$PWD\" ${(q)MD_BOARD_PY} -u -m microduck_toy.stage2.predict \\
    --data data --split \$s --detector rknn --model models/yolo11n_toys_\$m.rknn \\
    --out ${(q)remote_out}/npu_\${m}_\$s.jsonl || exit 1
done; done"

{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || exit $?
  board_status=0
  ssh "$MD_BOARD_HOST" "$remote" || board_status=$?
  copy_status=0
  scp -rq "$MD_BOARD_HOST:$remote_out" "$run/board" || copy_status=$?
  print -r -- "板上退出码：$board_status；结果回传退出码：$copy_status；本地目录：$run"
  (( board_status == 0 && copy_status == 0 )) || exit 1
  cd "$MD_ROOT"
  for m in fp16 int8; do
    b=$run/board/npu_$m
    "$MD_PY" -m microduck_toy.stage2.evaluate --data "$pack/data" --split test \
      --pred "${b}_test.jsonl" --out "$run/eval_${m}_test.json" || exit 1
    "$MD_PY" -m microduck_toy.stage2.evaluate --data "$pack/data" --split test,test_far \
      --pred "${b}_test.jsonl,${b}_test_far.jsonl" --out "$run/eval_${m}_test_far.json" || exit 1
    "$MD_PY" -m microduck_toy.stage2.evaluate --data "$pack/data" --split test_walk \
      --pred "${b}_test_walk.jsonl" --out "$run/eval_${m}_test_walk.json" || exit 1
    "$MD_PY" -m microduck_toy.stage2.parity --data "$pack/data" --split test --rule board \
      --ref "$pack/cloud/fp32_test.jsonl" --new "${b}_test.jsonl" --out "$run/compare_${m}_fp32.json"
    "$MD_PY" -m microduck_toy.stage2.parity --data "$pack/data" --split test --rule board \
      --ref "$pack/cloud/${m}sim_test.jsonl" --new "${b}_test.jsonl" --out "$run/compare_${m}_sim.json"
  done
  print -r -- "完成；结果在 $run"
} 2>&1 | tee "$run/console.log"
exit ${pipestatus[1]}
