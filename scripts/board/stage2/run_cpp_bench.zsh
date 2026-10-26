#!/bin/zsh
# Stage 2 2H on the RK3588: build the C++ benchmark (cpp/board_bench) natively, run its unit
# tests, run md_bench on the pack's camera payloads with the INT8 model, run the Python pipeline
# on the same payloads, and compare the two on this Mac. No robot, no camera; SSH and scp only.
#   run_cpp_bench.zsh runs/<UTC>_s2e_pack
# rknn_api.h is Rockchip's (not in git): $MD_ROOT/offline/rknn-2.3.2/include/rknn_api.h, downloaded
# there from GitHub (airockchip/rknn-toolkit2 v2.3.2, sha256 checked) the first time.
# Output: runs/<UTC>_s2h_cpp_bench/{bundle/, board/{build.log, unit_tests.log, cpp.jsonl, ref.jsonl,
# cpp_inputs/, ref_inputs/}, compare.json, console.log}.
set -uo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

(( $# == 1 )) || md_die '用法：run_cpp_bench.zsh runs/<UTC>_s2e_pack'
pack=${1:A}
[[ -f $pack/cpp/frames.tsv && -d $pack/cpp/uyvy ]] || md_die "$pack 里没有 C++ 帧包（cpp/frames.tsv、cpp/uyvy/）。"
[[ -f $pack/models/yolo11n_toys_int8.rknn ]] || md_die "评估包里缺 models/yolo11n_toys_int8.rknn。"

rknn_dir=$MD_ROOT/offline/rknn-2.3.2
rknn_sha=c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a
if [[ ! -f $rknn_dir/include/rknn_api.h ]]; then
  print -r -- "下载 rknn_api.h（Rockchip 头文件，2.3.2）到 $rknn_dir/include/"
  mkdir -p "$rknn_dir/include" &&
    curl -fsSL -o "$rknn_dir/include/rknn_api.h.part" \
      https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h ||
    md_die "下载 rknn_api.h 失败。"
  got=$(shasum -a 256 "$rknn_dir/include/rknn_api.h.part") || md_die "算不出 sha256。"
  [[ ${got%% *} == $rknn_sha ]] || md_die "rknn_api.h 的 sha256 不对：${got%% *}"
  mv "$rknn_dir/include/rknn_api.h.part" "$rknn_dir/include/rknn_api.h"
fi

export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 (as run_d5_mac.zsh)
run=$(md_run_dir s2h_cpp_bench) || exit 1
bundle=$run/bundle
md_bundle "$bundle" || md_die "打包代码失败。"
mkdir -p "$bundle/configs/vision" "$bundle/models" "$bundle/cpp" "$bundle/rknn/include"
cp "$pack/stage2.json" "$bundle/configs/vision/stage2.json" &&
  cp "$pack/models/yolo11n_toys_int8.rknn" "$bundle/models/" &&
  cp "$rknn_dir/include/rknn_api.h" "$bundle/rknn/include/" &&
  cp -R "$pack/cpp" "$bundle/frames" &&
  cp -R "$MD_ROOT/cpp/board_bench" "$bundle/cpp/board_bench" || md_die "复制文件失败。"
rm -rf "$bundle/cpp/board_bench/build"
md_sha256sums "$bundle" || md_die "生成 SHA256SUMS 失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}
src=cpp/board_bench
remote="set -o pipefail
cd ${(q)remote_bundle} &&
sha256sum -c --quiet SHA256SUMS &&
mkdir -p ${(q)remote_out}/cpp_inputs ${(q)remote_out}/ref_inputs || exit 1
echo '== 编译'
if command -v cmake >/dev/null; then
  { cmake -S $src -B build -DRKNN_ROOT=\"\$PWD/rknn\" && cmake --build build -j4; } > ${(q)remote_out}/build.log 2>&1
else
  mkdir -p build && {
    g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off -I$src/include -I$src/tests \\
      $src/tests/unit_tests.cpp $src/src/uyvy.cpp $src/src/frame_pack.cpp $src/src/preprocess.cpp \\
      $src/src/postprocess.cpp -o build/unit_tests &&
    g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off -I$src/include -Irknn/include \\
      $src/src/main.cpp $src/src/rknn_model.cpp $src/src/uyvy.cpp $src/src/frame_pack.cpp \\
      $src/src/preprocess.cpp $src/src/postprocess.cpp -lrknnrt -o build/md_bench
  } > ${(q)remote_out}/build.log 2>&1
fi || { tail -30 ${(q)remote_out}/build.log; echo '编译失败；完整日志在 build.log'; exit 2; }
echo '== 单元测试'
./build/unit_tests | tee ${(q)remote_out}/unit_tests.log
unit=\$?
echo '== C++ 跑帧包'
./build/md_bench --pack frames --model models/yolo11n_toys_int8.rknn --out ${(q)remote_out}/cpp.jsonl \\
  --dump-inputs ${(q)remote_out}/cpp_inputs --dump-n 20 || exit 3
echo '== Python 跑同样的帧'
MD_ROOT=\"\$PWD\" PYTHONPATH=\"\$PWD\" ${(q)MD_BOARD_PY} -u -m microduck_toy.stage2.cpp_bench reference \\
  --pack frames --model models/yolo11n_toys_int8.rknn --out ${(q)remote_out}/ref.jsonl \\
  --dump-inputs ${(q)remote_out}/ref_inputs --dump-n 20 || exit 4
exit \$unit"

{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || exit $?
  board_status=0
  ssh "$MD_BOARD_HOST" "$remote" || board_status=$?
  copy_status=0
  scp -rq "$MD_BOARD_HOST:$remote_out" "$run/board" || copy_status=$?
  print -r -- "板上退出码：$board_status（1 = 单元测试没全过）；结果回传退出码：$copy_status；本地目录：$run"
  (( copy_status == 0 )) || exit 1
  (( board_status == 0 || board_status == 1 )) || exit 1
  cd "$MD_ROOT"
  "$MD_PY" -m microduck_toy.stage2.cpp_bench compare --cpp "$run/board/cpp.jsonl" \
    --ref "$run/board/ref.jsonl" --cpp-inputs "$run/board/cpp_inputs" \
    --ref-inputs "$run/board/ref_inputs" --out "$run/compare.json" || exit 1
  (( board_status == 0 )) || exit 1
  print -r -- "完成；对拍结果在 $run/compare.json"
} 2>&1 | tee "$run/console.log"
exit ${pipestatus[1]}
