#!/bin/zsh
# Run robotd on the Mac against a sim body started with `run_sim_body.zsh --local`.
# Config: configs/robotd/mac_sim.toml.in rendered into runs/<UTC>_mac_robotd/robotd.toml.
# Socket: $MD_MAC_STATE/robotd.sock. Motion still needs a separate enable request.
set -euo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

md_require_commit "$MD_RUNTIME" "$MD_RUNTIME_COMMIT"
robotd=$MD_RUNTIME/target/debug/robotd
[[ -x $robotd ]] || md_die "找不到 $robotd；先运行 scripts/sim/build_mac_runtime.zsh。"
if ! md_port_busy 7801; then md_die "7801 端口没有仿真身体；先运行 scripts/sim/run_sim_body.zsh --local。"; fi
ort=($MD_RL/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.*.dylib(N))
(( $#ort == 1 )) || md_die "在 microduck_rl 的 .venv 里找不到唯一的 libonnxruntime dylib。"

run=$(md_run_dir mac_robotd) || exit 1
template=$(<"$MD_ROOT/configs/robotd/mac_sim.toml.in")
print -r -- "${template//@MD_ROOT@/$MD_ROOT}" > "$run/robotd.toml"
mkdir -p "$MD_MAC_STATE"
export DUCK_RUNTIME_DIR=$MD_MAC_STATE
export ORT_DYLIB_PATH=$ort[1]
print -r -- "启动 robotd，连接 127.0.0.1:7801；控制接口 $MD_MAC_STATE/robotd.sock；日志 $run/robotd.log"
"$robotd" --sim 127.0.0.1:7801 --params "$run/robotd.toml" --socket "$MD_MAC_STATE/robotd.sock" \
  2>&1 | tee "$run/robotd.log"
