#!/bin/zsh
# Stage 0 robotd on this Mac, against scripts/sim/stage0/run_probe_body.zsh.
# CONFIG is one of configs/robotd/stage0/*.toml.in: s1, s1u (alpha_stand + limp_fall),
# s2, s2u (velstand); "u" = upstream tuning. Rendered into runs/<UTC>_s0_robotd_CONFIG/robotd.toml.
# Socket: $MD_MAC_STATE/robotd.sock. The test scripts send robot.enable themselves.
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

(( $# == 1 )) || md_die '用法：run_robotd.zsh s1|s1u|s2|s2u'
template_file=$MD_ROOT/configs/robotd/stage0/$1.toml.in
[[ -f $template_file ]] || md_die "没有这个配置：$1（可选 s1、s1u、s2、s2u）"
md_require_commit "$MD_RUNTIME" "$MD_RUNTIME_COMMIT"
robotd=$MD_RUNTIME/target/debug/robotd
[[ -x $robotd ]] || md_die "找不到 $robotd；先运行 scripts/sim/build_mac_runtime.zsh。"
if ! md_port_busy 7801; then md_die "7801 端口没有仿真身体；先运行 scripts/sim/stage0/run_probe_body.zsh。"; fi
ort=($MD_RL/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.*.dylib(N))
(( $#ort == 1 )) || md_die "在 microduck_rl 的 .venv 里找不到唯一的 libonnxruntime dylib。"

template=$(<"$template_file")
for line in "${(@f)template}"; do
  if [[ $line =~ '@POLICY_DIR@/([^"]+)' ]] && [[ ! -f $MD_MODELS/policies/$match[1] ]]; then
    md_die "缺少策略文件 models/policies/$match[1]（见 docs/guides/2026-10-13_stage0_measurements.md 第 1 步）。"
  fi
done

run=$(md_run_dir s0_robotd_$1) || exit 1
print -r -- "${template//@POLICY_DIR@/$MD_MODELS/policies}" > "$run/robotd.toml"
mkdir -p "$MD_MAC_STATE"
export DUCK_RUNTIME_DIR=$MD_MAC_STATE
export ORT_DYLIB_PATH=$ort[1]
print -r -- "启动 robotd（配置 $1），连接 127.0.0.1:7801；控制接口 $MD_MAC_STATE/robotd.sock；日志 $run/robotd.log"
"$robotd" --sim 127.0.0.1:7801 --params "$run/robotd.toml" --socket "$MD_MAC_STATE/robotd.sock" \
  2>&1 | tee "$run/robotd.log"
