#!/bin/zsh
# Start robotd on the RK3588 against the Mac sim body and forward its control socket to this Mac
# ($MD_STARTUP_SOCKET). Needs the sim body on $MD_SIM_HOST:7801 (scripts/sim/run_sim_body.zsh).
# Config: configs/robotd/board.toml.in, rendered with the policies it names (from models/policies/)
# into a checksummed bundle sent to $MD_BOARD_ROOT/bundles/<run>/; the board checks it first.
# Ctrl+C stops robotd and the socket forward together; robotd's log is then copied back.
# Output: runs/<UTC>_board_robotd/{bundle/, board/robotd.log}.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

template=$(<"$MD_ROOT/configs/robotd/board.toml.in") || md_die "读不到 configs/robotd/board.toml.in。"
# Every "@POLICY_DIR@/<file>" in the template names a policy to send.
policies=()
for line in "${(@f)template}"; do
  [[ $line =~ '@POLICY_DIR@/([^"]+)' ]] && policies+=($match[1])
done
policies=(${(u)policies})
for policy in $policies; do
  [[ -f $MD_MODELS/policies/$policy ]] || md_die "缺少策略文件 models/policies/$policy。"
done

run=$(md_run_dir board_robotd) || exit 1
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$MD_BOARD_ROOT/run/${run:t}
bundle=$run/bundle
mkdir -p "$bundle/policies" || md_die "建不了 $bundle。"
for policy in $policies; do
  cp "$MD_MODELS/policies/$policy" "$bundle/policies/" || md_die "复制 $policy 失败。"
done
print -r -- "${template//@POLICY_DIR@/$remote_bundle/policies}" > "$bundle/robotd.toml"
md_sha256sums "$bundle" || md_die "生成 SHA256SUMS 失败。"

remote="set -o pipefail
cd ${(q)remote_bundle} &&
sha256sum -c --quiet SHA256SUMS &&
mkdir -p ${(q)remote_out} &&
export ORT_DYLIB_PATH=${(q)MD_BOARD_ORT} DUCK_RUNTIME_DIR=${(q)MD_BOARD_ROOT}/run &&
${(q)MD_BOARD_ROBOTD} --sim ${(q)MD_SIM_HOST}:7801 --params ${(q)remote_bundle}/robotd.toml \\
  --socket \"\$DUCK_RUNTIME_DIR/robotd.sock\" \\
  2>&1 | tee ${(q)remote_out}/robotd.log"
ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
  scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" || md_die "代码包传到板子失败。"
robotd_status=0
ssh -t -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes \
  -L "$MD_STARTUP_SOCKET:$MD_BOARD_ROOT/run/robotd.sock" "$MD_BOARD_HOST" "$remote" || robotd_status=$?
copy_status=0
scp -rq "$MD_BOARD_HOST:$remote_out" "$run/board" || copy_status=$?
print -r -- "robotd 退出码：$robotd_status；日志回传退出码：$copy_status；本地目录：$run"
(( robotd_status == 0 )) || exit $robotd_status
exit $copy_status
