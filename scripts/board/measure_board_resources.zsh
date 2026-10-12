#!/bin/zsh
# Sample board CPU, memory and robotd loop rate for 11 minutes (stdlib only; no motion, no installs),
# then copy the evidence back. Needs run_board_robotd.zsh already running.
# Output: runs/<UTC>_board_resources/{console.log, board/}.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

run=$(md_run_dir board_resources) || exit 1
bundle=$run/bundle
mkdir -p "$bundle" && cp "$MD_ROOT/scripts/board/measure_board_resources.py" "$bundle/" &&
  md_sha256sums "$bundle" || md_die "准备脚本失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" &&
    ssh -t "$MD_BOARD_HOST" "cd ${(q)remote_bundle} && sha256sum -c --quiet SHA256SUMS &&
      MD_BOARD_ROOT=${(q)MD_BOARD_ROOT} python3 -u measure_board_resources.py --duration 660"
} 2>&1 | tee "$run/console.log"
measure_status=$?

# The board prints its evidence directory; copy exactly that one back.
remote_dir=''
for line in "${(@f)$(<"$run/console.log")}"; do
  [[ $line == 资源证据目录：* ]] && remote_dir=${${line#资源证据目录：}%$'\r'}
done
[[ -n $remote_dir ]] || md_die "控制台输出里没有资源证据目录；退出码 $measure_status。"
scp -rq "$MD_BOARD_HOST:$remote_dir" "$run/board" || exit $?
print -r -- "资源证据已复制到 $run/board"
exit $measure_status
