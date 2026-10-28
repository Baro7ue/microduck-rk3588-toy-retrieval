#!/bin/zsh
# Stage 4 (plan 4A/4B, terminal D): sample the board's CPU, memory and NPU load while board batches run.
#   sample_board_load.zsh [--period 1.0]
# Read-only on the board: copies scripts/stage4/board_load.py there (checked by sha256) and runs it on
# the little cores (MD_BOARD_LOAD_CPUS, default 0-3; the brain has 4-6 and robotd 7) until Ctrl-C, then
# copies the rows back. The board's clock is put next to the Mac's (meta.json offset_s), so
# scripts/stage4/run_tables.zsh can give each board group the samples taken during its episodes.
# Output: runs/<UTC>_board_load/{meta.json, console.log, board/load.csv, board/summary.json}.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"
zmodload zsh/datetime

usage='用法：sample_board_load.zsh [--period 秒]（在终端 D 里开着，板端批量跑完后按 Ctrl-C）'
period=1.0
while (( $# )); do
  case $1 in
    --period) (( $# >= 2 )) || md_die "--period 后面要跟一个值。$usage"; period=$2; shift 2 ;;
    -h|--help) print -r -- "$usage"; exit 0 ;;
    *) md_die "不认识的参数：$1。$usage" ;;
  esac
done
[[ $period == <->(|.<->) ]] || md_die "--period 要是一个数（秒），比如 1 或 0.5。"
cpus=${MD_BOARD_LOAD_CPUS-0-3}

run=$(md_run_dir board_load) || exit 1
bundle=$run/bundle
mkdir -p "$bundle" && cp "$MD_ROOT/scripts/stage4/board_load.py" "$bundle/" &&
  md_sha256sums "$bundle" || md_die "准备脚本失败。"
remote_bundle=$MD_BOARD_ROOT/bundles/${run:t}
remote_out=$remote_bundle/out

# offset_s = Mac clock - board clock, from one ssh round trip (the middle of it).
clock_offset() {
  local t0 t1 b
  t0=$EPOCHREALTIME
  b=$(ssh -o BatchMode=yes "$MD_BOARD_HOST" 'date +%s.%N') || return 1
  t1=$EPOCHREALTIME
  printf '%.6f\n' $(( (t0 + t1) / 2.0 - b ))      # printf: zsh prints a whole float as "5.", not JSON
}
before=$(clock_offset) || md_die "ssh $MD_BOARD_HOST 连不上；先检查 ssh -o BatchMode=yes $MD_BOARD_HOST true。"

print -r -- "采样目录：$run（板上 $remote_out，核 $cpus）"
{
  ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
    scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" &&
    ssh -t "$MD_BOARD_HOST" "cd ${(q)remote_bundle} && sha256sum -c --quiet SHA256SUMS &&
      ${cpus:+taskset -c $cpus} nice -n 5 python3 -u board_load.py --out ${(q)remote_out} --period $period"
} 2>&1 | ( trap '' INT; tee "$run/console.log" )
rc=${pipestatus[1]}

after=$(clock_offset) || after=''
scp -rq "$MD_BOARD_HOST:$remote_out" "$run/board" || md_die "没能把 $remote_out 拷回来（退出码 $?）；板上的文件还在。"
offset=$before
[[ -n $after ]] && offset=$(printf '%.6f' $(( (before + after) / 2.0 )))
cat > "$run/meta.json" <<JSON
{
 "about": "Stage 4 board load samples (scripts/stage4/sample_board_load.zsh); offset_s = Mac clock - board clock",
 "host": "$MD_BOARD_HOST",
 "remote_dir": "$remote_out",
 "period_s": $period,
 "cpus": "$cpus",
 "offset_s": $offset,
 "offset_s_before": $before,
 "offset_s_after": ${after:-null},
 "exit_status": $rc
}
JSON
print -r -- "负载样本已拷回：$run/board/load.csv（Mac 时钟 − 板子时钟 = ${offset} s）"
exit 0
