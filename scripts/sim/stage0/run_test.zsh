#!/bin/zsh
# Run one Stage 0 test against a running probe body and robotd:
#   run_test.zsh 0c --config s2 [--quick] [--parts c1,c3] [--target board]
#   run_test.zsh 0c --rescore runs/<dir>    (score a finished run again; no robot needed)
#   0a = s0a_walk (head-down walking), 0b = s0b_pick (GroundPick), 0c = s0c_getup (getting up)
# Output: runs/<UTC>_s0<a|b|c>_<config>/ (truth.csv, state.jsonl, events.jsonl, summary.json).
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_test.zsh 0a|0b|0c --config s1|s1u|s2|s2u [--quick] [--parts …] [--target board]'
(( $# >= 1 )) || md_die "$usage"
case $1 in
  0a) module=microduck_toy.stage0.s0a_walk ;;
  0b) module=microduck_toy.stage0.s0b_pick ;;
  0c) module=microduck_toy.stage0.s0c_getup ;;
  *) md_die "$usage" ;;
esac
shift
if (( ! ${@[(I)--rescore]} )) && ! md_port_busy 7811; then
  md_die "7811 端口没有测试端口；先运行 scripts/sim/stage0/run_probe_body.zsh。"
fi
cd "$MD_ROOT"
exec "$MD_PY" -u -m $module "$@"
