#!/bin/zsh
# Run one Stage 1 test against a running arena body and robotd:
#   run_test.zsh 1a|1b|1c|1d|1f|1g --config s2 [--quick] [--parts …] [--target board]
#   run_test.zsh 1b --rescore runs/<dir>    (score a finished run again; no robot needed)
#   1a = s1a_scene (field + layouts), 1b = s1b_grasp (beak and GroundPick),
#   1c = s1c_back (backing off + odometry), 1d = s1d_range (box -> distance, colour boxes),
#   1f = s1f_approach (B + C approach with decide(), pick, from seeded starts),
#   1g = s1g_carry (pick, then stand, walk and turn with the toy in the beak)
# Output: runs/<UTC>_s1<a|b|c|d|f>_<config>/ (truth.csv, state.jsonl, events.jsonl, summary.json).
# The render check (P) does not need robotd or the body:
#   run_test.zsh parity --floor grid    (then --floor wood; 10 views each into runs/<UTC>_s1a_parity_<host>/)
#   run_test.zsh parity --compare <mine.json> <reference.json>
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_test.zsh 1a|1b|1c|1d|1f|1g --config s2 [--quick] [--parts …] [--target board]  或  run_test.zsh parity …'
(( $# >= 1 )) || md_die "$usage"
case $1 in
  1a) module=microduck_toy.stage1.s1a_scene ;;
  1b) module=microduck_toy.stage1.s1b_grasp ;;
  1c) module=microduck_toy.stage1.s1c_back ;;
  1d) module=microduck_toy.stage1.s1d_range ;;
  1f) module=microduck_toy.stage1.s1f_approach ;;
  1g) module=microduck_toy.stage1.s1g_carry ;;
  parity)
    shift
    cd "$MD_ROOT"
    exec "$MD_PY" -u -m microduck_toy.stage1.parity "$@" ;;
  *) md_die "$usage" ;;
esac
shift
if (( ! ${@[(I)--rescore]} )) && ! md_port_busy 7811; then
  md_die "7811 端口没有测试端口；先运行 scripts/sim/stage1/run_arena_body.zsh。"
fi
cd "$MD_ROOT"
exec "$MD_PY" -u -m $module "$@"
