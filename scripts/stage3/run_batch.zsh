#!/bin/zsh
# Stage 3G batch runner (plan 10): run a list of episodes one by one, one process and one folder each.
#   run_batch.zsh --target mac|board [--layouts FILE] [--seeds A-B] [--targets bus,henry,ketchup]
#                 [--backend cpu|fp16|int8] [--out runs/<dir>] [--formal] [-- more episode options]
# Episodes: every seed A..B (default: every seed in the layouts file, configs/control/stage3_layouts.json)
# x every target, seed by seed. Backend: cpu on the Mac; int8 (default), fp16 or cpu on the board.
# Batch folder (default runs/batch_<target>_<backend>_<layouts>[_<A-B>][_<targets>]):
#   batch.json   the settings, the code's sha256, every invocation   console.log   everything printed
#   summary.json the table's numbers                                  bundle/       the code it ran
#   episodes/<seed>_<target>/   one finished episode: a full run folder (truth.csv, summary.json, ...)
# Resume = run the same command again: episodes with a result are skipped. An attempt without one
# (Ctrl-C, a crash, a brain error, a sim not in real time, a missing log) is renamed
# <seed>_<target>.<interrupted|infra|incomplete><n> as evidence and run again next time; three in a row
# stop the batch. Ctrl-C stops after the running episode has saved. The table is printed either way.
# A resumed batch must run the same code: batch.json keeps the sha256 of the bundle's SHA256SUMS
# (package + configs + models) and of the layouts file. Formal seeds 1000-1019 need --formal (Stage 4).
# Needs, already running (as run_test.zsh / run_board_brain.zsh):
#   mac    A scripts/sim/stage1/run_arena_body.zsh --camera          B scripts/sim/stage0/run_robotd.zsh s2
#   board  A scripts/sim/stage1/run_arena_body.zsh --camera --board  B scripts/board/stage0/run_board_robotd.zsh s2
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_batch.zsh --target mac|board [--layouts 文件] [--seeds A-B] [--targets bus,henry,ketchup] [--backend cpu|fp16|int8] [--out runs/<目录>] [--formal] [-- 其它回合参数]'
orig=("$@")
target='' layouts=configs/control/stage3_layouts.json seeds='' targets=bus,henry,ketchup backend='' out='' formal=0
extra=()
while (( $# )); do
  case $1 in
    --target|--layouts|--seeds|--targets|--backend|--out)
      (( $# >= 2 )) || md_die "$1 后面要跟一个值。$usage"
      case $1 in
        --target)  target=$2 ;;
        --layouts) layouts=$2 ;;
        --seeds)   seeds=$2 ;;
        --targets) targets=$2 ;;
        --backend) backend=$2 ;;
        --out)     out=$2 ;;
      esac
      shift 2 ;;
    --formal) formal=1; shift ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) print -r -- "$usage"; exit 0 ;;
    *) md_die "不认识的参数：$1。$usage" ;;
  esac
done
case $target in
  mac)   : ${backend:=cpu}; [[ $backend == cpu ]] || md_die "Mac 上只有 CPU（--backend cpu）；fp16/int8 在板上的 NPU 跑。" ;;
  board) : ${backend:=int8}; [[ $backend == (cpu|fp16|int8) ]] || md_die "--backend 只能是 cpu、fp16、int8。" ;;
  *) md_die "$usage" ;;
esac
for a in $extra; do
  [[ $a == (--target|--episodes|--out|--layouts|--backend|--quick|--formal|--board-*)(|=*) ]] &&
    md_die "$a 由运行器自己给，不能放在 -- 后面。"
done

cd "$MD_ROOT"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2 for the detector's letterbox (as Stage 2)
batch() { "$MD_PY" -m microduck_toy.stage3.batch "$@" }

# ── what to run ─────────────────────────────────────────────────────────────
plan_args=(--layouts "$layouts" --targets "$targets")
[[ -n $seeds ]] && plan_args+=(--seeds "$seeds")
(( formal )) && plan_args+=(--formal)
listing=$(batch plan $plan_args) || md_die "列不出要跑的回合（原因在上面）。"
episodes=(${(f)listing})
(( ${#episodes} )) || md_die "没有要跑的回合。"
if [[ -z $out ]]; then
  out="$MD_RUNS/batch_${target}_${backend}_${${layouts:t}:r}"
  [[ -n $seeds ]] && out+="_$seeds"
  [[ $targets != bus,henry,ketchup ]] && out+="_${targets//,/+}"
fi
[[ $out == /* ]] || out=$MD_ROOT/$out

# ── checks before the first episode ─────────────────────────────────────────
md_stage3_check
md_port_busy 7811 || md_die "7811 端口没有测试端口；先在终端 A 运行 scripts/sim/stage1/run_arena_body.zsh --camera${${(M)target:#board}:+ --board}。"
md_port_busy 7901 || md_die "7901 端口没有头部相机；终端 A 要带 --camera 启动。"
if [[ $target == mac ]]; then
  [[ -S $MD_MAC_STATE/robotd.sock ]] || md_die "没有 robotd 控制口；先在终端 B 运行 scripts/sim/stage0/run_robotd.zsh s2。"
else
  ssh "$MD_BOARD_HOST" "test -S ${(q)MD_BOARD_ROOT}/run/robotd.sock" ||
    md_die "板上没有 robotd 控制口；先在终端 B 运行 scripts/board/stage0/run_board_robotd.zsh s2。"
fi

# ── the code: one bundle per batch, the same on every resume ────────────────
mkdir -p "$out/episodes" || md_die "建不了 $out。"
fresh="$out/.bundle.new"
[[ -e $fresh ]] && rm -rf -- "$fresh"
md_stage3_bundle "$fresh" || md_die "打包代码失败。"
code_sha=$(shasum -a 256 "$fresh/SHA256SUMS") || md_die "算不出代码包的 sha256。"
code_sha=${code_sha%% *}
commit=$(git rev-parse HEAD 2>/dev/null) || commit=''
dirty=no
[[ -n $(git status --porcelain -- src configs scripts models 2>/dev/null) ]] && dirty=yes
begin_args=(--target "$target" --layouts "$layouts" --episodes "${(j:,:)episodes}" --backend "$backend"
            --code-sha256 "$code_sha" "--extra=${(j: :)${(@q)extra}}" --dirty "$dirty")
[[ -n $commit ]] && begin_args+=(--commit "$commit")
(( formal )) && begin_args+=(--formal)
batch begin "$out" $begin_args || { rm -rf -- "$fresh"; exit 1 }
if [[ -d $out/bundle ]]; then rm -rf -- "$fresh"; else mv "$fresh" "$out/bundle"; fi
[[ $dirty == yes ]] && print -r -- "注意：src/configs/scripts/models 有没提交的改动；batch.json 记下了代码包的 sha256。"

if [[ $target == board ]]; then
  remote_bundle=$MD_BOARD_ROOT/bundles/${out:t}_${code_sha[1,12]}
  if ssh "$MD_BOARD_HOST" "test -e ${(q)remote_bundle}"; then
    ssh "$MD_BOARD_HOST" "cd ${(q)remote_bundle} && sha256sum -c --quiet SHA256SUMS" ||
      md_die "板上的 $remote_bundle 和本机的代码对不上；把它挪走再运行。"
  else
    ssh "$MD_BOARD_HOST" "mkdir -p ${(q)MD_BOARD_ROOT}/bundles" &&
      scp -rq "$out/bundle" "$MD_BOARD_HOST:$remote_bundle" &&
      ssh "$MD_BOARD_HOST" "cd ${(q)remote_bundle} && sha256sum -c --quiet SHA256SUMS" ||
      md_die "代码包传到板上或在板上校验失败。"
  fi
  print -r -- "代码包已在板上校验：$remote_bundle"
  # A brain left over from an earlier Ctrl-C would still be walking; '[m]' keeps pkill off its own shell.
  ssh "$MD_BOARD_HOST" "pkill -INT -f '[m]icroduck_toy.stage3.brain'; true"
  remote_root=$MD_BOARD_ROOT/run/${out:t}
fi

# ── one episode at a time ───────────────────────────────────────────────────
# set_aside DIR STATUS: rename an attempt without a result to DIR.STATUS<n> (made if the process left none).
set_aside() {
  local n=1
  while [[ -e $1.$2$n ]]; do (( n++ )); done
  if [[ -d $1 ]]; then mv "$1" "$1.$2$n"; else mkdir -p "$1.$2$n"; fi
  print -r -- "$1.$2$n"
}

interrupted=0
TRAPINT() { interrupted=1; return 0 }     # Ctrl-C: the episode process saves and exits; then stop here
streak=0 skipped=0 ran=0 stop=''
print -r -- "批量：${#episodes} 个回合 → $out" | tee -a "$out/console.log"
for item in $episodes; do
  (( interrupted )) && { stop=interrupted; break }
  seed=${item%%:*} toy=${item#*:}
  dir=$out/episodes/${seed}_$toy
  if [[ -e $dir ]]; then
    line=$(batch status "$dir") || line=incomplete
    st=${line%% *}
    if [[ $st == done ]]; then (( skipped++ )); continue; fi
    set_aside "$dir" "$st" >/dev/null
  fi
  (( ran++ ))
  print -r -- "== [${episodes[(i)$item]}/${#episodes}] $seed $toy  $(TZ=UTC date '+%H:%M:%SZ')" | tee -a "$out/console.log"
  cmd=("$MD_PY" -u -m microduck_toy.stage3.episode 3e --source stream --layouts "$layouts" --episodes "$item"
       --out "$dir" --backend "$backend")
  (( formal )) && cmd+=(--formal)
  if [[ $target == board ]]; then
    cmd+=(--target board --board-bundle "$remote_bundle"
          --board-out "$remote_root/${seed}_${toy}_$(TZ=UTC date '+%Y%m%dT%H%M%SZ')" --board-py "$MD_BOARD_PY"
          --board-socket "$MD_BOARD_ROOT/run/robotd.sock" --board-host "$MD_BOARD_HOST" --sim-host "$MD_SIM_HOST"
          --board-cpus "${MD_BOARD_CPUS-4-6}" --socket "$MD_STARTUP_SOCKET")
  fi
  cmd+=($extra)
  log=$out/episodes/.console.${seed}_$toy
  : >| "$log"
  # tee ignores SIGINT, so a Ctrl-C reaches the episode process and the lines it prints while saving.
  "${cmd[@]}" 2>&1 | ( trap '' INT; tee -a "$out/console.log" "$log" )
  rc=${pipestatus[1]}
  line=$(batch status "$dir" 2>/dev/null) || line='incomplete 读不出回合状态'
  st=${line%% *} why=${line#* }
  if [[ $st == done ]]; then
    mv "$log" "$dir/console.log"
    streak=0
    print -r -- "   → $why" | tee -a "$out/console.log"
    (( interrupted )) && { stop=interrupted; break }
    continue
  fi
  (( interrupted || rc == 130 )) && st=interrupted
  aside=$(set_aside "$dir" "$st")
  mv "$log" "$aside/console.log"
  print -r -- "   → 没算数（$why）；留在 ${aside:t}，下次补跑" | tee -a "$out/console.log"
  if [[ $st == interrupted ]]; then
    stop=interrupted
    break
  fi
  if (( ++streak >= 3 )); then
    stop=infra
    break
  fi
done

# ── the table ───────────────────────────────────────────────────────────────
print -r -- "本次跑了 $ran 个，跳过已有结果的 $skipped 个。" | tee -a "$out/console.log"
batch table "$out" | tee -a "$out/console.log"
left=${pipestatus[1]}
again="./scripts/stage3/run_batch.zsh ${(j: :)${(@q)orig}}"
case $stop in
  interrupted)
    print -r -- "已中断。接着跑：再运行同一条命令（有结果的回合会跳过）："
    print -r -- "  $again"
    exit 130 ;;
  infra)
    print -r -- "连续 3 个回合没算数，已停止：先看上面最后几个回合的原因（console.log 在各自的文件夹里），"
    print -r -- "修好以后再运行同一条命令接着跑：$again"
    exit 2 ;;
esac
if (( left != 0 )); then
  print -r -- "还有回合没有结果；再运行同一条命令补跑：$again"
  exit 1
fi
exit 0
