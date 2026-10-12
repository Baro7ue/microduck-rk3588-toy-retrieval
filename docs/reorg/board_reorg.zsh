#!/bin/zsh
# Reorganise the RK3588 home folder into ~/microduck/ (steps in docs/reorg/board_steps.md). Runs on the
# Mac and drives docs/reorg/board_reorg_remote.sh on the board over SSH, from a checksummed bundle.
#   precheck | inventory   read-only report  -> runs/<UTC>_board_reorg_<command>/<command>.txt
#   backup [NAME]          copy the board's evidence to evidence/NAME (default board_home_<UTC date>)
#                          and compare SHA-256 on both sides; B2 refuses to run before this passed
#   plan STEP              list what STEP would change, change nothing      (STEP: B2 B4 B6 B7)
#   apply STEP | undo STEP | status
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：board_reorg.zsh precheck | inventory | backup [名字] | plan 步骤 | apply 步骤 | undo 步骤 | status（步骤：B2 B4 B6 B7）'
cmd=${1:-}
(( $# )) && shift
step=''
case $cmd in
  plan|apply|undo) (( $# == 1 )) || md_die "$usage"; step=${(U)1}; [[ $step == B[2467] ]] || md_die "$usage" ;;
  backup) (( $# <= 1 )) || md_die "$usage" ;;
  precheck|inventory|status) (( $# == 0 )) || md_die "$usage" ;;
  *) md_die "$usage" ;;
esac

if [[ $cmd == backup ]]; then
  name=${1:-board_home_$(TZ=UTC date +%Y%m%d)}
  [[ $name =~ '^board_home_[a-z0-9_]+$' ]] ||
    md_die "备份名要以 board_home_ 开头，只用小写字母、数字和下划线：$name"
  dest=$MD_ROOT/evidence/$name
  [[ -e $dest ]] && md_die "$dest 已存在；换个 board_home_ 开头的名字，或先看看里面是什么。"
fi
if [[ $cmd == apply && $step == B2 ]]; then
  verified=("$MD_ROOT"/evidence/board_home_*/backup_verified.txt(N))
  (( $#verified )) || md_die "还没有核对通过的板端备份；先运行 board_reorg.zsh backup（B1）。"
fi
if [[ $cmd == apply && $step == B6 && $MD_BOARD_ROBOTD != */microduck/runtime/bin/robotd ]]; then
  md_die "env.sh 还没切到新布局（MD_BOARD_ROBOTD=$MD_BOARD_ROBOTD）；先完成 B5：应用 board_switch.patch 并通过冒烟测试。"
fi

run=$(md_run_dir board_reorg_$cmd${step:+_${(L)step}}) || exit 1
bundle=$run/bundle
mkdir -p "$bundle" &&
  cp "${0:A:h}/board_reorg_remote.sh" "$MD_ROOT/configs/board/venv_requirements_lock.txt" \
    "$MD_ROOT/configs/board/README_board.md" "$bundle/" &&
  md_sha256sums "$bundle" || md_die "准备脚本包失败。"
remote_tmp=${MD_REORG_REMOTE_TMP:-/tmp/microduck_reorg}   # tests use a local folder
remote_bundle=$remote_tmp/${run:t}
ssh "$MD_BOARD_HOST" "mkdir -p ${(q)remote_tmp}" && scp -rq "$bundle" "$MD_BOARD_HOST:$remote_bundle" ||
  md_die "脚本包传到板子失败。"

# board WORDS...: run board_reorg_remote.sh WORDS on the board, after checking the bundle.
board() {
  ssh "$MD_BOARD_HOST" "cd ${(q)remote_bundle} && sha256sum -c --quiet SHA256SUMS && bash board_reorg_remote.sh $*"
}

case $cmd in
  precheck|inventory)
    board $cmd > "$run/$cmd.txt" 2>&1
    board_status=$?
    print -r -- "已保存到 $run/$cmd.txt（退出码 $board_status）"
    exit $board_status ;;
  backup)
    {
      mkdir -p "$dest" || md_die "建不了 $dest"
      board backup-tar | tar -xf - -C "$dest" || md_die "从板子复制失败；$dest 里可能只有一部分，不算已核对。"
      board backup-sums > "$run/board_sha256.txt" || md_die "板上计算校验和失败。"
      (cd "$dest" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256) |
        sed 's#  \./#  #' > "$run/mac_sha256.txt" || md_die "Mac 上计算校验和失败。"
      LC_ALL=C sort -k2 "$run/board_sha256.txt" > "$run/board_sorted.txt"
      LC_ALL=C sort -k2 "$run/mac_sha256.txt" > "$run/mac_sorted.txt"
      cmp -s "$run/board_sorted.txt" "$run/mac_sorted.txt" ||
        md_die "两边校验和不一致，见 $run/board_sha256.txt 和 mac_sha256.txt；$dest 不算已核对。"
      cp "$run/board_sha256.txt" "$dest/board_sha256.txt" &&
        print -r -- "核对通过：$(wc -l < "$run/board_sha256.txt" | tr -d ' ') 个文件，两边 SHA-256 一致；$(TZ=UTC date +%Y-%m-%dT%H:%M:%SZ)；运行目录 runs/${run:t}" \
          > "$dest/backup_verified.txt" || md_die "写核对记录失败。"
      print -r -- "$(<"$dest/backup_verified.txt")"
      print -r -- "备份在 $dest。下一步：python3 docs/reorg/reorg.py manifest，然后提交 evidence/MANIFEST.tsv。"
    } 2>&1 | tee "$run/console.log"
    exit ${pipestatus[1]} ;;
  *)
    board $cmd $step 2>&1 | tee "$run/console.log"
    exit ${pipestatus[1]} ;;
esac
