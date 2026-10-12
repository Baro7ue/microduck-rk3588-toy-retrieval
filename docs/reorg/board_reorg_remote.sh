#!/usr/bin/env bash
# Board side of docs/reorg/board_reorg.zsh; runs on the RK3588 from the checksummed bundle folder.
#   board_reorg_remote.sh precheck | inventory | status | backup-tar | backup-sums
#   board_reorg_remote.sh plan|apply|undo B2|B4|B6|B7
# Every change is logged to ~/microduck/move_log.tsv (step, operation, from, to, UTC) before it is made,
# so a step can be undone even after it stopped halfway. Nothing is overwritten: all targets are checked
# before the first change. The only deletions are __pycache__ folders (B6) and, on undo, copies this
# script made. One step at a time: apply, undo and backup hold /tmp/microduck_reorg.lock.
set -uo pipefail

H=${MD_REORG_BOARD_HOME:-$HOME}   # tests point this at a fake board home
R=$H/microduck
A=$R/archive/2026-10_ball_follow
LOG=$R/move_log.tsv
UNDONE=$R/move_log.undone.tsv
BUNDLE=$PWD
ORT=onnxruntime-linux-aarch64-1.28.0
TAB=$'\t'
USAGE='用法：board_reorg_remote.sh precheck|inventory|status|backup-tar|backup-sums，或 plan|apply|undo B2|B4|B6|B7'

die() { printf '%s\n' "$*" >&2; exit 1; }
there() { [ -e "$1" ] || [ -L "$1" ]; }

# --- what each step does: one "operation<TAB>from<TAB>to" line per change ---

ops_B2() {   # skeleton, installers out of the home root; nothing the launchers use
  local d f
  for d in "$R" "$R/runtime" "$R/runtime/bin" "$R/bundles" "$R/run" "$R/build" "$R/toolchains" \
           "$R/downloads" "$R/downloads/vision_cpu_wheels" "$R/archive" "$A"; do
    printf 'mkdir\t-\t%s\n' "$d"
  done
  for f in microduck-g3-offline.tar.gz microduck-g3-offline.tar.gz.sha256 "$ORT.tgz" "$ORT.tgz.sha256" \
           rust-1.89.0-aarch64-linux-gnu lite2-2.3.2-cp311 venv-deb12u8; do
    printf 'mv\t%s\t%s\n' "$H/$f" "$R/downloads/$f"
  done
  for f in board-baseline-20261006-qG5aeM.txt yolov5-rk3588-smoke; do
    printf 'mv\t%s\t%s\n' "$H/$f" "$A/$f"
  done
  printf 'cp\t%s\t%s\n' "$H/microduck-board/yolo11n-cpu/wheelhouse" "$R/downloads/vision_cpu_wheels/wheelhouse"
  printf 'cp\t%s\t%s\n' "$H/microduck-board/yolo11n-cpu/requirements-lock.txt" \
    "$R/downloads/vision_cpu_wheels/requirements-lock.txt"
}

ops_B4() {   # new runtime next to the old one (copies only)
  printf 'cp\t%s\t%s\n' "$H/microduck-g3-offline/microduck/target/release/robotd" "$R/runtime/bin/robotd"
  printf 'sum\t%s\t%s\n' "$R/runtime/bin/robotd" "$R/runtime/bin/robotd.sha256"
  printf 'cp\t%s\t%s\n' "$H/$ORT" "$R/runtime/$ORT"
  printf 'cp\t%s\t%s\n' "$BUNDLE/venv_requirements_lock.txt" "$R/runtime/venv_requirements_lock.txt"
  printf 'venv\t%s\t%s\n' "$R/runtime/venv_requirements_lock.txt" "$R/runtime/venv"
}

ops_B6() {   # old locations out of the home root, after the B5 smoke test
  local f
  for f in microduck-board rk3588-npu-venv "$ORT"; do
    printf 'mv\t%s\t%s\n' "$H/$f" "$A/$f"
  done
  printf 'mv\t%s\t%s\n' "$H/microduck-g3-offline" "$R/build/microduck-g3-offline"
  printf 'mv\t%s\t%s\n' "$H/rust-1.89.0" "$R/toolchains/rust-1.89.0"
  printf 'pycache\t%s\t-\n' "$A/microduck-board"
}

ops_B7() {
  printf 'cp\t%s\t%s\n' "$BUNDLE/README_board.md" "$R/README.md"
}

previous_step() { case $1 in B4) echo B2 ;; B6) echo B4 ;; B7) echo B6 ;; esac; }
next_step() { case $1 in B2) echo B4 ;; B4) echo B6 ;; B6) echo B7 ;; esac; }
applied() { [ -f "$LOG" ] && cut -f1 "$LOG" | grep -qx "$1"; }
log() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$STEP" "$1" "$2" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG" ||
    die "写不进移动记录 $LOG；已停下。"
}

# Stop unless no robotd and no project Python program runs, and no other step holds the lock.
guard() {
  local pattern status
  for pattern in '-x robotd' '-f microduck_toy[.]|measure_board_resources'; do
    # shellcheck disable=SC2086  # two words: the pgrep option and its pattern
    pgrep ${pattern%% *} "${pattern#* }" >/dev/null 2>&1
    status=$?
    [ $status -eq 0 ] && die '板上 robotd 或实验程序正在运行；先停掉再整理。'
    [ $status -eq 1 ] || die "pgrep 出错（退出码 $status），无法确认 robotd 没在运行；已停下。"
  done
  exec 9>/tmp/microduck_reorg.lock || die '建不了锁文件 /tmp/microduck_reorg.lock。'
  flock -n 9 || die '另一个整理步骤还在板上运行（可能是刚才按 Ctrl+C 的那一步）；等它结束再来。'
}

# Print every change with its check; return 1 if any check fails. Paths made earlier in the
# same step count as present.
preflight() {
  local line op src dst parent bad=0 note wheels
  declare -A made=()
  for line in "${OPS[@]}"; do
    IFS=$TAB read -r op src dst <<<"$line"
    note=ok
    if [ "$op" = mkdir ]; then
      if [ -d "$dst" ]; then note='已存在，跳过'; elif there "$dst"; then note='错误：存在但不是目录'; fi
      made[$dst]=1
    elif [ "$op" = pycache ]; then
      there "$src" || [ -n "${made[$src]:-}" ] || note='错误：目录不存在'
    else
      parent=$(dirname "$dst")
      if ! there "$src" && [ -z "${made[$src]:-}" ]; then note='错误：来源不存在'
      elif there "$dst"; then note='错误：目标已存在'
      elif [ ! -d "$parent" ] && [ -z "${made[$parent]:-}" ]; then note='错误：上级目录不存在'
      fi
      if [ "$op" = venv ]; then
        command -v python3.11 >/dev/null || note='错误：没有 python3.11'
        for wheels in "$R/downloads/lite2-2.3.2-cp311/wheelhouse" "$R/downloads/vision_cpu_wheels/wheelhouse"; do
          [ -d "$wheels" ] || note="错误：缺少 $wheels"
        done
      fi
      made[$dst]=1
    fi
    [ "$op" = pycache ] && printf '%-7s %s 里的 __pycache__  [%s]\n' "$op" "$src" "$note" \
                        || printf '%-7s %s -> %s  [%s]\n' "$op" "$src" "$dst" "$note"
    case $note in 错误*) bad=1 ;; esac
  done
  return $bad
}

run_ops() {
  local line op src dst
  for line in "${OPS[@]}"; do
    IFS=$TAB read -r op src dst <<<"$line"
    case $op in
      mkdir)   # logged after: the log itself lives in the first folder made
        [ -d "$dst" ] && continue
        mkdir -- "$dst" || die "建目录失败：$dst"
        log mkdir - "$dst" ;;
      mv)
        log mv "$src" "$dst"
        mv -n -T -- "$src" "$dst" && there "$dst" && ! there "$src" || die "移动失败：$src" ;;
      cp)
        log cp "$src" "$dst"
        cp -a -T -- "$src" "$dst" || die "复制失败：$src"
        if [ -d "$src" ]; then diff -rq -- "$src" "$dst" >/dev/null; else cmp -s -- "$src" "$dst"; fi ||
          die "复制后内容不一致：$dst" ;;
      sum)
        log sum "$src" "$dst"
        (cd "$(dirname "$src")" && sha256sum -- "$(basename "$src")") > "$dst" || die "校验和失败：$src" ;;
      venv)
        log venv "$src" "$dst"
        python3.11 -m venv "$dst" || die "建 venv 失败：$dst"
        "$dst/bin/python" -m pip install --no-index --require-hashes \
          --find-links "$R/downloads/lite2-2.3.2-cp311/wheelhouse" \
          --find-links "$R/downloads/vision_cpu_wheels/wheelhouse" -r "$src" || die 'pip 安装失败'
        "$dst/bin/python" -m pip check || die 'pip check 报告依赖问题'
        "$dst/bin/python" -c 'import importlib.metadata as m, cv2, numpy, onnxruntime, rknnlite.api
print("rknn-toolkit-lite2", m.version("rknn-toolkit-lite2"), "onnxruntime", onnxruntime.__version__,
      "numpy", numpy.__version__, "opencv", cv2.__version__)' || die '导入检查失败' ;;
      pycache)
        log pycache "$src" -
        find "$src" -type d -name __pycache__ -prune -exec rm -rf -- {} + || die "清理缓存失败：$src" ;;
    esac
  done
}

undo_step() {
  local line op src dst
  mapfile -t lines < <(awk -F'\t' -v s="$STEP" '$1 == s' "$LOG" | tac)
  for line in "${lines[@]}"; do
    IFS=$TAB read -r _ op src dst _ <<<"$line"
    case $op in
      mv)
        if there "$src" && ! there "$dst"; then echo "已在原位，跳过：$src"; continue; fi
        there "$src" && die "无法撤销：$src 已经有东西"
        there "$dst" || die "无法撤销：$dst 不见了"
        mv -n -T -- "$dst" "$src" && there "$src" && ! there "$dst" || die "移回失败：$dst"
        echo "移回 $dst -> $src" ;;
      cp|sum|venv)
        case $dst in "$R"/*) ;; *) die "拒绝删除 ~/microduck 以外的路径：$dst" ;; esac
        there "$dst" || continue
        rm -rf -- "$dst" && ! there "$dst" || die "删除副本失败：$dst；记录保留，修好后再运行 undo $STEP。"
        echo "删除副本 $dst" ;;
      mkdir)
        if [ "$dst" = "$R" ]; then echo "保留 $R（里面有移动记录）"
        elif [ ! -d "$dst" ]; then continue
        elif rmdir -- "$dst" 2>/dev/null; then echo "删除空目录 $dst"
        else echo "保留非空目录 $dst"; fi ;;
      pycache)
        echo "$src 里的 __pycache__ 是缓存，删掉的不恢复" ;;
    esac
  done
  awk -F'\t' -v s="$STEP" '$1 == s' "$LOG" >> "$UNDONE"
  awk -F'\t' -v s="$STEP" '$1 != s' "$LOG" > "$LOG.tmp" && mv -- "$LOG.tmp" "$LOG"
  [ -s "$LOG" ] || rm -f -- "$LOG"
}

# --- read-only reports ---

precheck() {
  local r=$H/microduck-g3-offline/microduck/target/release/robotd
  echo '== 进程'; pgrep -a robotd; pgrep -af python3
  echo '== ~/microduck'; ls -ld "$R" 2>&1
  echo '== microduck-board 大小'; du -sh "$H"/microduck-board/* 2>&1 | sort -h
  echo '== 启动文件'
  grep -n -e rust-1.89 -e cargo -e microduck -e onnxruntime -e npu-venv -e ORT_DYLIB -e DUCK_RUNTIME \
    "$H/.bashrc" "$H/.profile" 2>&1
  echo '== ~/.cargo'; ls -la "$H/.cargo" 2>&1; grep -rn linaro "$H/.cargo" 2>&1
  echo '== 定时任务和服务'; crontab -l 2>&1; systemctl --user list-unit-files --state=enabled 2>&1 | head -20
  echo '== robotd'; ls -la "$r" 2>&1; sha256sum "$r" 2>&1
  readelf -d "$r" 2>&1 | grep -i -e rpath -e runpath -e needed -e 'not found'
  echo '== cargo vendor 配置'; cat "$H/microduck-g3-offline/microduck/.cargo/config.toml" 2>&1
  echo '== Rust'; "$H/rust-1.89.0/bin/rustc" -vV 2>&1; "$H/rust-1.89.0/bin/cargo" -V 2>&1; command -v cargo rustc
  echo '== venv'; cat "$H/rk3588-npu-venv/pyvenv.cfg" 2>&1
  "$H/rk3588-npu-venv/bin/python" -m pip --version 2>&1; "$H/rk3588-npu-venv/bin/python" -m pip freeze 2>&1
  echo '== 系统 Python'; python3 --version 2>&1; command -v python3.11; ls -la /usr/bin/python3.11 2>&1
  echo '== 运行文件校验'
  sha256sum "$H/$ORT/lib/libonnxruntime.so.1.28.0" "$H"/microduck-board/policies/*.onnx \
    "$H/microduck-board/robotd.toml" 2>&1
  echo '== run/robotd'; ls -la "$H/microduck-board/run/robotd" 2>&1
  echo '== 工具'; command -v rsync tar readelf diff cmp tac pgrep flock
  echo '== 磁盘'; df -h "$H"
}

inventory() {
  echo '== ls'; ls -la "$H"
  echo '== du'; du -sh "$H"/* "$H"/.[!.]* 2>/dev/null | sort -h
  if [ -d "$R" ]; then
    echo '== microduck'; du -sh "$R"/* "$R"/*/* 2>/dev/null
    echo '== find microduck'
    find "$R" -maxdepth 3 -not -path "$R/runtime/venv/*" -not -path "$R/build/*" | sort
  fi
  echo '== df'; df -h "$H"
}

status() {
  local s n
  for s in B2 B4 B6 B7; do
    n=0; [ -f "$LOG" ] && n=$(cut -f1 "$LOG" | grep -cx "$s")
    if [ "$n" -gt 0 ]; then echo "$s 已执行（$n 项记录）"; else echo "$s 未执行"; fi
  done
  for s in "$R/runtime/bin/robotd" "$R/runtime/$ORT" "$R/runtime/venv/bin/python" \
           "$H/microduck-board" "$H/rk3588-npu-venv"; do
    if there "$s"; then echo "有  $s"; else echo "无  $s"; fi
  done
}

backup_items() {
  local p
  for p in microduck-board/run microduck-board/robotd.toml microduck-board/SHA256SUMS \
           microduck-board/sources.json board-baseline-20261006-qG5aeM.txt yolov5-rk3588-smoke \
           microduck-g3-offline/tofd; do
    there "$H/$p" && printf '%s\n' "$p"
  done
  for p in "$H"/microduck-g3-offline/*; do
    [ -f "$p" ] && printf '%s\n' "${p#"$H"/}"
  done
}

cmd=${1:-}
STEP=${2:-}
case $cmd in
  precheck|inventory|status)
    [ $# -eq 1 ] || die "$USAGE"
    $cmd ;;
  backup-tar|backup-sums)
    [ $# -eq 1 ] || die "$USAGE"
    applied B2 && die '已经执行过 B2；备份要在 B2 之前做。'
    guard
    cd "$H" || die "进不去 $H"
    mapfile -t items < <(backup_items)
    [ ${#items[@]} -gt 0 ] || die '没有可备份的文件。'
    if [ "$cmd" = backup-tar ]; then
      find "${items[@]}" \( -type d -o -type f -o -type l \) -print0 | tar --null --no-recursion -T - -cf -
    else
      find "${items[@]}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    fi ;;
  plan|apply|undo)
    [ $# -eq 2 ] || die "$USAGE"
    case $STEP in B2|B4|B6|B7) ;; *) die "$USAGE" ;; esac
    mapfile -t OPS < <("ops_$STEP")
    case $cmd in
      plan)
        applied "$STEP" && echo "注意：$STEP 已经执行过。"
        preflight ;;
      apply)
        applied "$STEP" && die "$STEP 已经执行过；要重来先 undo $STEP。"
        prev=$(previous_step "$STEP")
        [ -z "$prev" ] || applied "$prev" || die "先执行 $prev。"
        if [ "$STEP" = B2 ] && there "$R" && ! there "$LOG" && ! there "$UNDONE"; then
          die "$R 已经存在，但不是这个脚本建的；先看看里面是什么。"
        fi
        guard
        echo "== 预检 $STEP"
        preflight || die '预检没通过，什么都没改。'
        echo "== 执行 $STEP"
        run_ops
        echo "$STEP 完成；移动记录：$LOG" ;;
      undo)
        applied "$STEP" || die "$STEP 没有执行记录，不需要撤销。"
        next=$(next_step "$STEP")
        [ -z "$next" ] || ! applied "$next" || die "先撤销 $next。"
        guard
        undo_step
        echo "$STEP 已撤销；撤销过的记录在 $UNDONE" ;;
    esac ;;
  *)
    die "$USAGE" ;;
esac
