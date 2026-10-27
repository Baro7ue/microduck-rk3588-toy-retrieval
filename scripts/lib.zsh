# Helpers shared by the zsh launchers; source after env.sh.

md_die() {
  print -u2 -r -- "$*"
  exit 1
}

# md_run_dir NAME: create and print a new runs/<UTC YYYYMMDDTHHMMSSZ>_NAME (never reuses one).
md_run_dir() {
  local dir try
  for try in 1 2 3; do
    dir="$MD_RUNS/$(TZ=UTC date '+%Y%m%dT%H%M%SZ')_$1"
    if mkdir -p "$MD_RUNS" && mkdir "$dir" 2>/dev/null; then
      print -r -- "$dir"
      return 0
    fi
    sleep 1
  done
  print -u2 -r -- "无法新建运行目录：$dir"
  return 1
}

# md_require_commit REPO COMMIT: stop unless the upstream checkout is at the pinned commit.
md_require_commit() {
  local head
  head=$(git -C "$1" rev-parse HEAD) || md_die "无法读取 $1 的提交。"
  [[ $head == "$2" ]] || md_die "$1 当前提交 $head，约定提交 $2；已停止。"
}

# md_port_busy PORT: true when something on this Mac listens on TCP PORT.
md_port_busy() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

# md_bundle DIR: copy the microduck_toy package (*.py only) into DIR.
md_bundle() {
  local file rel
  for file in "$MD_ROOT"/src/microduck_toy/**/*.py(N); do
    rel=${file#$MD_ROOT/src/}
    mkdir -p "$1/${rel:h}" && cp "$file" "$1/$rel" || return 1
  done
}

# md_sha256sums DIR: write DIR/SHA256SUMS over every file in DIR (checked on the board with sha256sum -c).
md_sha256sums() {
  (
    cd "$1" || exit 1
    local files=(**/*(.N))
    files=(${files:#SHA256SUMS})
    shasum -a 256 -- ${(o)files} > SHA256SUMS
  )
}

# The Stage 3 brain's bundle: the package, the four configs the brain reads and the frozen Stage 2 models.
MD_STAGE3_FILES=(configs/control/stage1.json configs/control/stage1_facts.json configs/control/stage3.json
                 configs/vision/stage2.json models/yolo/yolo11n_toys/yolo11n_toys_split_320.onnx
                 models/yolo/yolo11n_toys/yolo11n_toys_fp16.rknn models/yolo/yolo11n_toys/yolo11n_toys_int8.rknn)

# md_stage3_check: stop unless every file of the Stage 3 bundle is in the repo.
md_stage3_check() {
  local f
  for f in $MD_STAGE3_FILES; do
    [[ -f $MD_ROOT/$f ]] || md_die "缺 $f；先把仓库更新到阶段 3 的提交（PR #2 冻结的模型也要在）。"
  done
}

# md_stage3_bundle DIR: the package and MD_STAGE3_FILES in DIR, with DIR/SHA256SUMS.
md_stage3_bundle() {
  local f
  md_bundle "$1" || return 1
  for f in $MD_STAGE3_FILES; do
    mkdir -p "$1/${f:h}" && cp "$MD_ROOT/$f" "$1/$f" || return 1
  done
  md_sha256sums "$1"
}
