#!/bin/zsh
# Build robotd (control loop), robotctl (CLI) and tofd (ToF service) for the Mac sim twin.
# Output: third_party/microduck/target/debug/; full log in runs/<UTC>_build_mac_runtime/build.log.
set -euo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

md_require_commit "$MD_RUNTIME" "$MD_RUNTIME_COMMIT"
[[ -f $HOME/.cargo/env ]] && source "$HOME/.cargo/env"
run=$(md_run_dir build_mac_runtime) || exit 1
print -r -- "编译 robotd、robotctl、tofd；完整日志：$run/build.log"
cd "$MD_RUNTIME"
{
  rustc --version
  cargo --version
  git rev-parse HEAD
  # --locked keeps the committed Cargo.lock dependency versions.
  cargo build --locked -p robotd -p robotctl -p tof
} 2>&1 | tee "$run/build.log"
print -r -- "编译成功；程序位于 $MD_RUNTIME/target/debug/。"
