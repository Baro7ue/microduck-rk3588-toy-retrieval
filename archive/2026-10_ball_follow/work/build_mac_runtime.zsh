#!/bin/zsh
# Compile the pinned runtime and retain the complete output for troubleshooting.
set -euo pipefail

RUNTIME_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
RUNTIME_REPO="$RUNTIME_PROJECT/task/microduck"
RUNTIME_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
RUNTIME_LOG="$RUNTIME_PROJECT/task/evidence/G2/runtime_build_$RUNTIME_STAMP.log"

# Make rustc and cargo available to this terminal and their child processes.
source '/Users/ethan/.cargo/env'
cd "$RUNTIME_REPO"
if [[ "$(/usr/bin/git rev-parse HEAD)" != '43d09674221da03e1a8f8f6f5cf66c21a9978b2c' ]]; then
    print -u2 '仓库提交与计划不一致，已停止编译。'
    exit 1
fi
/bin/mkdir -p "$RUNTIME_PROJECT/task/evidence/G2"

print '编译 robotd（控制环）、tofd（测距服务）、robotctl（控制工具）。'
print "完整日志：$RUNTIME_LOG"
{
    rustc --version
    cargo --version
    /usr/bin/git rev-parse HEAD
    # --locked uses the committed dependency versions without changing Cargo.lock.
    cargo build --locked -p robotd -p robotctl -p tof
} 2>&1 | /usr/bin/tee "$RUNTIME_LOG"
print "编译成功；程序位于 $RUNTIME_REPO/target/debug/。"
