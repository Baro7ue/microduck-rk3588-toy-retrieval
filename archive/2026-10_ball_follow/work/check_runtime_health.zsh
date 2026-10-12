#!/bin/zsh
# Read daemon health and save the complete report; this does not enable motion.
set -euo pipefail

HEALTH_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
HEALTH_ROBOTCTL="$HEALTH_PROJECT/task/microduck/target/debug/robotctl"
HEALTH_SOCKET='/private/tmp/md-runtime-ethan/robotd.sock'
HEALTH_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
HEALTH_LOG="$HEALTH_PROJECT/task/evidence/G2/runtime_health_$HEALTH_STAMP.log"

if [[ ! -x "$HEALTH_ROBOTCTL" ]]; then
    print -u2 '未找到已编译的 robotctl，请保留完整输出。'
    exit 1
fi
if [[ ! -S "$HEALTH_SOCKET" ]]; then
    print -u2 '未找到 robotd 控制接口，请确认身体和 robotd 的终端仍在运行。'
    exit 1
fi
/bin/mkdir -p "$HEALTH_PROJECT/task/evidence/G2"
print '只读取控制环的健康状态，不启用运动。'
print "完整报告：$HEALTH_LOG"
"$HEALTH_ROBOTCTL" --robot-socket "$HEALTH_SOCKET" health 2>&1 | /usr/bin/tee "$HEALTH_LOG"
