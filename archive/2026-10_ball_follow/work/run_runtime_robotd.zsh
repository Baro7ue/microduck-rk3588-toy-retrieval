#!/bin/zsh
# Connect the real daemon to the simulator; motion needs a separate enable request.
set -euo pipefail

ROBOTD_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
ROBOTD_REPO="$ROBOTD_PROJECT/task/microduck"
ROBOTD_STATE='/private/tmp/md-runtime-ethan'
ROBOTD_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
ROBOTD_LOG="$ROBOTD_PROJECT/task/evidence/G2/runtime_robotd_$ROBOTD_STAMP.log"
cd "$ROBOTD_REPO"
if [[ "$(/usr/bin/git rev-parse HEAD)" != '43d09674221da03e1a8f8f6f5cf66c21a9978b2c' ]]; then
    print -u2 '运行时提交与计划不一致，已停止启动。'
    exit 1
fi
if ! /usr/sbin/lsof -nP -iTCP:7801 -sTCP:LISTEN >/dev/null 2>&1; then
    print -u2 '未检测到 7801 端口的仿真身体，请先保留身体窗口运行。'
    exit 1
fi
/bin/mkdir -p "$ROBOTD_STATE" "$ROBOTD_PROJECT/task/evidence/G2"
export DUCK_RUNTIME_DIR="$ROBOTD_STATE"
export ORT_DYLIB_PATH="$ROBOTD_PROJECT/task/microduck_r1/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.24.4.dylib"

print '启动 robotd，加载本地模型，并连接 MuJoCo 身体。'
print '本步保持控制关闭；下一步通过 health 检查加载与通信。'
print "完整日志：$ROBOTD_LOG"
print "控制接口：$ROBOTD_STATE/robotd.sock"
"$ROBOTD_REPO/target/debug/robotd" \
    --sim 127.0.0.1:7801 \
    --params "$ROBOTD_PROJECT/work/runtime_robotd.toml" \
    --socket "$ROBOTD_STATE/robotd.sock" \
    2>&1 | /usr/bin/tee "$ROBOTD_LOG"
