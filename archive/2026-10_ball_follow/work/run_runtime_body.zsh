#!/bin/zsh
# Start the pinned simulator body; policy control is a separate next step.
set -euo pipefail

BODY_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
BODY_REPO="$BODY_PROJECT/task/microduck_r1"
BODY_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
BODY_LOG="$BODY_PROJECT/task/evidence/G2/runtime_body_$BODY_STAMP.log"

cd "$BODY_REPO"
if [[ "$(/usr/bin/git rev-parse HEAD)" != 'cb70b792312d559a4da09064d92009079671815f' ]]; then
    print -u2 '仿真仓库提交与计划不一致，已停止启动。'
    exit 1
fi
if /usr/sbin/lsof -nP -iTCP:7801 -sTCP:LISTEN >/dev/null 2>&1; then
    print -u2 '7801 端口已被占用，请保留现场并把输出发回来。'
    /usr/sbin/lsof -nP -iTCP:7801 -sTCP:LISTEN
    exit 1
fi
/bin/mkdir -p "$BODY_PROJECT/task/evidence/G2"
print '启动 MuJoCo 身体，监听本机 TCP 7801，等待 robotd 连接。'
print '鸭子先保持 HOME 初始姿态；这一步尚未启用策略控制。'
print "完整日志：$BODY_LOG"
print '保持窗口与终端运行；结束时可在本终端按 Ctrl+C。'

PYTHONUNBUFFERED=1 "$BODY_REPO/.venv/bin/mjpython" \
    -m mjlab_microduck.sim.body_server \
    --ducks 1 --host 127.0.0.1 --port 7801 --keyframe HOME \
    2>&1 | /usr/bin/tee "$BODY_LOG"
