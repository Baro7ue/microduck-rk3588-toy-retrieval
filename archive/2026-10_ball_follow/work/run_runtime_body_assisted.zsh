#!/bin/zsh
set -euo pipefail
BODY_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
BODY_REPO="$BODY_PROJECT/task/microduck_r1"
BODY_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
BODY_LOG="$BODY_PROJECT/task/evidence/G2/runtime_body_assisted_$BODY_STAMP.log"
cd "$BODY_REPO"
if [[ "$(/usr/bin/git rev-parse HEAD)" != 'cb70b792312d559a4da09064d92009079671815f' ]]; then
    print -u2 '训练仓库提交与计划不一致，已停止启动。'
    exit 1
fi
if /usr/sbin/lsof -nP -iTCP:7801 -sTCP:LISTEN >/dev/null 2>&1; then
    print -u2 '7801 端口已有身体进程，请先停止旧仿真。'
    exit 1
fi
/bin/mkdir -p "$BODY_PROJECT/task/evidence/G2"
print '启动新的 MuJoCo 身体；起步时临时保持 HOME，策略首次输出后解除全部固定。'
print '起步辅助期间不计入站立成绩。此窗口等待 robotd 连接与启用。'
print "完整日志：$BODY_LOG"
PYTHONUNBUFFERED=1 "$BODY_REPO/.venv/bin/mjpython" \
    "$BODY_PROJECT/work/runtime_bootstrap_body.py" \
    --ducks 1 --host 127.0.0.1 --port 7801 --keyframe HOME \
    2>&1 | /usr/bin/tee "$BODY_LOG"
