#!/bin/zsh
# One camera on the established external startup helper; pinned sources stay unchanged.
set -euo pipefail
CAMERA_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
CAMERA_REPO="$CAMERA_PROJECT/task/microduck_r1"
CAMERA_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
CAMERA_LOG="$CAMERA_PROJECT/task/evidence/G2/runtime_body_camera_$CAMERA_STAMP.log"
cd "$CAMERA_REPO"
if [[ "$(/usr/bin/git rev-parse HEAD)" != 'cb70b792312d559a4da09064d92009079671815f' ]]; then
    print -u2 '训练仓库提交与计划不一致，已停止启动。'
    exit 1
fi
for CAMERA_PORT in 7801 7901; do
    if /usr/sbin/lsof -nP -iTCP:$CAMERA_PORT -sTCP:LISTEN >/dev/null 2>&1; then
        print -u2 "$CAMERA_PORT 端口已被占用；保留输出，不重复启动。"
        exit 1
    fi
done
/bin/mkdir -p "$CAMERA_PROJECT/task/evidence/G2"
print '启动一只鸭子的身体和头部相机：身体 7801，相机 7901。'
print '相机输出 640×360 UYVY，发送速率设为 15 fps；实际渲染和到达速率需另测。'
print '起步辅助只在策略接管前保持 HOME；不计入站立成绩。'
print "完整日志：$CAMERA_LOG"
PYTHONUNBUFFERED=1 "$CAMERA_REPO/.venv/bin/mjpython" \
    "$CAMERA_PROJECT/work/runtime_bootstrap_body.py" \
    --ducks 1 --host 127.0.0.1 --port 7801 --keyframe HOME \
    --cameras a --frame-port 7901 --camera-fps 15 "$@" \
    2>&1 | /usr/bin/tee "$CAMERA_LOG"
