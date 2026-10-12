#!/bin/zsh
# 用途：先确认完整相机帧，再并行观察75秒和行走60秒；75秒含前后站立。
# 前提：body/headless已运行，robotd已启用且稳定零速站立。
# 输入：可选 --socket 原样转发，默认 /private/tmp/md-runtime-ethan/robotd.sock。
# 设备：身体127.0.0.1:7801、相机127.0.0.1:7901、上面的robotd Unix socket。
# 进程：启动已有只读观测和行走探针；正常不启停body/robotd或自动enable。
# 异常：行走子探针会执行原有stop/disable策略清理；未确认须在robotd终端Ctrl+C。
# 输出/日志：/Users/ethan/Documents/RK3588 Microduck/task/evidence/G2/walk_camera_<微秒时间戳>/
# 父summary.json关联两个子summary，并保存observe/walk各自stdout/stderr日志。
set -euo pipefail
WALK_CAMERA_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
exec "$WALK_CAMERA_PROJECT/task/microduck_r1/.venv/bin/python" -u \
    "$WALK_CAMERA_PROJECT/work/probe_runtime_walk_camera.py" "$@"
