#!/bin/zsh
# Keep the existing training checkout unchanged; enable the external heading loop.
set -euo pipefail
STRAIGHT_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
STRAIGHT_REPO="$STRAIGHT_PROJECT/task/microduck_r1"
STRAIGHT_EVIDENCE="$STRAIGHT_PROJECT/task/evidence/G2"
STRAIGHT_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
STRAIGHT_PREFIX="$STRAIGHT_EVIDENCE/straight_hold_$STRAIGHT_STAMP"
/bin/mkdir -p "$STRAIGHT_EVIDENCE"
cd "$STRAIGHT_REPO"

print '航向保持已启用；先等5秒，再在本终端按 ↑ 前进。'
print 'A/E：手动转向；空格：停止；空格后按 ↑：锁定新方向。Q：退出。'
print "完整日志：$STRAIGHT_PREFIX.log"
print "观测与动作：$STRAIGHT_PREFIX.csv"
PYTHONUNBUFFERED=1 "$STRAIGHT_REPO/.venv/bin/mjpython" \
  "$STRAIGHT_PROJECT/work/replay_heading_hold.py" \
  --heading-kp 1.0 --heading-ki 0.05 \
  --walking "$STRAIGHT_PROJECT/task/policies/walk_17500.onnx" \
  --new-cmd-obs --lin-vel-x 0.1 --delay 1 \
  --save-csv "$STRAIGHT_PREFIX.csv" \
  2>&1 | /usr/bin/tee "$STRAIGHT_PREFIX.log"
