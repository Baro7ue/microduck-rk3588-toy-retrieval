#!/bin/zsh
# Use the existing pinned environment and record one interactive gait check.
set -euo pipefail

GAIT_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
GAIT_REPO="$GAIT_PROJECT/task/microduck_r1"
GAIT_EVIDENCE="$GAIT_PROJECT/task/evidence/G2"
GAIT_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
GAIT_PREFIX="$GAIT_EVIDENCE/gait_17500_delay1_$GAIT_STAMP"
/bin/mkdir -p "$GAIT_EVIDENCE"
cd "$GAIT_REPO"

print "使用模型：17,500 轮；执行器延迟：20 毫秒。"
print "请在本终端按键；按 Q 正常退出后写出 CSV。"
print "完整日志：$GAIT_PREFIX.log"
print "观测与动作：$GAIT_PREFIX.csv"

PYTHONUNBUFFERED=1 "$GAIT_REPO/.venv/bin/mjpython" \
  "$GAIT_REPO/scripts/infer_policy.py" \
  --walking "$GAIT_PROJECT/task/policies/walk_17500.onnx" \
  --new-cmd-obs \
  --lin-vel-x 0.0 \
  --lin-vel-y 0.0 \
  --ang-vel-z 0.0 \
  --delay 1 \
  --save-csv "$GAIT_PREFIX.csv" \
  2>&1 | /usr/bin/tee "$GAIT_PREFIX.log"
