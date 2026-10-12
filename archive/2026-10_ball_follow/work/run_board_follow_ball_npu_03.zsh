#!/bin/zsh
# Current A starts at disabled HOME: request lookdown, stand once, then NPU close approach.
set -o pipefail
RK_FOLLOW_NPU_ROOT="/Users/ethan/Documents/RK3588 Microduck"
RK_FOLLOW_NPU_BODY_LOG=$(ls -t "$RK_FOLLOW_NPU_ROOT/task/evidence/G4"/mac_body_camera_ball_*.log | head -1)
RK_FOLLOW_NPU_STAMP=$(date -u +%Y%m%d_%H%M%S)

MD_ROBOT_SOCKET=/private/tmp/md-rk-startup-ethan.sock MD_EVIDENCE_GATE=G4 \
PYTHONPATH="$RK_FOLLOW_NPU_ROOT/work" \
  "$RK_FOLLOW_NPU_ROOT/task/microduck_r1/.venv/bin/python" -u - "$RK_FOLLOW_NPU_BODY_LOG" <<'PY' \
  2>&1 | tee "$RK_FOLLOW_NPU_ROOT/task/evidence/G4/stand_npu_follow_console_$RK_FOLLOW_NPU_STAMP.log"
import json
from pathlib import Path
import sys
import probe_runtime_stand as stand

head = {"neck_pitch": 0.0, "head_pitch": 0.4, "head_yaw": 0.0, "head_roll": 0.0}
reply = stand.rpc("robot.head", head)
print("头部俯视请求（策略命令，弧度）：" + json.dumps({"head": head, "reply": reply}), flush=True)
if not reply.get("accepted"):
    raise RuntimeError("头部姿态请求未被接受")
sys.exit(stand.main(keep_enabled=True, support_log=Path(sys.argv[1])))
PY
RK_FOLLOW_NPU_STAND_STATUS=$?
(( RK_FOLLOW_NPU_STAND_STATUS == 0 )) || exit "$RK_FOLLOW_NPU_STAND_STATUS"
/bin/zsh "$RK_FOLLOW_NPU_ROOT/work/run_board_follow_ball_npu.zsh"
