#!/bin/zsh
# Mac entry: zero the head command deltas, validate standing, then reuse board camera detection.
set -o pipefail
RK_G4_ROOT="/Users/ethan/Documents/RK3588 Microduck"
RK_G4_BODY_LOG=$(ls -t "$RK_G4_ROOT/task/evidence/G4"/mac_body_camera_ball_*.log | head -1)
RK_G4_STAMP=$(date +%Y%m%d_%H%M%S)

MD_ROBOT_SOCKET=/private/tmp/md-rk-startup-ethan.sock MD_EVIDENCE_GATE=G4 \
PYTHONPATH="$RK_G4_ROOT/work" \
  "$RK_G4_ROOT/task/microduck_r1/.venv/bin/python" -u - "$RK_G4_BODY_LOG" <<'PY' \
  2>&1 | tee "$RK_G4_ROOT/task/evidence/G4/stand_camera_console_$RK_G4_STAMP.log"
import json
from pathlib import Path
import sys
import probe_runtime_stand as stand

head = {"neck_pitch": 0.0, "head_pitch": 0.0,
        "head_yaw": 0.0, "head_roll": 0.0}
reply = stand.rpc("robot.head", head)
print("头部命令增量归零：" + json.dumps({"head": head, "reply": reply}), flush=True)
if not reply.get("accepted"):
    raise RuntimeError("头部姿态请求未被接受")
sys.exit(stand.main(keep_enabled=True, support_log=Path(sys.argv[1])))
PY
RK_G4_STAND_STATUS=$?
(( RK_G4_STAND_STATUS == 0 )) || exit "$RK_G4_STAND_STATUS"
/bin/zsh "$RK_G4_ROOT/work/probe_board_yolo11_camera.zsh"
