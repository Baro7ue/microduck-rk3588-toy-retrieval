#!/bin/zsh
# Zero the head command offsets, check a 10 s stand on the board robotd (through the forwarded
# socket), then run the board YOLO camera probe. Arguments are passed on to probe_board_yolo_camera.zsh.
# Needs: run_sim_body.zsh (with camera) and run_board_robotd.zsh already running.
# Output: runs/<UTC>_board_stand/ (stand evidence + console.log), then the YOLO run directory.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

[[ -f $MD_RUNS/current_sim_body_log ]] ||
  md_die "找不到 runs/current_sim_body_log；先用 scripts/sim/run_sim_body.zsh 启动仿真身体。"
body_log=$(<"$MD_RUNS/current_sim_body_log")
[[ -f $body_log ]] || md_die "仿真身体日志不存在：$body_log"
run=$(md_run_dir board_stand) || exit 1

MD_ROBOT_SOCKET=$MD_STARTUP_SOCKET MD_RUNS=$run "$MD_PY" -u - "$body_log" <<'PY' \
  2>&1 | tee "$run/console.log"
import json
from pathlib import Path
import sys

from microduck_toy.rpc import stand_probe as stand

head = {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
reply = stand.rpc("robot.head", head)
print("头部命令增量归零：" + json.dumps({"head": head, "reply": reply}), flush=True)
if not reply.get("accepted"):
    raise RuntimeError("头部姿态请求未被接受")
sys.exit(stand.main(keep_enabled=True, support_log=Path(sys.argv[1])))
PY
stand_status=$?
(( stand_status == 0 )) || exit $stand_status
zsh "$MD_ROOT/scripts/board/probe_board_yolo_camera.zsh" "$@"
