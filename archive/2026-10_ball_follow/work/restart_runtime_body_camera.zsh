#!/bin/zsh
set -euo pipefail
CAMERA_RESTART_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
CAMERA_RESTART_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
CAMERA_RESTART_LOG="$CAMERA_RESTART_PROJECT/task/evidence/G2/runtime_camera_restart_$CAMERA_RESTART_STAMP.log"
/bin/mkdir -p "$CAMERA_RESTART_PROJECT/task/evidence/G2"
"$CAMERA_RESTART_PROJECT/task/microduck_r1/.venv/bin/python" -u \
    "$CAMERA_RESTART_PROJECT/work/reset_runtime_session.py" 2>&1 | \
    /usr/bin/tee "$CAMERA_RESTART_LOG"
exec /bin/zsh "$CAMERA_RESTART_PROJECT/work/run_runtime_body_camera.zsh" "$@"
