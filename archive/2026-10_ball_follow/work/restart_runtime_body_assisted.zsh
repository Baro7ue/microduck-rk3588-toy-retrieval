#!/bin/zsh
set -euo pipefail
RESTART_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
RESTART_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
RESTART_LOG="$RESTART_PROJECT/task/evidence/G2/runtime_restart_$RESTART_STAMP.log"
/bin/mkdir -p "$RESTART_PROJECT/task/evidence/G2"
"$RESTART_PROJECT/task/microduck_r1/.venv/bin/python" -u \
    "$RESTART_PROJECT/work/reset_runtime_session.py" 2>&1 | /usr/bin/tee "$RESTART_LOG"
exec /bin/zsh "$RESTART_PROJECT/work/run_runtime_body_assisted.zsh"
