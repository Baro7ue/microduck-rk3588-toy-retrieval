#!/bin/zsh
set -euo pipefail
WALK_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
WALK_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
WALK_LOG="$WALK_PROJECT/task/evidence/G2/runtime_walk_console_$WALK_STAMP.log"
/bin/mkdir -p "$WALK_PROJECT/task/evidence/G2"
print "终端日志：$WALK_LOG"
"$WALK_PROJECT/task/microduck_r1/.venv/bin/python" -u \
    "$WALK_PROJECT/work/probe_runtime_walk.py" "$@" 2>&1 | /usr/bin/tee "$WALK_LOG"
