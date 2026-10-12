#!/bin/zsh
# Run one bounded diagnostic against the Mac simulator; leave pinned sources untouched.
set -euo pipefail
PROBE_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
PROBE_PYTHON="$PROBE_PROJECT/task/microduck_r1/.venv/bin/python"
PROBE_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
PROBE_LOG="$PROBE_PROJECT/task/evidence/G2/runtime_stand_console_$PROBE_STAMP.log"
/bin/mkdir -p "$PROBE_PROJECT/task/evidence/G2"
print "终端日志：$PROBE_LOG"
"$PROBE_PYTHON" -u "$PROBE_PROJECT/work/probe_runtime_stand.py" "$@" 2>&1 | /usr/bin/tee "$PROBE_LOG"
