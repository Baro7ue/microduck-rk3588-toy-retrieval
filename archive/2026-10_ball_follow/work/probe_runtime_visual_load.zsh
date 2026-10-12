#!/bin/zsh
# Read-only observer; use the existing training environment without changing pinned sources.
set -euo pipefail
VISUAL_PROJECT='/Users/ethan/Documents/RK3588 Microduck'
VISUAL_PYTHON="$VISUAL_PROJECT/task/microduck_r1/.venv/bin/python"
VISUAL_STAMP=$(/bin/date '+%Y%m%d_%H%M%S')
/bin/mkdir -p "$VISUAL_PROJECT/task/evidence/G2"
VISUAL_LOG=$(/usr/bin/mktemp "$VISUAL_PROJECT/task/evidence/G2/visual_load_console_${VISUAL_STAMP}_XXXXXX")
print "终端日志：$VISUAL_LOG"
"$VISUAL_PYTHON" -u "$VISUAL_PROJECT/work/probe_runtime_visual_load.py" "$@" 2>&1 | /usr/bin/tee "$VISUAL_LOG"
