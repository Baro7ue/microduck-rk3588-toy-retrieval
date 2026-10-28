#!/bin/zsh
# Stage 4F: the report's tables and figures from the batch folders and the 4D run (plan section 4F).
#   run_tables.zsh [--runs DIR] [--out DIR] [--onef RUN_DIR] [--dev]
# Writes results/stage4/ (stage4_facts.json, CSV tables, figures/). Reads only; runs nothing on the robot.
# --dev checks the tool on the Stage 3 dev batches instead and writes results/stage4_dev/.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

cd "$MD_ROOT"
export PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE"   # cv2: tables.py reads episodes through stage3 (as run_batch.zsh)
[[ -x $MD_PY ]] || md_die "找不到项目的 Python：$MD_PY"
exec "$MD_PY" -u "$MD_ROOT/scripts/stage4/tables.py" "$@"
