#!/bin/zsh
# Run every unit test with the Mac project Python (microduck_rl .venv + offline vision packages).
# Extra arguments go to unittest, e.g. `-v` or `-k camera`.
set -uo pipefail
source "${0:A:h}/../../env.sh"

cd "$MD_ROOT"
PYTHONPATH="$PYTHONPATH:$MD_VISION_SITE" "$MD_PY" -m unittest discover -s tests "$@"
