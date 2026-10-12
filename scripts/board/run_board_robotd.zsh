#!/bin/zsh
# Start robotd on the RK3588 against the Mac sim body and forward its control socket to this Mac
# ($MD_STARTUP_SOCKET). Needs the sim body on $MD_SIM_HOST:7801 (scripts/sim/run_sim_body.zsh).
# The board checks its SHA256SUMS first; the robotd log stays in $MD_BOARD_ROOT/run/ on the board.
# Ctrl+C stops robotd and the socket forward together.
set -uo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

remote="set -o pipefail
cd ${(q)MD_BOARD_ROOT} &&
sha256sum -c SHA256SUMS &&
mkdir -p run &&
export ORT_DYLIB_PATH=${(q)MD_BOARD_ORT} DUCK_RUNTIME_DIR=${(q)MD_BOARD_ROOT}/run &&
${(q)MD_BOARD_ROBOTD} --sim ${(q)MD_SIM_HOST}:7801 --params ${(q)MD_BOARD_ROOT}/robotd.toml \\
  --socket \"\$DUCK_RUNTIME_DIR/robotd.sock\" \\
  2>&1 | tee \"\$DUCK_RUNTIME_DIR/robotd-\$(date +%Y%m%d-%H%M%S).log\""
ssh -t -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes \
  -L "$MD_STARTUP_SOCKET:$MD_BOARD_ROOT/run/robotd.sock" "$MD_BOARD_HOST" "$remote"
