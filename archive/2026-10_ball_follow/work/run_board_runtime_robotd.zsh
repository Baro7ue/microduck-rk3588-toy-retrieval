#!/bin/zsh
# Mac entry: start board robotd and its Unix-socket forward in one SSH session.
# Requires a fresh G3 body at 192.168.50.1:7801 and the existing board files.
# Saves the board daemon log under /home/linaro/microduck-board/run/.
set -o pipefail

ssh -t -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes \
  -L /private/tmp/md-rk-startup-ethan.sock:/home/linaro/microduck-board/run/robotd.sock \
  RK3588 '
set -o pipefail
cd "$HOME/microduck-board" &&
sha256sum -c SHA256SUMS &&
mkdir -p run &&
export ORT_DYLIB_PATH="$HOME/onnxruntime-linux-aarch64-1.28.0/lib/libonnxruntime.so.1.28.0" &&
export DUCK_RUNTIME_DIR="$HOME/microduck-board/run" &&
"$HOME/microduck-g3-offline/microduck/target/release/robotd" \
  --sim 192.168.50.1:7801 \
  --params "$HOME/microduck-board/robotd.toml" \
  --socket "$DUCK_RUNTIME_DIR/robotd.sock" \
  2>&1 | tee "$DUCK_RUNTIME_DIR/robotd-$(date +%Y%m%d-%H%M%S).log"
'
