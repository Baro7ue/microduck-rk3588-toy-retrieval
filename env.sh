# Project environment: `source env.sh` from zsh or bash, from any directory.
# Every path follows this file's location, so the folder can be moved or renamed.
# Variables already set in the shell win over the defaults below.
if [ -n "${ZSH_VERSION:-}" ]; then
  eval 'MD_ROOT=$(cd "$(dirname "${(%):-%x}")" && pwd -P)'
else
  MD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
fi
export MD_ROOT

# Repository layout.
export MD_RUNS="${MD_RUNS:-$MD_ROOT/runs}"
export MD_MODELS="$MD_ROOT/models"
export MD_RUNTIME="$MD_ROOT/third_party/microduck"      # pollen-robotics/microduck (Rust runtime)
export MD_RL="$MD_ROOT/third_party/microduck_rl"        # pollen-robotics/microduck_rl (MuJoCo + training)
export MD_RUNTIME_COMMIT=43d09674221da03e1a8f8f6f5cf66c21a9978b2c
export MD_RL_COMMIT=cb70b792312d559a4da09064d92009079671815f

# Mac Python: the microduck_rl uv environment (numpy, onnxruntime, mujoco) plus the
# offline vision packages (cv2, ultralytics) appended after it.
export MD_PY="$MD_RL/.venv/bin/python"
export MD_MJPY="$MD_RL/.venv/bin/mjpython"
export MD_VISION_SITE="$MD_ROOT/offline/vision-export/site-packages"

# Mac sim twin and RK3588 board link.
export MD_SIM_HOST="${MD_SIM_HOST:-192.168.50.1}"                         # Mac address the board reaches
export MD_MAC_STATE="${MD_MAC_STATE:-/private/tmp/md-runtime-ethan}"       # Mac-local robotd socket dir
export MD_STARTUP_SOCKET="${MD_STARTUP_SOCKET:-/private/tmp/md-rk-startup-ethan.sock}"  # forwarded board robotd.sock
export MD_BOARD_HOST="${MD_BOARD_HOST:-RK3588}"                            # ssh alias in ~/.ssh/config
export MD_BOARD_ROOT="${MD_BOARD_ROOT:-/home/linaro/microduck-board}"
# Board-side programs, as installed during G3/G4 (paths on the board, not on this Mac).
export MD_BOARD_ROBOTD="${MD_BOARD_ROBOTD:-/home/linaro/microduck-g3-offline/microduck/target/release/robotd}"
export MD_BOARD_ORT="${MD_BOARD_ORT:-/home/linaro/onnxruntime-linux-aarch64-1.28.0/lib/libonnxruntime.so.1.28.0}"
export MD_BOARD_PY="${MD_BOARD_PY:-/home/linaro/rk3588-npu-venv/bin/python}"
export MD_BOARD_SITE="${MD_BOARD_SITE:-/home/linaro/microduck-board/vision-cpu-python}"

case ":${PYTHONPATH:-}:" in
  *":$MD_ROOT/src:"*) ;;
  *) export PYTHONPATH="$MD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
