#!/bin/zsh
# Stage 0 sim body: run_sim_body.zsh's body plus probe_body.py's truth log and test port
# (127.0.0.1:7811). No camera by default (0a-0c do not use it).
# Default: robotd on this Mac (body on 127.0.0.1:7801). --board: body on $MD_SIM_HOST:7801
# for robotd on the RK3588, with startup support watching the forwarded socket.
# Log: runs/<UTC>_s0_probe_body/body.log.
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_probe_body.zsh [--board] [--scene 场景.xml] [--camera] [--viewer]'
scene='' host=127.0.0.1 startup=$MD_MAC_STATE/robotd.sock camera=0 viewer=0
while (( $# )); do
  case $1 in
    --board) host=$MD_SIM_HOST; startup=$MD_STARTUP_SOCKET; shift ;;
    --scene) (( $# >= 2 )) || md_die "$usage"; scene=$2; shift 2 ;;
    --camera) camera=1; shift ;;
    --viewer) viewer=1; shift ;;
    *) md_die "$usage" ;;
  esac
done

md_require_commit "$MD_RL" "$MD_RL_COMMIT"
ports=(7801 7811)
(( camera )) && ports+=(7901)
for port in $ports; do
  if md_port_busy $port; then md_die "端口 $port 已被占用；先退出旧的仿真身体。"; fi
done

args=(--probe-port 7811 --ducks 1 --host "$host" --port 7801 --keyframe HOME)
if [[ -n $scene ]]; then
  [[ -f $scene ]] || md_die "找不到场景文件：$scene"
  args=(--scene "${scene:A}" $args)
fi
(( camera )) && args+=(--cameras a --frame-port 7901 --camera-fps 15)
(( viewer )) || args+=(--headless)

run=$(md_run_dir s0_probe_body) || exit 1
print -r -- "仿真身体 $host:7801；真值和测试端口 127.0.0.1:7811；日志 $run/body.log"
print -r -- "保持本终端运行；结束时按 Ctrl+C。"
cd "$MD_RL"
MD_STARTUP_SOCKET=$startup PYTHONUNBUFFERED=1 \
  "$MD_MJPY" "$MD_ROOT/src/microduck_toy/sim/probe_body.py" $args 2>&1 | tee "$run/body.log"
