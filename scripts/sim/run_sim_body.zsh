#!/bin/zsh
# Start the MuJoCo sim body (TCP 7801) and, unless --no-camera, the head camera (TCP 7901, 15 fps).
# Default host is MD_SIM_HOST so the board can connect; --local binds 127.0.0.1 for a Mac robotd.
# Log: runs/<UTC>_sim_body/body.log; its path is also written to runs/current_sim_body_log.
set -euo pipefail
source "${0:A:h}/../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_sim_body.zsh [--scene 场景.xml] [--local] [--no-camera] [--viewer]'
scene='' host=$MD_SIM_HOST startup=$MD_STARTUP_SOCKET camera=1 viewer=0
while (( $# )); do
  case $1 in
    --scene) (( $# >= 2 )) || md_die "$usage"; scene=$2; shift 2 ;;
    --local) host=127.0.0.1; startup=$MD_MAC_STATE/robotd.sock; shift ;;
    --no-camera) camera=0; shift ;;
    --viewer) viewer=1; shift ;;
    *) md_die "$usage" ;;
  esac
done

md_require_commit "$MD_RL" "$MD_RL_COMMIT"
ports=(7801)
(( camera )) && ports+=(7901)
for port in $ports; do
  if md_port_busy $port; then md_die "端口 $port 已被占用；先退出旧的仿真身体。"; fi
done

args=(--ducks 1 --host "$host" --port 7801 --keyframe HOME)
if [[ -n $scene ]]; then
  [[ -f $scene ]] || md_die "找不到场景文件：$scene"
  args=(--scene "${scene:A}" $args)
fi
(( camera )) && args+=(--cameras a --frame-port 7901 --camera-fps 15)
(( viewer )) || args+=(--headless)

run=$(md_run_dir sim_body) || exit 1
print -r -- "$run/body.log" > "$MD_RUNS/current_sim_body_log"
camera_label=关闭
(( camera )) && camera_label=$host:7901
print -r -- "仿真身体 $host:7801；相机 $camera_label；日志 $run/body.log"
print -r -- "保持本终端运行；结束时按 Ctrl+C。"
cd "$MD_RL"
MD_STARTUP_SOCKET=$startup PYTHONUNBUFFERED=1 \
  "$MD_MJPY" "$MD_ROOT/src/microduck_toy/sim/bootstrap_body.py" $args 2>&1 | tee "$run/body.log"
