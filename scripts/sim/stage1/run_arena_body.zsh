#!/bin/zsh
# Stage 1 sim body: the Stage 0 probe body plus the 2 x 2 m field, the five toys, the tray, the
# beak weld and the oracle boxes (src/microduck_toy/sim/arena_body.py). Truth log and test port
# on 127.0.0.1:7811 as in Stage 0.
#   run_arena_body.zsh [--floor grid|wood] [--camera [--oracle-fps N]] [--board] [--viewer]
#                      [--render-inline] [--room]
# --camera   head camera on 127.0.0.1:7901 plus the oracle boxes (needed by 1D and the 1A
#            image checks); N renders per second (default 15, upstream's rate; 0 = only when a
#            test asks for a frame). Frames are drawn on their own thread, so drawing no longer
#            holds up the physics (Stage 3 Mac batch 1: inline drawing stalled 17% of the 20 ms
#            steps and robotd's contact odometry then under-counted 16-35%).
# --render-inline  draw on the physics thread as before (debugging only; the physics stalls).
# --board    body on $MD_SIM_HOST:7801 for robotd on the RK3588 (as Stage 0's --board).
# --room     Stage 5 room demo (configs/demo/room.json): 4.4 x 4.4 m room with furniture, 3 x 3 m
#            field, 12 toys (copies of the five); light-wood floor unless --floor says otherwise.
# Log: runs/<UTC>_s1_arena_body/body.log.
set -euo pipefail
source "${0:A:h}/../../../env.sh"
source "$MD_ROOT/scripts/lib.zsh"

usage='用法：run_arena_body.zsh [--floor grid|wood] [--camera [--oracle-fps N]] [--board] [--viewer] [--render-inline] [--room]'
floor= host=127.0.0.1 startup=$MD_MAC_STATE/robotd.sock camera=0 fps=15 viewer=0 inline=0 room=0
while (( $# )); do
  case $1 in
    --floor) (( $# >= 2 )) || md_die "$usage"; floor=$2; shift 2 ;;
    --camera) camera=1; shift ;;
    --oracle-fps) (( $# >= 2 )) || md_die "$usage"; fps=$2; shift 2 ;;
    --board) host=$MD_SIM_HOST; startup=$MD_STARTUP_SOCKET; shift ;;
    --viewer) viewer=1; shift ;;
    --render-inline) inline=1; shift ;;
    --room) room=1; shift ;;
    *) md_die "$usage" ;;
  esac
done
[[ -n $floor ]] || { (( room )) && floor=wood || floor=grid; }
[[ $floor == grid || $floor == wood ]] || md_die "$usage"

md_require_commit "$MD_RL" "$MD_RL_COMMIT"
ports=(7801 7811)
(( camera )) && ports+=(7901)
for port in $ports; do
  if md_port_busy $port; then md_die "端口 $port 已被占用；先退出旧的仿真身体。"; fi
done

args=(--floor $floor --stage1-config "$MD_ROOT/configs/control/stage1.json"
      --probe-port 7811 --ducks 1 --host "$host" --port 7801 --keyframe HOME)
(( camera )) && args+=(--cameras a --frame-port 7901 --oracle-fps $fps)
(( inline )) && args+=(--render-inline)
(( room )) && args+=(--room "$MD_ROOT/configs/demo/room.json")
(( viewer )) || args+=(--headless)

run=$(md_run_dir s1_arena_body) || exit 1
(( room )) && print -r -- "阶段 5 房间演示场景（configs/demo/room.json：房间、家具、12 个玩具）"
print -r -- "阶段 1 场地（地面 $floor）：仿真身体 $host:7801；真值和测试端口 127.0.0.1:7811；日志 $run/body.log"
(( camera )) && print -r -- "头部相机 127.0.0.1:7901，每秒 $fps 帧（0 = 测试要帧时才渲染）"
print -r -- "保持本终端运行；结束时按 Ctrl+C。"
cd "$MD_RL"
MD_STARTUP_SOCKET=$startup PYTHONUNBUFFERED=1 \
  "$MD_MJPY" "$MD_ROOT/src/microduck_toy/sim/arena_body.py" $args 2>&1 | tee "$run/body.log"
