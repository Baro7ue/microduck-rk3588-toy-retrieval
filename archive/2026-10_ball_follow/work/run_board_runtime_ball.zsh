#!/bin/zsh
# One terminal owns the Mac body and the foreground board SSH session.
[[ -o interactive ]] || exec /bin/zsh -f -i "$0" "$@"
setopt MONITOR TRAPS_ASYNC || exit 1
unsetopt CHECK_JOBS
set -o pipefail
RK_RUNTIME_ROOT=${MD_RUNTIME_ROOT:-"/Users/ethan/Documents/RK3588 Microduck"}
RK_BODY_PID=''
RK_BOARD_PID=''

if (( $# > 1 )) || [[ $# == 1 && $1 != --viewer ]]; then
  print -u2 -r -- "用法：run_board_runtime_ball.zsh [--viewer]"
  exit 2
fi

listener_pids() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t
}

stop_group() {
  local child=$1 tick
  [[ -n $child ]] || return 0
  kill -TERM -- -"$child" 2>/dev/null
  for tick in {1..20}; do
    kill -0 -- -"$child" 2>/dev/null || break
    sleep .05
  done
  kill -KILL -- -"$child" 2>/dev/null
  wait "$child" 2>/dev/null
  return 0
}

cleanup() {
  local result=$?
  trap - EXIT INT TERM HUP
  stop_group "$RK_BOARD_PID"
  stop_group "$RK_BODY_PID"
  return "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

for RK_PORT in 7801 7901; do
  if [[ -n $(listener_pids "$RK_PORT") ]]; then
    print -u2 -r -- "端口 $RK_PORT 仍被占用；先退出旧 B 和旧 A，保留已有日志。"
    exit 1
  fi
done

/bin/zsh "$RK_RUNTIME_ROOT/work/run_board_runtime_body_camera_ball_no_reflection.zsh" "$@" &
RK_BODY_PID=$!
RK_START=$SECONDS
while true; do
  if ! kill -0 "$RK_BODY_PID" 2>/dev/null; then
    wait "$RK_BODY_PID"
    RK_BODY_RESULT=$?
    (( RK_BODY_RESULT == 0 )) && RK_BODY_RESULT=1
    exit "$RK_BODY_RESULT"
  fi
  RK_READY=1
  for RK_PORT in 7801 7901; do
    RK_PIDS=(${=$(listener_pids "$RK_PORT")})
    (( ${#RK_PIDS} == 1 )) || { RK_READY=0; break; }
    RK_GROUP=$(ps -o pgid= -p "$RK_PIDS[1]")
    [[ ${RK_GROUP//[[:space:]]/} == $RK_BODY_PID ]] || { RK_READY=0; break; }
  done
  (( RK_READY )) && break
  if (( SECONDS-RK_START >= 30 )); then
    print -u2 -r -- "仿真身体和相机在 30 秒内未准备好；板端控制尚未启动。"
    exit 1
  fi
  sleep .1
done

print -r -- "身体与相机已就绪；启动板端控制。实验在另一个终端运行。"
/bin/zsh "$RK_RUNTIME_ROOT/work/run_board_runtime_robotd.zsh" &
RK_BOARD_PID=$!
fg '%?run_board_runtime_robotd.zsh'
RK_BOARD_RESULT=$?
if (( RK_BOARD_RESULT == 127 )); then
  wait "$RK_BOARD_PID"  # B may have exited before foregrounding; retain its result.
  RK_BOARD_RESULT=$?
fi
exit "$RK_BOARD_RESULT"
