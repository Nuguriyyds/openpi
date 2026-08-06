#!/usr/bin/env bash
set -euo pipefail

# Start only the ROS2 observation stack required by the breakfast progress
# client. This script does not start a policy server, robot control service, or
# inference client.

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PIPER_SETUP="${PIPER_SETUP:-/home/geekplus/agilex/piper_ros/install/setup.bash}"
SESSION="${SESSION:-paper_rtc_ros2}"

ROS_DOMAIN_ID="${BREAKFAST_ROS_DOMAIN_ID:-21}"
ROS_LOCALHOST_ONLY="${BREAKFAST_ROS_LOCALHOST_ONLY:-0}"

LEFT_CAMERA_SERIAL="${LEFT_CAMERA_SERIAL:-260322272339}"
RIGHT_CAMERA_SERIAL="${RIGHT_CAMERA_SERIAL:-260322275836}"
TOP_CAMERA_SERIAL="${TOP_CAMERA_SERIAL:-049222071894}"
CAMERA_PROFILE="${CAMERA_PROFILE:-640x480x30}"

CAN_LEFT="${CAN_LEFT:-can0}"
CAN_RIGHT="${CAN_RIGHT:-can1}"

TERM_TIMEOUT_SECONDS="${TERM_TIMEOUT_SECONDS:-8}"

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 1; }
}

require_file "$ROS_SETUP"
require_file "$PIPER_SETUP"
require_command tmux

proc_starttime() {
  awk '{print $22}' "/proc/$1/stat" 2>/dev/null
}

proc_is_alive() {
  local pid="$1" expected_starttime="$2" current_starttime state
  current_starttime="$(proc_starttime "$pid")" || return 1
  [[ "$current_starttime" == "$expected_starttime" ]] || return 1
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)" || return 1
  [[ "$state" != "Z" ]]
}

proc_cmdline() {
  tr '\0\t\n' '   ' <"/proc/$1/cmdline" 2>/dev/null || true
}

resolve_process_arg() {
  local pid="$1" arg="$2" cwd candidate
  [[ "$arg" != *$'\n'* && "$arg" != *$'\t'* && "$arg" != *' '* ]] || return 1
  if [[ "$arg" == /* ]]; then
    candidate="$arg"
  else
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" || return 1
    candidate="$cwd/$arg"
  fi
  readlink -f -- "$candidate" 2>/dev/null
}

is_hardware_process() {
  local pid="$1" arg base resolved index
  local -a argv=()
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  while IFS= read -r -d '' arg; do argv+=("$arg"); done <"/proc/$pid/cmdline" || true
  ((${#argv[@]})) || return 1

  for arg in "${argv[@]}"; do
    base="${arg##*/}"
    case "$base" in
      realsense2_camera_node)
        resolved="$(resolve_process_arg "$pid" "$arg" 2>/dev/null || true)"
        [[ "$resolved" == /opt/ros/humble/lib/realsense2_camera/realsense2_camera_node ]] && return 0
        ;;
      piper_single_ctrl)
        resolved="$(resolve_process_arg "$pid" "$arg" 2>/dev/null || true)"
        [[ "$resolved" == /home/geekplus/agilex/piper_ros/install/piper/lib/piper/piper_single_ctrl ]] && return 0
        ;;
    esac
  done

  for ((index = 0; index < ${#argv[@]}; index++)); do
    [[ "${argv[index]##*/}" == ros2 ]] || continue
    if [[ "${argv[index+1]:-}" == launch && "${argv[index+2]:-}" == realsense2_camera && "${argv[index+3]:-}" == rs_launch.py ]]; then
      return 0
    fi
    if [[ "${argv[index+1]:-}" == run && "${argv[index+2]:-}" == realsense2_camera && "${argv[index+3]:-}" == realsense2_camera_node ]]; then
      return 0
    fi
    if [[ "${argv[index+1]:-}" == launch && "${argv[index+2]:-}" == piper && "${argv[index+3]:-}" == start_two_piper.launch.py ]]; then
      return 0
    fi
  done
  return 1
}

is_control_process() {
  local pid="$1" arg base
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  while IFS= read -r -d '' arg; do
    base="${arg##*/}"
    case "$base" in
      online_inference_execution.py|rtc_online_inference_execution.py|paper_rtc_online_inference_execution.py|training_paper_rtc_online_inference_execution.py|ros_only_training_paper_rtc_online_inference_execution.py|start_robot_arm_service.py|start_robot_arm_service_chw.py|ros_only_robot_control_service.py|uni_training_rtc_hardware_service.py)
        return 0
        ;;
    esac
  done <"/proc/$pid/cmdline" || true
  return 1
}

list_process_records() {
  local matcher="$1" proc pid starttime cmdline
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ "$pid" != "$$" ]] || continue
    "$matcher" "$pid" || continue
    starttime="$(proc_starttime "$pid")" || continue
    cmdline="$(proc_cmdline "$pid")"
    printf '%s\t%s\t%s\n' "$pid" "$starttime" "$cmdline"
  done
}

terminate_records() {
  local -a records=("$@")
  local record pid starttime cmdline deadline any_alive
  ((${#records[@]})) || return 0

  for record in "${records[@]}"; do
    IFS=$'\t' read -r pid starttime cmdline <<<"$record"
    if proc_is_alive "$pid" "$starttime"; then
      echo "[breakfast-progress] stopping hardware pid=$pid: $cmdline"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  deadline=$((SECONDS + TERM_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    any_alive=0
    for record in "${records[@]}"; do
      IFS=$'\t' read -r pid starttime cmdline <<<"$record"
      proc_is_alive "$pid" "$starttime" && any_alive=1
    done
    ((any_alive == 0)) && return 0
    sleep 0.25
  done

  for record in "${records[@]}"; do
    IFS=$'\t' read -r pid starttime cmdline <<<"$record"
    if proc_is_alive "$pid" "$starttime"; then
      echo "[breakfast-progress] hardware pid=$pid did not stop; sending KILL"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

ensure_no_active_control() {
  local -a records=()
  mapfile -t records < <(list_process_records is_control_process)
  if ((${#records[@]})); then
    echo "[ERROR] An inference client or robot control service is still running." >&2
    echo "[ERROR] Refusing to interrupt another operator. Active process(es):" >&2
    printf '  %s\n' "${records[@]}" >&2
    return 1
  fi

  if ss -H -ltn 2>/dev/null | grep -Eq ':(9900|9901)[[:space:]]'; then
    echo "[ERROR] Robot service port 9900 or 9901 is still listening." >&2
    echo "[ERROR] Refusing to clean hardware until its owner stops the service." >&2
    ss -H -ltnp 2>/dev/null | grep -E ':(9900|9901)[[:space:]]' >&2 || true
    return 1
  fi
}

cleanup_hardware_stack() {
  local round
  local -a records=()

  ensure_no_active_control || exit 1

  tmux kill-session -t "$SESSION" 2>/dev/null || true

  # Launch parents may leave children briefly while handling TERM. Rescan until
  # all approved RealSense/Piper hardware processes are gone.
  for round in 1 2 3 4; do
    mapfile -t records < <(list_process_records is_hardware_process)
    ((${#records[@]})) || break
    echo "[breakfast-progress] hardware cleanup round $round: ${#records[@]} process(es)"
    terminate_records "${records[@]}"
  done

  mapfile -t records < <(list_process_records is_hardware_process)
  if ((${#records[@]})); then
    echo "[ERROR] Hardware processes remain after cleanup:" >&2
    printf '  %s\n' "${records[@]}" >&2
    exit 1
  fi
}

for can_device in "$CAN_LEFT" "$CAN_RIGHT"; do
  ip link show "$can_device" >/dev/null 2>&1 || {
    echo "[ERROR] CAN interface is missing: $can_device" >&2
    exit 1
  }
  ip link show "$can_device" | grep -q '<[^>]*UP' || {
    echo "[ERROR] CAN interface is not UP: $can_device" >&2
    exit 1
  }
done

ROS_ENV="set +u; source '$ROS_SETUP'; source '$PIPER_SETUP'; set -u; export ROS_DOMAIN_ID='$ROS_DOMAIN_ID'; export ROS_LOCALHOST_ONLY='$ROS_LOCALHOST_ONLY'"

echo "[breakfast-progress] checking for active inference/control tasks"
cleanup_hardware_stack

echo "[breakfast-progress] starting clean ROS2 hardware session: $SESSION"

tmux new-session -d -s "$SESSION" -n cam-left \
  "bash -lc \"$ROS_ENV; exec ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera/left camera_name:=camera serial_no:=\\\"'$LEFT_CAMERA_SERIAL'\\\" enable_color:=true enable_depth:=true depth_module.color_profile:='$CAMERA_PROFILE' depth_module.depth_profile:='$CAMERA_PROFILE' enable_gyro:=false enable_accel:=false\""

tmux new-window -t "$SESSION" -n cam-right \
  "bash -lc \"$ROS_ENV; exec ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera/right camera_name:=camera serial_no:=\\\"'$RIGHT_CAMERA_SERIAL'\\\" enable_color:=true enable_depth:=true depth_module.color_profile:='$CAMERA_PROFILE' depth_module.depth_profile:='$CAMERA_PROFILE' enable_gyro:=false enable_accel:=false\""

tmux new-window -t "$SESSION" -n cam-top \
  "bash -lc \"$ROS_ENV; exec ros2 run realsense2_camera realsense2_camera_node --ros-args -r __ns:=/camera/top -r __node:=camera -p serial_no:=\\\"'$TOP_CAMERA_SERIAL'\\\" -p enable_color:=true -p enable_depth:=false -p enable_infra1:=false -p enable_infra2:=false -p enable_gyro:=false -p enable_accel:=false -p rgb_camera.color_profile:='$CAMERA_PROFILE'\""

# The installed Piper launch file uses the misspelled launch argument
# "girpper_exist". Preserve it to match the working public-machine setup.
tmux new-window -t "$SESSION" -n piper-arms \
  "bash -lc \"$ROS_ENV; exec ros2 launch piper start_two_piper.launch.py log_level:=warn can_left_port:='$CAN_LEFT' can_right_port:='$CAN_RIGHT' auto_enable:=true girpper_exist:=true\""

echo "[breakfast-progress] ROS2 hardware nodes started"
echo "  session: $SESSION"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "  left/right/top cameras: $LEFT_CAMERA_SERIAL / $RIGHT_CAMERA_SERIAL / $TOP_CAMERA_SERIAL"
echo "  inspect: tmux attach -t $SESSION"
echo "  stop:    tmux kill-session -t $SESSION"
echo ""
echo "Before running the client in another shell:"
echo "  export ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
