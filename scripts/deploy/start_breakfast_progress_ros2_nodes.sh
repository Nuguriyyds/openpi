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

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 1; }
}

require_file "$ROS_SETUP"
require_file "$PIPER_SETUP"
require_command tmux

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

echo "[breakfast-progress] restarting ROS2 hardware session: $SESSION"
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n cam-left \
  "bash -lc \"$ROS_ENV; ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera/left camera_name:=camera serial_no:=\\\"'$LEFT_CAMERA_SERIAL'\\\" enable_color:=true enable_depth:=true depth_module.color_profile:='$CAMERA_PROFILE' depth_module.depth_profile:='$CAMERA_PROFILE' enable_gyro:=false enable_accel:=false\""

tmux new-window -t "$SESSION" -n cam-right \
  "bash -lc \"$ROS_ENV; ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera/right camera_name:=camera serial_no:=\\\"'$RIGHT_CAMERA_SERIAL'\\\" enable_color:=true enable_depth:=true depth_module.color_profile:='$CAMERA_PROFILE' depth_module.depth_profile:='$CAMERA_PROFILE' enable_gyro:=false enable_accel:=false\""

tmux new-window -t "$SESSION" -n cam-top \
  "bash -lc \"$ROS_ENV; ros2 run realsense2_camera realsense2_camera_node --ros-args -r __ns:=/camera/top -r __node:=camera -p serial_no:=\\\"'$TOP_CAMERA_SERIAL'\\\" -p enable_color:=true -p enable_depth:=false -p enable_infra1:=false -p enable_infra2:=false -p enable_gyro:=false -p enable_accel:=false -p rgb_camera.color_profile:='$CAMERA_PROFILE'\""

# The installed Piper launch file uses the misspelled launch argument
# "girpper_exist". Preserve it to match the working public-machine setup.
tmux new-window -t "$SESSION" -n piper-arms \
  "bash -lc \"$ROS_ENV; ros2 launch piper start_two_piper.launch.py log_level:=warn can_left_port:='$CAN_LEFT' can_right_port:='$CAN_RIGHT' auto_enable:=true girpper_exist:=true\""

echo "[breakfast-progress] ROS2 hardware nodes started"
echo "  session: $SESSION"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "  left/right/top cameras: $LEFT_CAMERA_SERIAL / $RIGHT_CAMERA_SERIAL / $TOP_CAMERA_SERIAL"
echo "  inspect: tmux attach -t $SESSION"
echo "  stop:    tmux kill-session -t $SESSION"
echo ""
echo "Before running the client in another shell:"
echo "  export ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
