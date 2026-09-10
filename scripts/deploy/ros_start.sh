#!/usr/bin/env bash
set -Eeuo pipefail

# ROS-only Training Paper RTC supervisor.
#
# Safety invariants:
#   * Only piper_ros may open can0/can1 while this stack is running.
#   * Legacy cleanup matches exact, approved argv signatures. It never uses a
#     broad pkill/killall pattern and never targets generic Python/training jobs.
#     Root-owned approved leftovers are killed through sudo after the same PID
#     starttime guard passes.
#   * Every signal is guarded by the PID's /proc starttime to prevent PID-reuse
#     mistakes. New components run in dedicated setsid sessions and are recorded.
#   * Startup exact-cleans the approved old full chain, including stale policy
#     and inference clients.  This supervisor starts and owns infrastructure
#     only: cameras, piper_ros and the ROS-only adapter.
#   * Shutdown is ordered: ROS-only service, then piper_ros, then cameras.

umask 077

readonly STACK_NAME="ros_only_training_paper_rtc"
readonly STATE_VERSION="2"
readonly EXPECTED_ROS_DOMAIN_ID="21"
readonly LEGACY_TMUX_SESSION="ros_data_collect"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PIPER_SETUP="${PIPER_SETUP:-/home/geekplus/agilex/piper_ros/install/setup.bash}"
PIPER_SDK_DIR="${PIPER_SDK_DIR:-/home/geekplus/agilex/piper_sdk/piper_sdk}"
ROS_SERVICE_PYTHON="${ROS_SERVICE_PYTHON:-/home/geekplus/miniforge3/envs/piper_ros/bin/python}"

# Port 8001 is part of full-chain cleanup/pre-start conflict detection, but is
# never opened or managed by this infrastructure supervisor.
POLICY_PORT="${POLICY_PORT:-8001}"

ROS_SERVICE_SCRIPT="${OPENPI_DIR}/scripts/deploy/ros_only_robot_control_service.py"
ROS_SERVICE_HOST="${ROS_SERVICE_HOST:-127.0.0.1}"
ROS_SERVICE_PORT="${ROS_SERVICE_PORT:-9901}"
SUDO_PASSWORD_FILE="${SUDO_PASSWORD_FILE:-${HOME}/.config/ttrtc/sudo.pass}"

CAN_LEFT="${CAN_LEFT:-can0}"
CAN_RIGHT="${CAN_RIGHT:-can1}"
CAN_LEFT_USB="${CAN_LEFT_USB:-1-4:1.0}"
CAN_RIGHT_USB="${CAN_RIGHT_USB:-1-9:1.0}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"

LEFT_CAMERA_SERIAL="${LEFT_CAMERA_SERIAL:-260322272339}"
RIGHT_CAMERA_SERIAL="${RIGHT_CAMERA_SERIAL:-260322275836}"
TOP_CAMERA_SERIAL="${TOP_CAMERA_SERIAL:-049222071894}"
CAMERA_PROFILE="${CAMERA_PROFILE:-640x480x30}"

ROS_START_TIMEOUT="${ROS_START_TIMEOUT:-90}"
SERVICE_START_TIMEOUT="${SERVICE_START_TIMEOUT:-120}"
TERM_TIMEOUT="${TERM_TIMEOUT:-12}"
STOP_SUPERVISOR_TIMEOUT="${STOP_SUPERVISOR_TIMEOUT:-180}"

STATE_FILE="${RTC_STATE_FILE:-/tmp/${STACK_NAME}-${UID}.state}"
LOCK_DIR="${RTC_LOCK_DIR:-/tmp/${STACK_NAME}-${UID}.lock}"
LOG_BASE="${RTC_LOG_BASE:-${OPENPI_DIR}/logs/${STACK_NAME}}"

MODE="start"
LOCK_HELD=0
CLEANUP_ARMED=0
CLEANUP_RUNNING=0
LOG_DIR=""

declare -a PROC_NAMES=()
declare -a PROC_PIDS=()
declare -a PROC_STARTS=()
declare -a PROC_PGIDS=()
declare -a PROC_SIDS=()
declare -a PROC_LOGS=()

STATE_SUPERVISOR_PID=""
STATE_SUPERVISOR_START=""

PENDING_NAME=""
PENDING_PID=""
PENDING_START=""
PENDING_SID=""
PENDING_LOG=""
SPAWN_CRITICAL=0
DEFERRED_EXIT_CODE=0

timestamp() { date '+%F %T'; }
log() { printf '[%s] [INFO] %s\n' "$(timestamp)" "$*"; }
warn() { printf '[%s] [WARN] %s\n' "$(timestamp)" "$*" >&2; }
error() { printf '[%s] [ERROR] %s\n' "$(timestamp)" "$*" >&2; }
die() { error "$*"; exit 1; }

handle_signal() {
  local signal_name="$1" exit_code
  case "$signal_name" in
    INT) exit_code=130 ;;
    HUP) exit_code=129 ;;
    TERM) exit_code=143 ;;
    *) exit_code=1 ;;
  esac
  if ((SPAWN_CRITICAL)); then
    DEFERRED_EXIT_CODE="$exit_code"
    warn "${signal_name} received during process registration; deferring cleanup until PID/starttime are recorded"
    return 0
  fi
  exit "$exit_code"
}

usage() {
  cat <<EOF
Usage:
  $0 [start]
  $0 preflight
  $0 status
  $0 validate
  $0 stop
  $0 force-clean

Modes:
  start       Exact-clean the approved old full RTC/ROS stack, activate CAN,
              then start cameras, piper_ros and the ROS-only service only.
  preflight   Read-only checks. Reports conflicts and exits nonzero on any.
  status      Read-only managed-process, port, CAN and ROS summary.
  validate    Verify the currently managed stack, ROS data and no SDK writer.
  stop        Stop only process sessions recorded by this supervisor state.
  force-clean Stop recorded state plus all known ROS/RTC/OpenPI experiment
              processes and tmux sessions that can conflict with this run.

Default ROS_DOMAIN_ID is forcibly set to 21 for this stack. Logs are written below:
  ${LOG_BASE}
EOF
}

parse_args() {
  local mode_seen=0
  while (($#)); do
    case "$1" in
      start|preflight|status|validate|stop|force-clean)
        ((mode_seen == 0)) || die "only one mode may be selected"
        MODE="$1"
        mode_seen=1
        ;;
      -h|--help|help) usage; exit 0 ;;
      *) die "unknown argument: $1" ;;
    esac
    shift
  done
}

require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
require_exec() { [[ -x "$1" ]] || die "missing executable: $1"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

load_ros_env() {
  set +u
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  # shellcheck disable=SC1090
  source "$PIPER_SETUP"
  set -u
  export ROS_DOMAIN_ID="$EXPECTED_ROS_DOMAIN_ID"
  export ROS_LOCALHOST_ONLY=0
  export PYTHONUNBUFFERED=1
}

proc_stat_tail() {
  local pid="$1" stat
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  IFS= read -r stat <"/proc/${pid}/stat" 2>/dev/null || return 1
  [[ "$stat" == *") "* ]] || return 1
  printf '%s\n' "${stat##*) }"
}

proc_starttime() {
  local tail
  tail="$(proc_stat_tail "$1")" || return 1
  # Fields after comm begin at field 3; starttime (field 22) is item 20.
  set -- $tail
  (($# >= 20)) || return 1
  printf '%s\n' "${20}"
}

proc_state() {
  local tail
  tail="$(proc_stat_tail "$1")" || return 1
  set -- $tail
  printf '%s\n' "$1"
}

proc_alive_same() {
  local pid="$1" expected_start="$2" current state
  current="$(proc_starttime "$pid" 2>/dev/null)" || return 1
  [[ "$current" == "$expected_start" ]] || return 1
  state="$(proc_state "$pid" 2>/dev/null)" || return 1
  [[ "$state" != "Z" ]]
}

proc_pgid() { ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'; }
proc_sid() { ps -o sid= -p "$1" 2>/dev/null | tr -d '[:space:]'; }

proc_cmdline() {
  local pid="$1"
  [[ -r "/proc/${pid}/cmdline" ]] || return 0
  tr '\0\t\n' '   ' <"/proc/${pid}/cmdline" 2>/dev/null || true
}

resolve_process_arg() {
  local pid="$1" arg="$2" cwd candidate
  [[ "$arg" != *$'\n'* && "$arg" != *$'\t'* && "$arg" != *' '* ]] || return 1
  if [[ "$arg" == /* ]]; then
    candidate="$arg"
  else
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null)" || return 1
    candidate="${cwd}/${arg}"
  fi
  readlink -f -- "$candidate" 2>/dev/null
}

is_legacy_process() {
  local pid="$1" arg base resolved entry_resolved="" i entry_index=-1 ros_index=-1 is_policy=0 has_target_port=0
  local has_left_ns=0 has_right_ns=0 has_top_ns=0 has_left_serial=0 has_right_serial=0 has_top_serial=0
  local has_can_left=0 has_can_right=0 has_left_node=0 has_right_node=0
  local -a argv=()
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  while IFS= read -r -d '' arg; do argv+=("$arg"); done <"/proc/${pid}/cmdline" || true
  ((${#argv[@]})) || return 1

  # Only inspect the executable entrypoint position.  A diagnostic process
  # that merely carries one of these paths as a later data argument must never
  # be selected for cleanup.
  base="${argv[0]##*/}"
  case "$base" in
    python|python[0-9]|python[0-9].*|python3)
      for ((i = 1; i < ${#argv[@]}; i++)); do
        case "${argv[i]}" in
          -u|-B|-O|-OO|--isolated|--no-site|--) continue ;;
          -*) entry_index=-1; break ;;
          *) entry_index=$i; break ;;
        esac
      done
      ;;
    bash|sh)
      for ((i = 1; i < ${#argv[@]}; i++)); do
        case "${argv[i]}" in
          -e|-E|-u|-x|-Eeuo|pipefail|--) continue ;;
          -*) entry_index=-1; break ;;
          *) entry_index=$i; break ;;
        esac
      done
      ;;
    uv)
      [[ "${argv[1]:-}" == "run" ]] && entry_index=2
      ;;
    *) entry_index=0 ;;
  esac

  if ((entry_index >= 0 && entry_index < ${#argv[@]})); then
    resolved="$(resolve_process_arg "$pid" "${argv[entry_index]}" 2>/dev/null || true)"
    entry_resolved="$resolved"
    [[ "${entry_resolved##*/}" == "ros2" ]] && ros_index="$entry_index"
    case "$resolved" in
      # RobotArmService has historically been started from several OpenPI
      # clones.  Match only the two exact service entrypoints below a user's
      # OpenPI tree; this catches those clones without broad-matching Python.
      /home/geekplus/openpi/scripts/deploy/start_robot_arm_service_chw.py|/home/geekplus/openpi/scripts/deploy/start_robot_arm_service.py|/home/geekplus/*/openpi/scripts/deploy/start_robot_arm_service_chw.py|/home/geekplus/*/openpi/scripts/deploy/start_robot_arm_service.py)
        return 0
        ;;
      /home/geekplus/new_start.sh|/home/geekplus/agilex/collect_data/start_ros_nodes.sh|/home/geekplus/waic/openpi/start.sh)
        return 0
        ;;
      /home/geekplus/waic/openpi/scripts/deploy/start_robot_arm_service_chw.py|/home/geekplus/waic/openpi/scripts/deploy/start_robot_arm_service.py|/home/geekplus/waic/openpi/scripts/deploy/online_inference_execution.py|/home/geekplus/waic/openpi/scripts/deploy/rtc_online_inference_execution.py|/home/geekplus/waic/openpi/scripts/deploy/training_time_rtc_online_inference_execution.py|/home/geekplus/waic/openpi/scripts/deploy/paper_rtc_online_inference_execution.py|/home/geekplus/waic/openpi/scripts/deploy/training_paper_rtc_online_inference_execution.py|"${OPENPI_DIR}"/scripts/deploy/start_robot_arm_service_chw.py|"${OPENPI_DIR}"/scripts/deploy/start_robot_arm_service.py|"${OPENPI_DIR}"/scripts/deploy/online_inference_execution.py|"${OPENPI_DIR}"/scripts/deploy/rtc_online_inference_execution.py|"${OPENPI_DIR}"/scripts/deploy/training_time_rtc_online_inference_execution.py|"${OPENPI_DIR}"/scripts/deploy/paper_rtc_online_inference_execution.py|"${OPENPI_DIR}"/scripts/deploy/training_paper_rtc_online_inference_execution.py|"${OPENPI_DIR}"/scripts/deploy/ros_only_training_paper_rtc_online_inference_execution.py|"${OPENPI_DIR}"/scripts/deploy/ros_only_robot_control_service.py)
        return 0
        ;;
      /home/geekplus/waic/openpi/scripts/deploy/start_agilex_training_paper_rtc_policy_server.sh|/home/geekplus/waic/openpi/scripts/serve_training_paper_rtc_policy.py|/home/geekplus/waic/openpi/scripts/serve_paper_rtc_policy.py|/home/geekplus/waic/openpi/scripts/serve_rtc_policy.py|"${OPENPI_DIR}"/scripts/deploy/start_agilex_training_paper_rtc_policy_server.sh|"${OPENPI_DIR}"/scripts/serve_training_paper_rtc_policy.py|"${OPENPI_DIR}"/scripts/serve_paper_rtc_policy.py|"${OPENPI_DIR}"/scripts/serve_rtc_policy.py)
        is_policy=1
        ;;
    esac
  fi

  for ((i = 0; i < ${#argv[@]}; i++)); do
    [[ "${argv[i]}" == "--port" && "${argv[i+1]:-}" == "$POLICY_PORT" ]] && has_target_port=1
    case "${argv[i]}" in
      camera_namespace:=camera/left|-r__ns:=/camera/left|__ns:=/camera/left) has_left_ns=1 ;;
      camera_namespace:=camera/right|-r__ns:=/camera/right|__ns:=/camera/right) has_right_ns=1 ;;
      __ns:=/camera/top) has_top_ns=1 ;;
      *"${LEFT_CAMERA_SERIAL}"*) has_left_serial=1 ;;
      *"${RIGHT_CAMERA_SERIAL}"*) has_right_serial=1 ;;
      *"${TOP_CAMERA_SERIAL}"*) has_top_serial=1 ;;
      "can_left_port:=${CAN_LEFT}") has_can_left=1 ;;
      "can_right_port:=${CAN_RIGHT}") has_can_right=1 ;;
      __node:=piper_left_ctrl_node) has_left_node=1 ;;
      __node:=piper_right_ctrl_node) has_right_node=1 ;;
    esac
  done
  ((is_policy && has_target_port)) && return 0

  # Exact ROS launch/run entrypoint at argv[0], plus this stack's known
  # namespace/serial/CAN arguments.  Other cameras, arms and CAN buses are not
  # selected.
  [[ "${argv[0]##*/}" == "ros2" ]] && ros_index=0
  if ((ros_index >= 0)); then
    if [[ "${argv[ros_index+1]:-}" == "launch" && "${argv[ros_index+2]:-}" == "realsense2_camera" && "${argv[ros_index+3]:-}" == "rs_launch.py" ]]; then
      ((has_left_ns && has_left_serial)) && return 0
      ((has_right_ns && has_right_serial)) && return 0
    fi
    if [[ "${argv[ros_index+1]:-}" == "run" && "${argv[ros_index+2]:-}" == "realsense2_camera" && "${argv[ros_index+3]:-}" == "realsense2_camera_node" ]]; then
      ((has_top_ns && has_top_serial)) && return 0
    fi
    if [[ "${argv[ros_index+1]:-}" == "launch" && "${argv[ros_index+2]:-}" == "piper" && "${argv[ros_index+3]:-}" == "start_two_piper.launch.py" ]]; then
      ((has_can_left && has_can_right)) && return 0
    fi
  fi

  resolved="$(resolve_process_arg "$pid" "${argv[0]}" 2>/dev/null || true)"
  if [[ "$resolved" == /opt/ros/humble/lib/realsense2_camera/realsense2_camera_node || "$entry_resolved" == /opt/ros/humble/lib/realsense2_camera/realsense2_camera_node ]]; then
    ((has_top_ns && has_top_serial)) && return 0
    ((has_left_ns || has_right_ns)) && return 0
  fi
  if [[ "$resolved" == /home/geekplus/agilex/piper_ros/install/piper/lib/piper/piper_single_ctrl || "$entry_resolved" == /home/geekplus/agilex/piper_ros/install/piper/lib/piper/piper_single_ctrl ]]; then
    ((has_left_node || has_right_node)) && return 0
  fi
  return 1
}

list_legacy_records() {
  local proc pid start cmd
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ "$pid" != "$$" ]] || continue
    is_legacy_process "$pid" || continue
    start="$(proc_starttime "$pid" 2>/dev/null)" || continue
    cmd="$(proc_cmdline "$pid")"
    printf '%s\t%s\t%s\n' "$pid" "$start" "$cmd"
  done
}

sudo_kill_pid() {
  local signal="$1" pid="$2" err_file
  ensure_sudo
  err_file="$(mktemp /tmp/${STACK_NAME}.sudo-kill.XXXXXX)"
  if sudo -n kill -s "$signal" "$pid" 2>"$err_file"; then
    log "sudo ${signal} pid=${pid} succeeded"
    rm -f "$err_file"
    return 0
  fi
  error "sudo ${signal} pid=${pid} failed: $(tr '\n' ' ' <"$err_file")"
  rm -f "$err_file"
  return 1
}

signal_pid_if_same() {
  local signal="$1" pid="$2" expected_start="$3" label="$4" owner_uid err_file
  if proc_alive_same "$pid" "$expected_start"; then
    log "${signal} pid=${pid} starttime=${expected_start} component=${label}"
    err_file="$(mktemp /tmp/${STACK_NAME}.kill.XXXXXX)"
    if kill -s "$signal" "$pid" 2>"$err_file"; then
      rm -f "$err_file"
      return 0
    fi
    warn "plain ${signal} pid=${pid} failed: $(tr '\n' ' ' <"$err_file")"
    rm -f "$err_file"
    owner_uid="$(awk '/^Uid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)"
    if [[ -n "$owner_uid" && "$owner_uid" != "$UID" ]]; then
      sudo_kill_pid "$signal" "$pid" || return 1
      return 0
    fi
    return 1
  fi
}

terminate_pid_records() {
  local timeout_s="$1" first_signal="$2"; shift 2
  local -a records=("$@")
  local record pid start cmd deadline any
  ((${#records[@]})) || return 0

  for record in "${records[@]}"; do
    IFS=$'\t' read -r pid start cmd <<<"$record"
    signal_pid_if_same "$first_signal" "$pid" "$start" "$cmd" || true
  done

  deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    any=0
    for record in "${records[@]}"; do
      IFS=$'\t' read -r pid start cmd <<<"$record"
      proc_alive_same "$pid" "$start" && any=1
    done
    ((any == 0)) && return 0
    sleep 0.25
  done

  for record in "${records[@]}"; do
    IFS=$'\t' read -r pid start cmd <<<"$record"
    signal_pid_if_same KILL "$pid" "$start" "$cmd" || true
  done

  deadline=$((SECONDS + 5))
  while ((SECONDS < deadline)); do
    any=0
    for record in "${records[@]}"; do
      IFS=$'\t' read -r pid start cmd <<<"$record"
      proc_alive_same "$pid" "$start" && any=1
    done
    ((any == 0)) && return 0
    sleep 0.25
  done
  return 1
}

legacy_tmux_exists() {
  command -v tmux >/dev/null 2>&1 || return 1
  tmux has-session -t "=${LEGACY_TMUX_SESSION}" 2>/dev/null
}

cleanup_legacy_stack() {
  local round
  local -a records=()

  if legacy_tmux_exists; then
    log "killing exact legacy tmux session =${LEGACY_TMUX_SESSION}"
    tmux kill-session -t "=${LEGACY_TMUX_SESSION}"
  fi

  # A launch parent can briefly leave a child while reacting to TERM, so rescan.
  for round in 1 2 3 4; do
    mapfile -t records < <(list_legacy_records)
    ((${#records[@]})) || break
    log "legacy cleanup round ${round}: ${#records[@]} exact process(es)"
    terminate_pid_records "$TERM_TIMEOUT" TERM "${records[@]}" || true
  done

  mapfile -t records < <(list_legacy_records)
  if ((${#records[@]})); then
    error "approved legacy processes remain after cleanup:"
    printf '  %s\n' "${records[@]}" >&2
    return 1
  fi
  legacy_tmux_exists && return 1
  return 0
}

is_legacy_infra_process() {
  local pid="$1" cmd
  is_legacy_process "$pid" || return 1
  cmd="$(proc_cmdline "$pid")"
  case "$cmd" in
    *realsense2_camera*|*piper_single_ctrl*|*start_two_piper.launch.py*|*ros_only_robot_control_service.py*|*start_robot_arm_service_chw.py*|*start_robot_arm_service.py*)
      return 0
      ;;
  esac
  return 1
}

list_legacy_infra_records() {
  local proc pid start cmd
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ "$pid" != "$$" ]] || continue
    is_legacy_infra_process "$pid" || continue
    start="$(proc_starttime "$pid" 2>/dev/null)" || continue
    cmd="$(proc_cmdline "$pid")"
    printf '%s\t%s\t%s\n' "$pid" "$start" "$cmd"
  done
}

cleanup_legacy_infra_stack() {
  local round
  local -a records=()

  for round in 1 2 3 4; do
    mapfile -t records < <(list_legacy_infra_records)
    ((${#records[@]})) || break
    log "legacy infrastructure cleanup round ${round}: ${#records[@]} exact process(es)"
    terminate_pid_records "$TERM_TIMEOUT" TERM "${records[@]}" || true
  done

  mapfile -t records < <(list_legacy_infra_records)
  if ((${#records[@]})); then
    error "approved legacy infrastructure processes remain after cleanup:"
    printf '  %s\n' "${records[@]}" >&2
    return 1
  fi
  return 0
}

is_force_clean_process() {
  local pid="$1" cmd
  [[ "$pid" != "$$" ]] || return 1
  cmd="$(proc_cmdline "$pid")"
  [[ -n "$cmd" ]] || return 1
  case "$cmd" in
    *" tmux "*|tmux\ *|*"paper_rtc_ros2"*|*"ros_data_collect"*|*"piper_service"*)
      return 0
      ;;
    *"/ros2 daemon"*|*" ros2 daemon "*)
      return 0
      ;;
    *"/ros2 launch realsense2_camera"*|*"/ros2 run realsense2_camera"*|*"realsense2_camera_node"*|*"realsense2_camera rs_launch.py"*)
      return 0
      ;;
    *"start_two_piper.launch.py"*|*"piper_single_ctrl"*|*"ros2 launch piper"*)
      return 0
      ;;
    *"ros_only_robot_control_service.py"*|*"start_robot_arm_service_chw.py"*|*"start_robot_arm_service.py"*|*"RobotArmService"*)
      return 0
      ;;
    *"online_inference_execution.py"*|*"rtc_online_inference_execution.py"*|*"paper_rtc_online_inference_execution.py"*|*"training_time_rtc_online_inference_execution.py"*|*"training_paper_rtc_online_inference_execution.py"*|*"ros_only_training_paper_rtc_online_inference_execution.py"*)
      return 0
      ;;
    *"scripts/serve_training_paper_rtc_policy.py"*|*"scripts/serve_paper_rtc_policy.py"*|*"scripts/serve_rtc_policy.py"*|*"scripts/serve_policy.py"*|*"start_agilex_training_paper_rtc_policy_server.sh"*|*"start_agilex_paper_rtc_policy_server"*)
      return 0
      ;;
    *"policy:checkpoint"*|*" --policy.dir "*|*" --policy.config "*)
      return 0
      ;;
    *"/scripts/deploy/ros_start.sh start"*)
      return 0
      ;;
  esac
  return 1
}

list_port_owner_records() {
  local port pid start cmd
  for port in "$POLICY_PORT" "$ROS_SERVICE_PORT" 9900; do
    while IFS= read -r pid; do
      [[ "$pid" =~ ^[0-9]+$ && "$pid" != "$$" ]] || continue
      start="$(proc_starttime "$pid" 2>/dev/null)" || continue
      cmd="$(proc_cmdline "$pid")"
      printf '%s\t%s\t%s\n' "$pid" "$start" "$cmd"
    done < <(ss -H -ltnp "sport = :${port}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  done
}

list_force_clean_records() {
  local proc pid start cmd
  {
    for proc in /proc/[0-9]*; do
      pid="${proc##*/}"
      [[ "$pid" != "$$" ]] || continue
      is_force_clean_process "$pid" || continue
      start="$(proc_starttime "$pid" 2>/dev/null)" || continue
      cmd="$(proc_cmdline "$pid")"
      printf '%s\t%s\t%s\n' "$pid" "$start" "$cmd"
    done
    list_port_owner_records
  } | sort -u -k1,1
}

sudo_signal_pid_if_same() {
  local signal="$1" pid="$2" expected_start="$3" label="$4"
  proc_alive_same "$pid" "$expected_start" || return 0
  log "sudo ${signal} pid=${pid} starttime=${expected_start} component=${label}"
  sudo_kill_pid "$signal" "$pid" || return 1
}

force_terminate_records() {
  local timeout_s="$1"; shift
  local -a records=("$@")
  local record pid start cmd deadline any
  ((${#records[@]})) || return 0

  ensure_sudo
  for record in "${records[@]}"; do
    IFS=$'\t' read -r pid start cmd <<<"$record"
    sudo_signal_pid_if_same TERM "$pid" "$start" "$cmd" || true
  done

  deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    any=0
    for record in "${records[@]}"; do
      IFS=$'\t' read -r pid start cmd <<<"$record"
      proc_alive_same "$pid" "$start" && any=1
    done
    ((any == 0)) && return 0
    sleep 0.25
  done

  for record in "${records[@]}"; do
    IFS=$'\t' read -r pid start cmd <<<"$record"
    sudo_signal_pid_if_same KILL "$pid" "$start" "$cmd" || true
  done

  deadline=$((SECONDS + 5))
  while ((SECONDS < deadline)); do
    any=0
    for record in "${records[@]}"; do
      IFS=$'\t' read -r pid start cmd <<<"$record"
      proc_alive_same "$pid" "$start" && any=1
    done
    ((any == 0)) && return 0
    sleep 0.25
  done
  return 1
}

port_listener() {
  local port="$1"
  ss -H -ltnp "sport = :${port}" 2>/dev/null || true
}

port_is_listening() { [[ -n "$(port_listener "$1")" ]]; }

assert_port_free() {
  local port="$1" label="$2" owner
  owner="$(port_listener "$port")"
  if [[ -n "$owner" ]]; then
    error "${label} port ${port} is occupied:"
    printf '%s\n' "$owner" >&2
    return 1
  fi
}

can_bus_info() { ethtool -i "$1" 2>/dev/null | awk '$1 == "bus-info:" {print $2}'; }

check_can_mapping() {
  local left_bus right_bus
  ip link show "$CAN_LEFT" >/dev/null 2>&1 || { error "missing CAN interface ${CAN_LEFT}"; return 1; }
  ip link show "$CAN_RIGHT" >/dev/null 2>&1 || { error "missing CAN interface ${CAN_RIGHT}"; return 1; }
  left_bus="$(can_bus_info "$CAN_LEFT")"
  right_bus="$(can_bus_info "$CAN_RIGHT")"
  [[ "$left_bus" == "$CAN_LEFT_USB" ]] || { error "${CAN_LEFT} maps to ${left_bus:-<unknown>}, expected ${CAN_LEFT_USB}"; return 1; }
  [[ "$right_bus" == "$CAN_RIGHT_USB" ]] || { error "${CAN_RIGHT} maps to ${right_bus:-<unknown>}, expected ${CAN_RIGHT_USB}"; return 1; }
  [[ "$left_bus" != "$right_bus" ]] || { error "both CAN names resolve to the same USB interface"; return 1; }
  log "CAN mapping verified: ${CAN_LEFT}->${left_bus}, ${CAN_RIGHT}->${right_bus}"
}

check_can_active() {
  local dev details
  for dev in "$CAN_LEFT" "$CAN_RIGHT"; do
    details="$(ip -details link show "$dev" 2>/dev/null)" || return 1
    grep -Eq '^[0-9]+: [^:]+: <[^>]*UP[^>]*>' <<<"$details" || { error "${dev} is not UP"; return 1; }
    grep -q 'can state ERROR-ACTIVE' <<<"$details" || { error "${dev} is not ERROR-ACTIVE"; return 1; }
    grep -Eq "bitrate[[:space:]]+${CAN_BITRATE}([[:space:]]|$)" <<<"$details" || { error "${dev} bitrate is not ${CAN_BITRATE}"; return 1; }
    log "${dev} is UP, ERROR-ACTIVE, bitrate=${CAN_BITRATE}"
  done
}

check_no_can_receivers() {
  local dev count failures=0
  for dev in "$CAN_LEFT" "$CAN_RIGHT"; do
    if ! count="$(can_receiver_count "$dev")"; then
      error "${dev}: unable to inspect raw CAN receivers"
      failures=1
      continue
    fi
    if [[ "$count" != "0" ]]; then
      error "${dev} already has ${count} raw CAN receiver(s); refusing to start another CAN owner"
      failures=1
    else
      log "${dev} has no raw CAN receiver before piper_ros startup"
    fi
  done
  return "$failures"
}

check_piper_can_ownership_gate() {
  local dev count process_count failures=0
  for dev in "$CAN_LEFT" "$CAN_RIGHT"; do
    if ! count="$(can_receiver_count "$dev")"; then
      error "${dev}: unable to inspect raw CAN receivers after piper_ros startup"
      failures=1
      continue
    fi
    if [[ "$count" != "1" ]]; then
      error "${dev}: expected exactly one raw CAN receiver after piper_ros startup, found ${count}"
      failures=1
    fi
  done
  process_count="$(count_exact_arg_basename piper_single_ctrl)"
  if [[ "$process_count" != "2" ]]; then
    error "expected exactly two piper_single_ctrl processes before starting the ROS-only service, found ${process_count}"
    failures=1
  fi
  ((failures == 0)) && log "CAN action gate passed: one receiver per bus and exactly two piper_single_ctrl processes"
  return "$failures"
}

load_sudo_password() {
  local mode
  [[ -z "${SUDO_PASSWORD:-}" ]] || return 0
  [[ -r "$SUDO_PASSWORD_FILE" ]] || return 0
  [[ -O "$SUDO_PASSWORD_FILE" ]] || die "refusing sudo password file not owned by uid ${UID}: ${SUDO_PASSWORD_FILE}"
  mode="$(stat -c '%a' "$SUDO_PASSWORD_FILE")"
  [[ "$mode" == "600" || "$mode" == "400" ]] || \
    die "sudo password file must be chmod 600 or 400: ${SUDO_PASSWORD_FILE}"
  IFS= read -r SUDO_PASSWORD <"$SUDO_PASSWORD_FILE" || die "failed to read sudo password file: ${SUDO_PASSWORD_FILE}"
  export SUDO_PASSWORD
}

ensure_sudo() {
  if sudo -n true 2>/dev/null; then
    return 0
  fi
  load_sudo_password
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' -v >/dev/null || die "sudo authentication failed"
    return 0
  fi
  [[ -t 0 ]] || die "sudo requires cached/passwordless sudo, SUDO_PASSWORD, or ${SUDO_PASSWORD_FILE}"
  log "sudo authentication is required for privileged cleanup/CAN activation"
  sudo -v
}

activate_can() {
  if check_can_mapping && check_can_active; then
    log "CAN interfaces are already correctly mapped and active; no privileged reset needed"
    return 0
  fi
  ensure_sudo
  log "activating ${CAN_LEFT} at ${CAN_LEFT_USB}"
  bash "${PIPER_SDK_DIR}/can_activate.sh" "$CAN_LEFT" "$CAN_BITRATE" "$CAN_LEFT_USB"
  log "activating ${CAN_RIGHT} at ${CAN_RIGHT_USB}"
  bash "${PIPER_SDK_DIR}/can_activate.sh" "$CAN_RIGHT" "$CAN_BITRATE" "$CAN_RIGHT_USB"
  check_can_mapping
  check_can_active
}

check_camera_serials() {
  local inventory serial count attempt all_present
  for attempt in {1..10}; do
    inventory="$(timeout 20 rs-enumerate-devices -s 2>/dev/null || true)"
    all_present=1
    for serial in "$LEFT_CAMERA_SERIAL" "$RIGHT_CAMERA_SERIAL" "$TOP_CAMERA_SERIAL"; do
      count="$(awk -v wanted="$serial" 'index($0, wanted) {count++} END {print count+0}' <<<"$inventory")"
      [[ "$count" == "1" ]] || all_present=0
    done
    if ((all_present)); then
      log "camera serials verified: ${LEFT_CAMERA_SERIAL}, ${RIGHT_CAMERA_SERIAL}, ${TOP_CAMERA_SERIAL}"
      return 0
    fi
    warn "camera inventory not stable yet (attempt ${attempt}/10); retrying"
    sleep 2
  done
  error "camera serial verification failed after 10 attempts"
  printf '%s\n' "$inventory" >&2
  return 1
}

source_has_forbidden_sdk() {
  local output
  output="$(grep -En 'C_PiperInterface(_V2)?|piper_sdk|MotionCtrl_2|JointCtrl[[:space:]]*\(|GripperCtrl[[:space:]]*\(|AF_CAN|PF_CAN|SOCK_RAW|socketcan|python-can|(^|[[:space:]])import[[:space:]]+can([[:space:]]|$)|(^|[[:space:]])from[[:space:]]+can[[:space:]]+import([[:space:]]|$)' \
    "$ROS_SERVICE_SCRIPT" 2>/dev/null || true)"
  if [[ -n "$output" ]]; then
    error "ROS-only source contains forbidden direct Piper SDK/CAN calls:"
    printf '%s\n' "$output" >&2
    return 0
  fi
  return 1
}

static_preflight() {
  local command_name
  for command_name in bash setsid ps ss ip ethtool awk sed grep timeout readlink realpath sudo rs-enumerate-devices; do
    require_command "$command_name"
  done
  require_file "$ROS_SETUP"
  require_file "$PIPER_SETUP"
  require_dir "$PIPER_SDK_DIR"
  require_file "${PIPER_SDK_DIR}/can_activate.sh"
  require_dir "$OPENPI_DIR"
  require_file "$ROS_SERVICE_SCRIPT"
  require_exec "$ROS_SERVICE_PYTHON"

  load_ros_env
  require_command ros2

  [[ "$POLICY_PORT" =~ ^[0-9]+$ && "$ROS_SERVICE_PORT" =~ ^[0-9]+$ ]] || die "ports must be numeric"
  [[ "$CAN_BITRATE" =~ ^[0-9]+$ ]] || die "CAN_BITRATE must be numeric"
  [[ "$ROS_DOMAIN_ID" == "$EXPECTED_ROS_DOMAIN_ID" ]] || die "internal ROS_DOMAIN_ID invariant failed"
  if source_has_forbidden_sdk; then
    return 1
  fi
  return 0
}

report_preflight_conflicts() {
  local failures=0 owner port label
  local -a records=()
  mapfile -t records < <(list_legacy_records)
  if ((${#records[@]})); then
    error "legacy RTC/camera/Piper processes are running:"
    printf '  %s\n' "${records[@]}" >&2
    failures=1
  fi
  if legacy_tmux_exists; then
    error "exact legacy tmux session ${LEGACY_TMUX_SESSION} exists"
    failures=1
  fi
  for owner in "${POLICY_PORT}:policy" "${ROS_SERVICE_PORT}:ROS-only-service" "9900:legacy-RobotArmService"; do
    IFS=: read -r port label <<<"$owner"
    assert_port_free "$port" "$label" || failures=1
  done
  check_camera_serials || failures=1
  check_can_mapping || failures=1
  check_can_active || failures=1
  check_no_can_receivers || failures=1
  return "$failures"
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_HELD=1
    return 0
  fi
  die "stack lock exists at ${LOCK_DIR}; run status or stop first"
}

release_lock() {
  if ((LOCK_HELD)); then
    rmdir "$LOCK_DIR" 2>/dev/null || true
    LOCK_HELD=0
  fi
}

write_state_header() {
  local self_start
  self_start="$(proc_starttime "$$")" || die "cannot read supervisor starttime"
  : >"$STATE_FILE"
  printf 'VERSION\t%s\n' "$STATE_VERSION" >>"$STATE_FILE"
  printf 'SUPERVISOR\t%s\t%s\n' "$$" "$self_start" >>"$STATE_FILE"
  printf 'LOG_DIR\t%s\n' "$LOG_DIR" >>"$STATE_FILE"
}

is_managed_component_name() {
  case "$1" in
    camera_left|camera_right|camera_top|piper|ros_service) return 0 ;;
    *) return 1 ;;
  esac
}

append_process_state() {
  local index="$1"
  printf 'PROCESS\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${PROC_NAMES[index]}" "${PROC_PIDS[index]}" "${PROC_STARTS[index]}" \
    "${PROC_PGIDS[index]}" "${PROC_SIDS[index]}" "${PROC_LOGS[index]}" >>"$STATE_FILE"
}

load_state() {
  local kind a b c d e f version=""
  [[ -f "$STATE_FILE" ]] || return 1
  [[ -O "$STATE_FILE" ]] || die "refusing state file not owned by uid ${UID}: ${STATE_FILE}"

  PROC_NAMES=(); PROC_PIDS=(); PROC_STARTS=(); PROC_PGIDS=(); PROC_SIDS=(); PROC_LOGS=()
  STATE_SUPERVISOR_PID=""; STATE_SUPERVISOR_START=""
  while IFS=$'\t' read -r kind a b c d e f; do
    case "$kind" in
      VERSION)
        [[ -z "$version" && "$a" == "$STATE_VERSION" ]] || \
          die "unsupported or duplicate state version ${a:-<missing>}; refusing to signal any process"
        version="$a"
        ;;
      SUPERVISOR)
        [[ "$a" =~ ^[0-9]+$ && "$b" =~ ^[0-9]+$ ]] || die "invalid supervisor state"
        STATE_SUPERVISOR_PID="$a"; STATE_SUPERVISOR_START="$b"
        ;;
      PROCESS)
        is_managed_component_name "$a" || \
          die "unmanaged component '${a}' in state; refusing to signal any process"
        [[ "$b" =~ ^[0-9]+$ && "$c" =~ ^[0-9]+$ && "$d" =~ ^[0-9]+$ && "$e" =~ ^[0-9]+$ ]] || die "invalid process state row"
        PROC_NAMES+=("$a"); PROC_PIDS+=("$b"); PROC_STARTS+=("$c")
        PROC_PGIDS+=("$d"); PROC_SIDS+=("$e"); PROC_LOGS+=("$f")
        ;;
      LOG_DIR) LOG_DIR="$a" ;;
      '') ;;
      *) die "unknown state row '${kind}'; refusing to signal any process" ;;
    esac
  done <"$STATE_FILE"
  [[ "$version" == "$STATE_VERSION" ]] || die "state file lacks supported version ${STATE_VERSION}"
  [[ -n "$STATE_SUPERVISOR_PID" ]] || die "state file lacks supervisor identity"
}

state_file_matches_supervisor() {
  local expected_pid="$1" expected_start="$2" kind pid start rest
  [[ -f "$STATE_FILE" && -O "$STATE_FILE" ]] || return 1
  while IFS=$'\t' read -r kind pid start rest; do
    if [[ "$kind" == "SUPERVISOR" ]]; then
      [[ "$pid" == "$expected_pid" && "$start" == "$expected_start" ]]
      return
    fi
  done <"$STATE_FILE"
  return 1
}

spawn_group() {
  local name="$1" cwd="$2" log_file pid start pgid sid index
  shift 2
  is_managed_component_name "$name" || die "refusing to spawn unmanaged component: ${name}"
  log_file="${LOG_DIR}/${name}.log"
  : >"$log_file"
  log "starting ${name}; log=${log_file}"
  SPAWN_CRITICAL=1
  (
    cd "$cwd"
    exec setsid --wait "$@"
  ) >>"$log_file" 2>&1 &
  pid=$!
  PENDING_NAME="$name"
  PENDING_PID="$pid"
  PENDING_START=""
  PENDING_SID=""
  PENDING_LOG="$log_file"
  SPAWN_CRITICAL=0
  if ((DEFERRED_EXIT_CODE)); then
    local deferred_code="$DEFERRED_EXIT_CODE"
    DEFERRED_EXIT_CODE=0
    exit "$deferred_code"
  fi

  local deadline=$((SECONDS + 10))
  while ((SECONDS < deadline)); do
    start="$(proc_starttime "$pid" 2>/dev/null || true)"
    [[ -n "$start" ]] && break
    sleep 0.1
  done
  [[ -n "${start:-}" ]] || { tail -80 "$log_file" >&2 || true; die "${name} failed before PID registration"; }
  PENDING_START="$start"
  pgid="$(proc_pgid "$pid")"
  sid="$(proc_sid "$pid")"
  PENDING_SID="$sid"
  if [[ "$pgid" != "$pid" || "$sid" != "$pid" ]]; then
    terminate_pid_records "$TERM_TIMEOUT" TERM "$pid"$'\t'"$start"$'\t'"$name" || true
    die "${name} is not isolated by setsid: pid=${pid} pgid=${pgid} sid=${sid}"
  fi

  PROC_NAMES+=("$name"); PROC_PIDS+=("$pid"); PROC_STARTS+=("$start")
  PROC_PGIDS+=("$pgid"); PROC_SIDS+=("$sid"); PROC_LOGS+=("$log_file")
  index=$((${#PROC_NAMES[@]} - 1))
  append_process_state "$index"
  PENDING_NAME=""
  PENDING_PID=""
  PENDING_START=""
  PENDING_SID=""
  PENDING_LOG=""
  log "registered ${name}: pid=${pid} starttime=${start} pgid=${pgid} sid=${sid}"
}

session_member_records() {
  local sid="$1" pid member_sid start cmd
  while read -r pid member_sid; do
    [[ "$pid" =~ ^[0-9]+$ && "$member_sid" == "$sid" ]] || continue
    [[ "$pid" != "$$" ]] || continue
    start="$(proc_starttime "$pid" 2>/dev/null)" || continue
    cmd="$(proc_cmdline "$pid")"
    printf '%s\t%s\t%s\n' "$pid" "$start" "$cmd"
  done < <(ps -eo pid=,sid=)
}

cleanup_pending_group() {
  local current_start current_ppid sid first_signal=TERM timeout_s="$TERM_TIMEOUT"
  local -a records=()
  [[ -n "$PENDING_PID" ]] || return 0

  current_start="$(proc_starttime "$PENDING_PID" 2>/dev/null || true)"
  if [[ -z "$current_start" ]]; then
    if [[ -n "$PENDING_SID" ]]; then
      mapfile -t records < <(session_member_records "$PENDING_SID")
      if ((${#records[@]})); then
        log "pending leader exited; cleaning surviving session ${PENDING_SID} for ${PENDING_NAME}"
        terminate_pid_records "$timeout_s" "$first_signal" "${records[@]}" || return 1
      fi
    fi
    PENDING_NAME=""; PENDING_PID=""; PENDING_START=""; PENDING_SID=""; PENDING_LOG=""
    return 0
  fi
  if [[ -z "$PENDING_START" ]]; then
    current_ppid="$(ps -o ppid= -p "$PENDING_PID" 2>/dev/null | tr -d '[:space:]')"
    [[ "$current_ppid" == "$$" ]] || {
      error "refusing ambiguous pending PID ${PENDING_PID}: it is no longer a direct child"
      return 1
    }
    PENDING_START="$current_start"
  elif [[ "$current_start" != "$PENDING_START" ]]; then
    error "refusing reused pending PID ${PENDING_PID}: recorded=${PENDING_START}, current=${current_start}"
    return 1
  fi

  sid="$(proc_sid "$PENDING_PID")"
  PENDING_SID="$sid"
  if [[ "$sid" == "$PENDING_PID" ]]; then
    mapfile -t records < <(session_member_records "$sid")
  else
    records+=("${PENDING_PID}"$'\t'"${PENDING_START}"$'\t'"pending ${PENDING_NAME}")
  fi
  log "cleaning pending component ${PENDING_NAME}: pid=${PENDING_PID} sid=${sid:-unknown}"
  terminate_pid_records "$timeout_s" "$first_signal" "${records[@]}" || return 1
  PENDING_NAME=""; PENDING_PID=""; PENDING_START=""; PENDING_SID=""; PENDING_LOG=""
}

terminate_group_index() {
  local index="$1" timeout_s="$2" first_signal="${3:-TERM}" name leader leader_start sid current_start round
  local -a records=()
  name="${PROC_NAMES[index]}"
  leader="${PROC_PIDS[index]}"
  leader_start="${PROC_STARTS[index]}"
  sid="${PROC_SIDS[index]}"

  if [[ -e "/proc/${leader}" ]]; then
    current_start="$(proc_starttime "$leader" 2>/dev/null || true)"
    if [[ -n "$current_start" && "$current_start" != "$leader_start" ]]; then
      error "refusing reused leader PID ${leader} for ${name}: recorded=${leader_start}, current=${current_start}"
      return 1
    fi
  fi

  for round in 1 2 3; do
    mapfile -t records < <(session_member_records "$sid")
    ((${#records[@]})) || return 0
    log "ordered shutdown ${name} round ${round}: ${#records[@]} PID(s) in recorded sid=${sid}"
    terminate_pid_records "$timeout_s" "$first_signal" "${records[@]}" || true
  done

  mapfile -t records < <(session_member_records "$sid")
  if ((${#records[@]})); then
    error "session ${sid} for ${name} still has processes"
    printf '  %s\n' "${records[@]}" >&2
    return 1
  fi
}

find_process_index() {
  local wanted="$1" i
  for ((i = 0; i < ${#PROC_NAMES[@]}; i++)); do
    [[ "${PROC_NAMES[i]}" == "$wanted" ]] && { printf '%s\n' "$i"; return 0; }
  done
  return 1
}

terminate_named_group() {
  local name="$1" timeout_s="$2" first_signal="${3:-TERM}" index
  index="$(find_process_index "$name" 2>/dev/null)" || return 0
  terminate_group_index "$index" "$timeout_s" "$first_signal"
}

any_recorded_session_alive() {
  local sid
  for sid in "${PROC_SIDS[@]}"; do
    [[ -n "$(session_member_records "$sid")" ]] && return 0
  done
  return 1
}

ordered_group_cleanup() {
  local failed=0
  terminate_named_group ros_service "$TERM_TIMEOUT" || failed=1
  terminate_named_group piper "$TERM_TIMEOUT" || failed=1
  terminate_named_group camera_top "$TERM_TIMEOUT" || failed=1
  terminate_named_group camera_right "$TERM_TIMEOUT" || failed=1
  terminate_named_group camera_left "$TERM_TIMEOUT" || failed=1
  return "$failed"
}

cleanup_current_stack() {
  local rc=0
  ((CLEANUP_RUNNING == 0)) || return 0
  CLEANUP_RUNNING=1
  set +e
  log "begin exact ordered cleanup"
  cleanup_pending_group || rc=1
  ordered_group_cleanup || rc=1
  cleanup_legacy_infra_stack || rc=1
  if ! any_recorded_session_alive; then
    post_cleanup_checks || rc=1
  else
    rc=1
  fi
  if ((rc == 0)); then
    rm -f "$STATE_FILE"
    release_lock
    log "all recorded process sessions stopped"
  else
    error "cleanup left a recorded process; state retained at ${STATE_FILE}"
  fi
  CLEANUP_RUNNING=0
  set -e
  return "$rc"
}

on_exit() {
  local rc=$?
  trap - EXIT
  # A second Ctrl+C/TERM during ordered cleanup must not strand later
  # components.  The current cleanup continues to completion.
  trap '' INT TERM HUP
  if ((CLEANUP_ARMED)); then
    cleanup_current_stack || rc=1
  else
    release_lock
  fi
  exit "$rc"
}

stop_recorded_stack() {
  local verify_global_cleanup="${1:-1}"
  local supervisor_pid supervisor_start deadline current_start
  local -a saved_names saved_pids saved_starts saved_pgids saved_sids saved_logs
  [[ "$verify_global_cleanup" == "0" || "$verify_global_cleanup" == "1" ]] || \
    die "invalid stop verification mode: ${verify_global_cleanup}"
  if ! load_state; then
    if [[ -d "$LOCK_DIR" ]]; then
      die "stack lock exists without a readable state file; refusing to remove an active or ambiguous lock: ${LOCK_DIR}"
    fi
    log "no managed state at ${STATE_FILE}"
    return 0
  fi

  supervisor_pid="$STATE_SUPERVISOR_PID"
  supervisor_start="$STATE_SUPERVISOR_START"
  saved_names=("${PROC_NAMES[@]}"); saved_pids=("${PROC_PIDS[@]}")
  saved_starts=("${PROC_STARTS[@]}"); saved_pgids=("${PROC_PGIDS[@]}")
  saved_sids=("${PROC_SIDS[@]}"); saved_logs=("${PROC_LOGS[@]}")

  if [[ "$supervisor_pid" != "$$" ]] && proc_alive_same "$supervisor_pid" "$supervisor_start"; then
    signal_pid_if_same TERM "$supervisor_pid" "$supervisor_start" supervisor || true
    deadline=$((SECONDS + STOP_SUPERVISOR_TIMEOUT))
    while ((SECONDS < deadline)) && proc_alive_same "$supervisor_pid" "$supervisor_start"; do sleep 0.5; done
    if proc_alive_same "$supervisor_pid" "$supervisor_start"; then
      die "supervisor is still performing ordered cleanup after ${STOP_SUPERVISOR_TIMEOUT}s; refusing to interrupt it"
    fi
  elif [[ -e "/proc/${supervisor_pid}" ]]; then
    current_start="$(proc_starttime "$supervisor_pid" 2>/dev/null || true)"
    [[ "$current_start" == "$supervisor_start" ]] || warn "supervisor PID was reused; it was not signalled"
  fi

  # Restore the loaded snapshot in case the supervisor removed the state file.
  PROC_NAMES=("${saved_names[@]}"); PROC_PIDS=("${saved_pids[@]}")
  PROC_STARTS=("${saved_starts[@]}"); PROC_PGIDS=("${saved_pgids[@]}")
  PROC_SIDS=("${saved_sids[@]}"); PROC_LOGS=("${saved_logs[@]}")
  local cleanup_failed=0
  cleanup_pending_group || cleanup_failed=1
  ordered_group_cleanup || cleanup_failed=1
  if any_recorded_session_alive; then
    die "recorded process session remains; state retained"
  fi
  if ((verify_global_cleanup)); then
    post_cleanup_checks || cleanup_failed=1
  else
    # A restart performs exact legacy cleanup immediately after the recorded
    # stack is gone.  Do not let an unrelated, approved legacy CAN owner keep
    # a stale managed state from reaching that cleanup phase.
    log "recorded sessions stopped; deferring global CAN/port checks to restart preflight"
  fi
  ((cleanup_failed == 0)) || die "cleanup verification failed; state retained"
  if [[ -f "$STATE_FILE" ]]; then
    state_file_matches_supervisor "$supervisor_pid" "$supervisor_start" || \
      die "state now belongs to another supervisor; refusing to delete ${STATE_FILE} or ${LOCK_DIR}"
    rm -f "$STATE_FILE"
    rmdir "$LOCK_DIR" 2>/dev/null || \
      die "could not remove the stopped supervisor lock ${LOCK_DIR}"
  elif [[ -d "$LOCK_DIR" ]]; then
    die "state disappeared but a lock is held; refusing to remove a possibly new supervisor lock"
  fi
  log "managed stack stopped"
}

node_count() {
  local node="$1"
  ros2 node list 2>/dev/null | grep -Fxc "$node" || true
}

topic_counts() {
  local topic="$1" info pub sub
  info="$(ros2 topic info "$topic" 2>/dev/null || true)"
  pub="$(awk -F': ' '/^Publisher count:/ {print $2}' <<<"$info")"
  sub="$(awk -F': ' '/^Subscription count:/ {print $2}' <<<"$info")"
  printf '%s %s\n' "${pub:-0}" "${sub:-0}"
}

wait_for_ros_base() {
  local deadline=$((SECONDS + ROS_START_TIMEOUT)) now count pub sub ok
  local -a nodes=(
    /camera/left/camera /camera/right/camera /camera/top/camera
    /piper_left_ctrl_node /piper_right_ctrl_node
  )
  local -a topics=(
    /camera/left/camera/color/image_raw
    /camera/right/camera/color/image_raw
    /camera/top/camera/color/image_raw
    /joint_states_left /joint_states_right
  )
  while ((SECONDS < deadline)); do
    ok=1
    for now in "${nodes[@]}"; do
      count="$(node_count "$now")"
      [[ "$count" == "1" ]] || ok=0
      ((count <= 1)) || { error "duplicate ROS node appeared: ${now} count=${count}"; return 1; }
    done
    for now in "${topics[@]}"; do
      read -r pub sub < <(topic_counts "$now")
      [[ "$pub" == "1" ]] || ok=0
      ((pub <= 1)) || { error "duplicate publisher appeared: ${now} count=${pub}"; return 1; }
    done
    ((ok)) && { log "one camera/Piper node and one publisher per feedback topic are ready"; return 0; }
    sleep 1
  done
  error "timed out waiting for unique camera/Piper ROS graph"
  ros2 node list >&2 || true
  return 1
}

wait_for_port() {
  local port="$1" timeout_s="$2" label="$3" deadline
  deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    port_is_listening "$port" && { log "${label} is listening on port ${port}"; return 0; }
    sleep 1
    ((SECONDS % 15)) || log "waiting for ${label} port ${port}"
  done
  error "timed out waiting for ${label} port ${port}"
  return 1
}

wait_for_log_marker() {
  local log_file="$1" marker="$2" timeout_s="$3" label="$4" deadline
  deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    grep -Fq "$marker" "$log_file" 2>/dev/null && { log "${label} marker observed: ${marker}"; return 0; }
    sleep 1
  done
  error "timed out waiting for ${label} marker '${marker}' in ${log_file}"
  tail -100 "$log_file" >&2 || true
  return 1
}

wait_for_old_ros_nodes_to_leave() {
  local deadline=$((SECONDS + 30)) node count any
  local -a nodes=(
    /camera/left/camera /camera/right/camera /camera/top/camera
    /piper_left_ctrl_node /piper_right_ctrl_node /robot_arm_service_collector
    /ros_only_robot_control_service
  )
  while ((SECONDS < deadline)); do
    any=0
    for node in "${nodes[@]}"; do
      count="$(node_count "$node")"
      ((count == 0)) || any=1
    done
    ((any == 0)) && return 0
    sleep 1
  done
  error "old target ROS nodes remain visible after exact process cleanup"
  return 1
}

post_cleanup_checks() {
  local failures=0 dev count port

  check_can_mapping || failures=1
  check_can_active || failures=1
  for dev in "$CAN_LEFT" "$CAN_RIGHT"; do
    if ! count="$(can_receiver_count "$dev")"; then
      error "post-cleanup ${dev}: unable to inspect raw CAN receivers"
      failures=1
      continue
    fi
    if [[ "$count" != "0" ]]; then
      error "post-cleanup ${dev}: expected zero raw CAN receivers, found ${count}"
      failures=1
    else
      log "post-cleanup ${dev}: UP/ERROR-ACTIVE with zero raw receivers"
    fi
  done

  for port in "$ROS_SERVICE_PORT" 9900; do
    if port_is_listening "$port"; then
      error "post-cleanup port ${port} is still listening: $(port_listener "$port")"
      failures=1
    fi
  done

  load_ros_env
  wait_for_old_ros_nodes_to_leave || failures=1

  ((failures == 0)) && log "post-cleanup verification passed: CAN active, infrastructure ports free, managed ROS nodes absent"
  return "$failures"
}

topic_has_data() {
  local topic="$1" output
  output="$(timeout 5 ros2 topic hz --window 10 "$topic" 2>&1 || true)"
  if ! grep -q 'average rate:' <<<"$output"; then
    error "no data observed on ${topic} during 5-second sample"
    return 1
  fi
  log "data observed on ${topic}: $(grep 'average rate:' <<<"$output" | tail -1)"
}

count_exact_arg_basename() {
  local wanted="$1" proc pid arg count=0
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ -r "${proc}/cmdline" ]] || continue
    while IFS= read -r -d '' arg; do
      if [[ "$arg" != *' '* && "${arg##*/}" == "$wanted" ]]; then
        ((count += 1))
        break
      fi
    done <"${proc}/cmdline" || true
  done
  printf '%s\n' "$count"
}

count_exact_arg_value() {
  local wanted="$1" proc arg count=0
  for proc in /proc/[0-9]*; do
    [[ -r "${proc}/cmdline" ]] || continue
    while IFS= read -r -d '' arg; do
      if [[ "$arg" == "$wanted" ]]; then
        ((count += 1))
        break
      fi
    done <"${proc}/cmdline" || true
  done
  printf '%s\n' "$count"
}

can_receiver_count() {
  local device="$1" receiver_table="/proc/net/can/rcvlist_all"

  # On a fresh boot the gs_usb CAN netdevices can already be configured and UP
  # while PF_CAN/can_raw has never been requested.  In that state the kernel
  # does not expose /proc/net/can yet, and there cannot be a raw CAN socket.
  # Treat only that precise state as zero.  Once can_raw is loaded, a missing
  # accounting table is an inspection failure rather than an empty count.
  if [[ ! -r "$receiver_table" ]]; then
    if [[ ! -d /sys/module/can_raw ]]; then
      printf '0\n'
      return 0
    fi
    error "can_raw is loaded but ${receiver_table} is unavailable"
    return 1
  fi

  awk -v device="$device" '$1 == device {count++} END {print count+0}' \
    "$receiver_table"
}

validate_running_stack() {
  local failures=0 node expected count topic pub sub process_count receiver_count
  local service_log="" i
  local -a node_specs=(
    /camera/left/camera:1 /camera/right/camera:1 /camera/top/camera:1
    /piper_left_ctrl_node:1 /piper_right_ctrl_node:1
    /ros_only_robot_control_service:1
  )
  local -a feedback_topics=(
    /camera/left/camera/color/image_raw
    /camera/right/camera/color/image_raw
    /camera/top/camera/color/image_raw
    /joint_states_left /joint_states_right
  )

  check_can_mapping || failures=1
  check_can_active || failures=1

  for ((i = 0; i < ${#PROC_NAMES[@]}; i++)); do
    proc_alive_same "${PROC_PIDS[i]}" "${PROC_STARTS[i]}" || {
      error "managed component is not alive: ${PROC_NAMES[i]} pid=${PROC_PIDS[i]}"
      failures=1
    }
  done

  for node in "${node_specs[@]}"; do
    expected="${node##*:}"; node="${node%:*}"
    count="$(node_count "$node")"
    [[ "$count" == "$expected" ]] || { error "node ${node}: expected ${expected}, found ${count}"; failures=1; }
  done

  for topic in "${feedback_topics[@]}"; do
    read -r pub sub < <(topic_counts "$topic")
    [[ "$pub" == "1" ]] || { error "topic ${topic}: expected one publisher, found ${pub}"; failures=1; }
  done

  for topic in /joint_ctrl_cmd_left /joint_ctrl_cmd_right; do
    read -r pub sub < <(topic_counts "$topic")
    [[ "$pub" == "1" ]] || { error "command topic ${topic}: expected ROS-only service publisher=1, found ${pub}"; failures=1; }
    [[ "$sub" == "1" ]] || { error "command topic ${topic}: expected piper_ros subscriber=1, found ${sub}"; failures=1; }
  done

  process_count="$(count_exact_arg_basename piper_single_ctrl)"
  [[ "$process_count" == "2" ]] || { error "expected exactly two piper_single_ctrl processes, found ${process_count}"; failures=1; }
  process_count="$(count_exact_arg_value /opt/ros/humble/lib/realsense2_camera/realsense2_camera_node)"
  [[ "$process_count" == "3" ]] || { error "expected exactly three RealSense node processes, found ${process_count}"; failures=1; }
  process_count="$((
    $(count_exact_arg_basename start_robot_arm_service_chw.py) +
    $(count_exact_arg_basename start_robot_arm_service.py)
  ))"
  [[ "$process_count" == "0" ]] || {
    error "legacy SDK RobotArmService process(es) are still running: ${process_count}"
    failures=1
  }
  source_has_forbidden_sdk && failures=1

  for node in "$CAN_LEFT" "$CAN_RIGHT"; do
    if ! receiver_count="$(can_receiver_count "$node")"; then
      error "${node}: unable to inspect raw CAN receivers during validation"
      failures=1
      continue
    fi
    [[ "$receiver_count" == "1" ]] || {
      error "${node}: expected one raw CAN receiver owned by piper_ros, found ${receiver_count}"
      failures=1
    }
  done

  port_is_listening "$ROS_SERVICE_PORT" || { error "ROS-only service port ${ROS_SERVICE_PORT} is not listening"; failures=1; }

  for topic in "${feedback_topics[@]}"; do topic_has_data "$topic" || failures=1; done

  for ((i = 0; i < ${#PROC_NAMES[@]}; i++)); do
    case "${PROC_NAMES[i]}" in
      ros_service) service_log="${PROC_LOGS[i]}" ;;
    esac
  done
  [[ -n "$service_log" && -f "$service_log" ]] || { error "managed ROS-only service log not found"; failures=1; }
  [[ -z "$service_log" ]] || grep -Fq 'READY' "$service_log" || { error "ROS-only service READY marker absent"; failures=1; }

  ((failures == 0)) || return 1
  log "validation passed: cameras, piper_ros and ROS-only adapter are ready; no SDK RobotArmService"
}

start_components() {
  spawn_group camera_left "$OPENPI_DIR" \
    ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera/left camera_name:=camera \
    "serial_no:='${LEFT_CAMERA_SERIAL}'" enable_color:=true enable_depth:=true \
    "depth_module.color_profile:=${CAMERA_PROFILE}" "depth_module.depth_profile:=${CAMERA_PROFILE}" \
    enable_gyro:=false enable_accel:=false

  spawn_group camera_right "$OPENPI_DIR" \
    ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera/right camera_name:=camera \
    "serial_no:='${RIGHT_CAMERA_SERIAL}'" enable_color:=true enable_depth:=true \
    "depth_module.color_profile:=${CAMERA_PROFILE}" "depth_module.depth_profile:=${CAMERA_PROFILE}" \
    enable_gyro:=false enable_accel:=false

  spawn_group camera_top "$OPENPI_DIR" \
    ros2 run realsense2_camera realsense2_camera_node --ros-args \
    -r __ns:=/camera/top -r __node:=camera \
    -p "serial_no:='${TOP_CAMERA_SERIAL}'" \
    -p enable_color:=true -p enable_depth:=false -p enable_infra1:=false -p enable_infra2:=false \
    -p enable_gyro:=false -p enable_accel:=false -p "rgb_camera.color_profile:=${CAMERA_PROFILE}"

  spawn_group piper "$OPENPI_DIR" \
    ros2 launch piper start_two_piper.launch.py \
    "can_left_port:=${CAN_LEFT}" "can_right_port:=${CAN_RIGHT}" \
    auto_enable:=true gripper_exist:=true

  wait_for_ros_base
  check_piper_can_ownership_gate || die "CAN ownership changed before ROS-only control service startup"

  spawn_group ros_service "$OPENPI_DIR" \
    "$ROS_SERVICE_PYTHON" "$ROS_SERVICE_SCRIPT" \
    --host "$ROS_SERVICE_HOST" --port "$ROS_SERVICE_PORT" \
    --topic-wait-timeout "$ROS_START_TIMEOUT"
  wait_for_port "$ROS_SERVICE_PORT" "$SERVICE_START_TIMEOUT" "ROS-only robot service"
  wait_for_log_marker "${LOG_DIR}/ros_service.log" "READY" "$SERVICE_START_TIMEOUT" "ROS-only robot service"

}

monitor_components() {
  local i
  log "ROS infrastructure is running; start policy/client separately. Ctrl+C stops only this managed infrastructure"
  while true; do
    sleep 2
    for ((i = 0; i < ${#PROC_NAMES[@]}; i++)); do
      if ! proc_alive_same "${PROC_PIDS[i]}" "${PROC_STARTS[i]}"; then
        error "managed component exited: ${PROC_NAMES[i]} (log ${PROC_LOGS[i]})"
        tail -100 "${PROC_LOGS[i]}" >&2 || true
        return 1
      fi
    done
  done
}

preflight_mode() {
  static_preflight
  if report_preflight_conflicts; then
    log "preflight passed: no old full-chain conflicts, CAN is active, startup ports are free"
  else
    die "preflight found conflicts; start mode can exact-clean approved legacy processes"
  fi
}

status_mode() {
  local i status owner
  printf 'stack=%s state=%s lock=%s\n' "$STACK_NAME" "$STATE_FILE" "$LOCK_DIR"
  printf 'ROS_DOMAIN_ID=%s ROS_LOCALHOST_ONLY=0\n' "$EXPECTED_ROS_DOMAIN_ID"
  if load_state; then
    printf 'supervisor pid=%s starttime=%s alive=%s\n' \
      "$STATE_SUPERVISOR_PID" "$STATE_SUPERVISOR_START" \
      "$(proc_alive_same "$STATE_SUPERVISOR_PID" "$STATE_SUPERVISOR_START" && echo yes || echo no)"
    for ((i = 0; i < ${#PROC_NAMES[@]}; i++)); do
      status=no; proc_alive_same "${PROC_PIDS[i]}" "${PROC_STARTS[i]}" && status=yes
      printf 'process name=%s pid=%s starttime=%s pgid=%s sid=%s alive=%s log=%s\n' \
        "${PROC_NAMES[i]}" "${PROC_PIDS[i]}" "${PROC_STARTS[i]}" \
        "${PROC_PGIDS[i]}" "${PROC_SIDS[i]}" "$status" "${PROC_LOGS[i]}"
    done
  else
    echo 'managed_state=absent'
  fi
  echo '-- exact legacy tmux --'
  legacy_tmux_exists && echo "${LEGACY_TMUX_SESSION}: present" || echo "${LEGACY_TMUX_SESSION}: absent"
  echo '-- ports --'
  for owner in "$ROS_SERVICE_PORT" 9900; do
    printf 'port %s: ' "$owner"
    if port_is_listening "$owner"; then echo listening; port_listener "$owner"; else echo free; fi
  done
  echo '-- CAN --'
  for owner in "$CAN_LEFT" "$CAN_RIGHT"; do
    ip -details link show "$owner" 2>&1 | grep -E '^[0-9]+:|can state|bitrate' || true
    printf 'bus-info=%s\n' "$(can_bus_info "$owner")"
  done
  echo '-- approved next-start full-chain cleanup matches --'
  list_legacy_records || true
  if [[ -f "$ROS_SETUP" && -f "$PIPER_SETUP" ]]; then
    load_ros_env
    echo '-- target ROS nodes --'
    ros2 node list 2>/dev/null | grep -E '^/camera/(left|right|top)/camera$|^/piper_(left|right)_ctrl_node$|^/(robot_arm_service_collector|ros_only_robot_control_service)$' || true
  fi
}

validate_mode() {
  load_state || die "no managed state to validate"
  static_preflight
  validate_running_stack
}

force_clean_mode() {
  local round sessions command_name failures=0
  local -a records=()

  for command_name in ps ss awk sed grep cut sort sudo bash; do
    require_command "$command_name"
  done
  ensure_sudo

  if command -v tmux >/dev/null 2>&1; then
    sessions="$(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)"
    if [[ -n "$sessions" ]]; then
      log "force-clean killing all tmux sessions:"
      printf '%s\n' "$sessions" >&2
      tmux kill-server 2>/dev/null || true
    fi
  fi

  if [[ -f "$ROS_SETUP" && -f "$PIPER_SETUP" ]]; then
    load_ros_env
    timeout 5 ros2 daemon stop >/dev/null 2>&1 || true
  fi

  for round in 1 2 3 4; do
    mapfile -t records < <(list_force_clean_records)
    ((${#records[@]})) || break
    log "force-clean round ${round}: ${#records[@]} process(es)"
    printf '  %s\n' "${records[@]}" >&2
    force_terminate_records 5 "${records[@]}" || true
  done

  mapfile -t records < <(list_force_clean_records)
  if ((${#records[@]})); then
    error "force-clean left matching process(es):"
    printf '  %s\n' "${records[@]}" >&2
    failures=1
  fi

  rm -f "$STATE_FILE"
  rmdir "$LOCK_DIR" 2>/dev/null || true

  if [[ -f "$ROS_SETUP" && -f "$PIPER_SETUP" ]]; then
    load_ros_env
    wait_for_old_ros_nodes_to_leave || failures=1
  fi
  post_cleanup_checks || failures=1

  ((failures == 0)) || return 1
  log "force-clean completed: experiment processes, tmux sessions, ROS daemon, conflict ports and stale state are clear"
}

start_mode() {
  # Validate immutable artifacts before stopping anything.  Camera enumeration
  # and CAN activation are intentionally performed after exact legacy cleanup,
  # because a stale camera process can temporarily hold a RealSense device.
  static_preflight

  if [[ -f "$STATE_FILE" ]]; then
    warn "previous managed state exists; stopping it exactly before restart"
    stop_recorded_stack 0
  fi
  trap on_exit EXIT
  trap 'handle_signal INT' INT
  trap 'handle_signal TERM' TERM
  trap 'handle_signal HUP' HUP
  acquire_lock
  CLEANUP_ARMED=1

  LOG_DIR="${LOG_BASE}/$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$LOG_DIR"
  write_state_header
  log "state=${STATE_FILE} logs=${LOG_DIR} ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"

  cleanup_legacy_stack || die "legacy cleanup failed"
  wait_for_old_ros_nodes_to_leave || die "stale ROS graph did not clear"
  assert_port_free "$POLICY_PORT" policy || die "policy port conflict"
  assert_port_free "$ROS_SERVICE_PORT" ROS-only-service || die "ROS-only service port conflict"
  assert_port_free 9900 legacy-RobotArmService || die "legacy RobotArmService port conflict"
  check_camera_serials || die "required RealSense inventory is not ready"
  cleanup_legacy_stack || die "legacy cleanup failed after camera inventory"
  wait_for_old_ros_nodes_to_leave || die "stale ROS graph did not clear after camera inventory"
  check_no_can_receivers || die "CAN is already owned after legacy cleanup"

  activate_can
  check_no_can_receivers || die "CAN receiver appeared before piper_ros startup"
  start_components

  validate_running_stack || die "post-start validation failed"
  monitor_components
}

main() {
  parse_args "$@"
  # Never inherit a conflicting ROS domain from the caller.
  export ROS_DOMAIN_ID="$EXPECTED_ROS_DOMAIN_ID"
  export ROS_LOCALHOST_ONLY=0
  export PYTHONUNBUFFERED=1

  case "$MODE" in
    start) start_mode ;;
    preflight) preflight_mode ;;
    status) status_mode ;;
    validate) validate_mode ;;
    stop) stop_recorded_stack ;;
    force-clean) force_clean_mode ;;
    *) die "internal mode error: ${MODE}" ;;
  esac
}

main "$@"
