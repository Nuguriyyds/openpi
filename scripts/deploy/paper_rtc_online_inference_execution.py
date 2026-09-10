#!/usr/bin/env python3
"""
RTC 在线机械臂推理执行脚本
===========================

Paper-RTC client chunk-only mode:
  - client maintains and executes one local H-step action chunk at control_hz
  - after s_min executed steps, a background thread sends one full chunk request
  - the request carries A_cur[s:H] as the RTC prefix attention constraint
  - when the new chunk returns, client installs it at fixed delay index d
  - legacy per-step obs/tick WebSocket requests are disabled

使用方法：
  python3 paper_rtc_online_inference_execution.py \
    --host 127.0.0.1 --port 8001 \
    --prompt "pick up the cube" \
    --control_hz 30.0
"""

import sys
import json
import time
import threading
import argparse
import math
import shutil
import csv
import socket
import struct
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


import cv2
import numpy as np
from openpi_client import websocket_client_policy, msgpack_numpy

try:
    import yaml
except ImportError:
    yaml = None

_EXECUTION_ACTION_SPACE = "absolute_joint"

# Hardware I/O is intentionally not implemented in this process.
# start_robot_arm_service.py owns ROS2 observation collection and Piper SDK/CAN.


# =====================================================
#              观察缓冲
# =====================================================

class ObservationBuffer:
    """线程安全的观察缓冲"""

    def __init__(self):
        self._lock = threading.Lock()
        self.images = {
            'cam_top': None,
            'cam_left_wrist': None,
            'cam_right_wrist': None,
        }
        self.joint_left = np.zeros(6, dtype=np.float32)
        self.joint_right = np.zeros(6, dtype=np.float32)
        self.end_pose_left = np.zeros(6, dtype=np.float32)
        self.end_pose_right = np.zeros(6, dtype=np.float32)
        self.gripper_position = np.zeros(2, dtype=np.float32)
        self._last_update = 0.0
        self._joint_callback_count = 0   # diagnostic: total joint callbacks received

    def update_image(self, cam_name: str, image: np.ndarray):
        with self._lock:
            if cam_name in self.images:
                # HxWxC → CxHxW
                if image.ndim == 3 and image.shape[2] == 3:
                    self.images[cam_name] = np.transpose(image, (2, 0, 1))
                else:
                    self.images[cam_name] = image
                self._last_update = time.time()

    def update_joints(self, side: str, joints: np.ndarray):
        with self._lock:
            if side == 'left':
                self.joint_left = np.asarray(joints[:6], dtype=np.float32)
            else:
                self.joint_right = np.asarray(joints[:6], dtype=np.float32)

    def update_gripper_one_side(self, side: str, value: float):
        with self._lock:
            if side == 'left':
                self.gripper_position[0] = value
            else:
                self.gripper_position[1] = value

    def set_snapshot(self, snapshot: Dict[str, Any]):
        with self._lock:
            images = snapshot.get('images', {})
            self.images = {
                'cam_top': images.get('cam_top'),
                'cam_left_wrist': images.get('cam_left_wrist'),
                'cam_right_wrist': images.get('cam_right_wrist'),
            }
            self.joint_left = np.asarray(snapshot['joint_left'], dtype=np.float32).copy()
            self.joint_right = np.asarray(snapshot['joint_right'], dtype=np.float32).copy()
            self.end_pose_left = np.asarray(snapshot['end_pose_left'], dtype=np.float32).copy()
            self.end_pose_right = np.asarray(snapshot['end_pose_right'], dtype=np.float32).copy()
            self.gripper_position = np.asarray(
                snapshot['gripper_position'],
                dtype=np.float32,
            ).copy()
            self._last_update = time.time()

    def is_fresh(self, threshold_s: float = 1.0) -> bool:
        with self._lock:
            return (time.time() - self._last_update) < threshold_s

    def get_snapshot(self, copy_images: bool = False) -> Dict[str, Any]:
        with self._lock:
            if copy_images:
                images = {k: v.copy() if v is not None else v
                          for k, v in self.images.items()}
            else:
                images = dict(self.images)
            return {
                'images': images,
                'joint_left': self.joint_left.copy(),
                'joint_right': self.joint_right.copy(),
                'end_pose_left': self.end_pose_left.copy(),
                'end_pose_right': self.end_pose_right.copy(),
                'gripper_position': self.gripper_position.copy(),
            }


# =====================================================
#              RobotArmService TCP 后端
# =====================================================

def _tcp_send(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)


def _tcp_recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("robot service connection closed")
        buf += chunk
    return bytes(buf)


def _tcp_recv(sock: socket.socket) -> bytes:
    header = _tcp_recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    return _tcp_recv_exact(sock, length)


def robot_service_obs_to_snapshot(obs: Dict[str, Any]) -> Dict[str, Any]:
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.shape[0] < 12:
        raise ValueError(f"RobotArmService obs['state'] must have >=12 dims, got {state.shape}")
    end_state = np.asarray(obs.get("end_effector_state"), dtype=np.float32)
    if end_state.shape != (12,):
        raise ValueError(
            "RobotArmService obs['end_effector_state'] must have 12 dims, "
            f"got {end_state.shape}"
        )
    return {
        "images": dict(obs["images"]),
        "joint_left": state[:6].copy(),
        "joint_right": state[6:12].copy(),
        "end_pose_left": end_state[:6].copy(),
        "end_pose_right": end_state[6:12].copy(),
        "gripper_position": np.asarray(obs["gripper_position"], dtype=np.float32).copy(),
    }


_DEFAULT_INIT_ACTION = np.array(
    [
        -0.47566299, 0.12493393, -0.49851463, 0.09187755, 0.84711553, -0.11246147,
        0.000,
        0.05784430, 0.45746890, -0.47620376, 0.25457774, 0.83889940, -0.19685554,
        0.060,
    ],
    dtype=np.float32,
)


def snapshot_to_robot_service_action(snapshot: Dict[str, Any]) -> np.ndarray:
    if _EXECUTION_ACTION_SPACE == "absolute_end_effector":
        left = np.asarray(snapshot["end_pose_left"], dtype=np.float32)
        right = np.asarray(snapshot["end_pose_right"], dtype=np.float32)
    else:
        left = np.asarray(snapshot["joint_left"], dtype=np.float32)
        right = np.asarray(snapshot["joint_right"], dtype=np.float32)
    return np.concatenate(
        [
            left,
            np.array([float(snapshot["gripper_position"][0])], dtype=np.float32),
            right,
            np.array([float(snapshot["gripper_position"][1])], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def snapshot_to_policy_action(snapshot: Dict[str, Any]) -> np.ndarray:
    gripper = np.asarray(snapshot["gripper_position"], dtype=np.float32)
    if _EXECUTION_ACTION_SPACE == "absolute_end_effector":
        left = np.asarray(snapshot["end_pose_left"], dtype=np.float32)
        right = np.asarray(snapshot["end_pose_right"], dtype=np.float32)
    else:
        left = np.asarray(snapshot["joint_left"], dtype=np.float32)
        right = np.asarray(snapshot["joint_right"], dtype=np.float32)
    return np.concatenate(
        [
            left,
            np.array([float(gripper[0]) / 0.105], dtype=np.float32),
            right,
            np.array([float(gripper[1]) / 0.105], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


class RobotArmServiceClient:
    """TCP client for start_robot_arm_service.py, matching run_inference's hardware path."""

    def __init__(
        self,
        endpoint: str,
        *,
        prompt: str,
        recv_timeout_ms: int = 30000,
    ) -> None:
        self.endpoint = endpoint
        self.prompt = prompt
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.action_space = "absolute_joint"
        self._sock: Optional[socket.socket] = None
        self._packer = msgpack_numpy.Packer()

    def connect(self) -> None:
        addr = self.endpoint.replace("tcp://", "")
        host, port_str = addr.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.recv_timeout_ms / 1000)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((host, int(port_str)))
        self._sock = sock
        reply = self._request({"cmd": "ping"})
        if not reply.get("pong"):
            raise RuntimeError(f"unexpected RobotArmService ping reply: {reply}")
        print(f"  [RobotService] connected to {self.endpoint}")

    def reset(self) -> Dict[str, Any]:
        reply = self._request({"cmd": "reset"})
        obs = dict(reply["obs"])
        if self.prompt:
            obs["prompt"] = self.prompt
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        reply = self._request(
            {
                "cmd": "step",
                "action": np.asarray(action, dtype=np.float32),
                "action_space": self.action_space,
            }
        )
        obs = dict(reply["obs"])
        if self.prompt:
            obs["prompt"] = self.prompt
        return obs, bool(reply.get("done", False)), dict(reply.get("info", {}))

    def stop(self) -> None:
        if self._sock is None:
            return
        try:
            self._request({"cmd": "stop"})
        except Exception as exc:
            print(f"  [RobotService] stop failed: {exc}")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._sock is None:
            raise RuntimeError("RobotArmServiceClient is not connected")
        _tcp_send(self._sock, self._packer.pack(msg))
        raw = _tcp_recv(self._sock)
        reply = msgpack_numpy.unpackb(raw)
        if "error" in reply:
            raise RuntimeError(f"RobotArmService error: {reply['error']}")
        return reply


# =====================================================
#              辅助函数
# =====================================================

def make_agilex_observation(snapshot: Dict[str, Any], prompt: str) -> dict:
    if _EXECUTION_ACTION_SPACE == "absolute_end_effector":
        state_arm = np.concatenate(
            [snapshot['end_pose_left'], snapshot['end_pose_right']], axis=0
        )
    else:
        state_arm = np.concatenate(
            [snapshot['joint_left'], snapshot['joint_right']], axis=0
        )
    images = {}
    for cam in ('cam_top', 'cam_left_wrist', 'cam_right_wrist'):
        images[cam] = snapshot['images'][cam]
    return {
        "state": state_arm,
        "gripper_position": snapshot['gripper_position'],
        "images": images,
        "prompt": prompt,
    }


def obs_payload_kib(obs: dict) -> float:
    images = obs.get("images") if isinstance(obs, dict) else None
    if not isinstance(images, dict):
        return 0.0
    total = 0
    for img in images.values():
        if isinstance(img, np.ndarray):
            total += img.nbytes
    return total / 1024


def make_paper_rtc_client_chunk_request(
    snapshot: Dict[str, Any],
    prompt: str,
    *,
    client_step: int,
    local_chunk_id: int,
    local_chunk_index: int,
    local_remaining: int,
    prev_leftover_robot: Optional[np.ndarray],
    fixed_delay_steps: int,
    prefix_attention_horizon: int,
) -> dict:
    obs = make_agilex_observation(snapshot, prompt)
    obs["_paper_rtc_client_chunk_request"] = True
    obs["_paper_rtc_client_step"] = int(client_step)
    obs["_paper_rtc_local_chunk_id"] = int(local_chunk_id)
    obs["_paper_rtc_local_chunk_index"] = int(local_chunk_index)
    obs["_paper_rtc_local_remaining"] = int(local_remaining)
    obs["_paper_rtc_inference_delay_steps"] = int(fixed_delay_steps)
    obs["_paper_rtc_prefix_attention_horizon"] = int(prefix_attention_horizon)
    if prev_leftover_robot is not None:
        obs["_paper_rtc_prev_leftover_robot"] = np.asarray(
            prev_leftover_robot,
            dtype=np.float32,
        ).copy()
    return obs


def compute_client_chunk_delay_steps(
    *,
    delay_mode: str,
    fixed_delay_steps: int,
    elapsed_s: Optional[float],
    control_hz: float,
    max_delay_steps: int,
    prefix_attention_horizon: int,
) -> tuple[int, str]:
    # Convert measured request latency into the conditioned action prefix length.
    # Ceil is intentional here: any partial control period means the robot can
    # already have missed the next action slot by the time the new chunk arrives.
    mode = str(delay_mode or "fixed").strip().lower()
    max_delay_steps = max(0, int(max_delay_steps))
    prefix_attention_horizon = max(0, int(prefix_attention_horizon))
    cap = min(max_delay_steps, prefix_attention_horizon)
    fixed_delay_steps = max(0, min(int(fixed_delay_steps), cap))
    if mode == "fixed":
        return fixed_delay_steps, "fixed"
    if mode == "realtime_ceil":
        if elapsed_s is None:
            # The first request has no previous latency sample, so use the
            # configured fixed delay as a conservative warmup fallback.
            return fixed_delay_steps, "realtime_ceil_warmup"
        estimated = int(math.ceil(max(0.0, float(elapsed_s)) * float(control_hz)))
        return max(0, min(estimated, cap)), "realtime_ceil"
    raise ValueError(f"unsupported client_chunk_delay_mode: {delay_mode!r}")


class AsyncClientChunkPlanner:
    """Requests full Paper-RTC chunks while the control loop executes locally."""

    def __init__(
        self,
        *,
        client: Any,
        obs_buffer: ObservationBuffer,
        prompt: str,
        fixed_delay_steps: int,
        delay_mode: str = "fixed",
        control_hz: float = 30.0,
        max_delay_steps: int = 0,
    ) -> None:
        self.client = client
        self.obs_buffer = obs_buffer
        self.prompt = prompt
        self.fixed_delay_steps = max(0, int(fixed_delay_steps))
        self.delay_mode = str(delay_mode or "fixed").strip().lower()
        if self.delay_mode not in {"fixed", "realtime_ceil"}:
            raise ValueError(f"unsupported client_chunk_delay_mode: {delay_mode!r}")
        self.control_hz = float(control_hz)
        if self.control_hz <= 0.0:
            raise ValueError(f"control_hz must be positive, got {control_hz!r}")
        self.max_delay_steps = max(0, int(max_delay_steps))
        # Store the last complete request latency for realtime_ceil mode. It is
        # updated only after a chunk returns, then used by the next request.
        self._last_request_elapsed_s: Optional[float] = None
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._request: Optional[dict[str, Any]] = None
        self._result: Optional[dict[str, Any]] = None
        self._inflight = False
        self._seq = 0
        self._last_error = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="paper-rtc-client-chunk", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def pending(self) -> bool:
        with self._lock:
            return self._request is not None or self._inflight or self._result is not None

    def ready(self) -> bool:
        with self._lock:
            return self._result is not None

    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def request(
        self,
        *,
        client_step: int,
        local_chunk_id: int,
        local_chunk_index: int,
        prev_leftover_robot: Optional[np.ndarray],
        reason: str,
    ) -> bool:
        prev_robot = None
        if prev_leftover_robot is not None:
            prev_robot = np.asarray(prev_leftover_robot, dtype=np.float32).copy()
        # The server can only condition on leftover actions that still exist in
        # the previous chunk, so delay is capped by prefix_attention_horizon.
        prefix_attention_horizon = 0 if prev_robot is None else int(len(prev_robot))
        delay_steps, delay_source = compute_client_chunk_delay_steps(
            delay_mode=self.delay_mode,
            fixed_delay_steps=self.fixed_delay_steps,
            elapsed_s=self._last_request_elapsed_s,
            control_hz=self.control_hz,
            max_delay_steps=self.max_delay_steps,
            prefix_attention_horizon=prefix_attention_horizon,
        )
        req = {
            "client_step": int(client_step),
            "local_chunk_id": int(local_chunk_id),
            "local_chunk_index": int(local_chunk_index),
            "local_remaining": 0 if prev_robot is None else int(len(prev_robot)),
            "prev_leftover_robot": prev_robot,
            "prefix_attention_horizon": prefix_attention_horizon,
            "fixed_delay_steps": delay_steps,
            "delay_mode": self.delay_mode,
            "delay_source": delay_source,
            "delay_elapsed_s": self._last_request_elapsed_s,
            "reason": str(reason),
            "request_monotonic_s": time.monotonic(),
        }
        with self._cv:
            if self._request is not None or self._inflight:
                return False
            self._request = req
            self._cv.notify_all()
            return True

    def wait_first(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                if self._result is not None:
                    return True
                last_error = self._last_error
            if last_error:
                print(f"[CHUNK] waiting for first chunk; last error: {last_error}")
            time.sleep(0.02)
        with self._lock:
            return self._result is not None

    def pop_ready(self) -> Optional[dict[str, Any]]:
        with self._lock:
            result = self._result
            self._result = None
            return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                self._cv.wait_for(lambda: self._stop.is_set() or self._request is not None)
                if self._stop.is_set():
                    return
                req = self._request
                self._request = None
                self._inflight = True
            if req is None:
                continue

            t0 = time.monotonic()
            try:
                snapshot = self.obs_buffer.get_snapshot(copy_images=False)
                t_snap = time.monotonic()
                obs = make_paper_rtc_client_chunk_request(
                    snapshot,
                    self.prompt,
                    client_step=req["client_step"],
                    local_chunk_id=req["local_chunk_id"],
                    local_chunk_index=req["local_chunk_index"],
                    local_remaining=req["local_remaining"],
                    prev_leftover_robot=req["prev_leftover_robot"],
                    fixed_delay_steps=req["fixed_delay_steps"],
                    prefix_attention_horizon=req["prefix_attention_horizon"],
                )
                payload_kib = obs_payload_kib(obs)
                t_obs = time.monotonic()
                out = self.client.infer(obs)
                t_infer = time.monotonic()
                # End-to-end client request time: snapshot copy, observation
                # packing, websocket transfer, and server inference.
                request_elapsed_s = t_infer - t0
                robot_actions = np.asarray(out.get("actions"), dtype=np.float32)
                if robot_actions.ndim != 2:
                    raise RuntimeError(f"chunk response actions must be rank-2, got {robot_actions.shape}")
                with self._cv:
                    # A no-motion warmup runs before the first planner request,
                    # so the initial request latency is a clean runtime sample.
                    self._last_request_elapsed_s = request_elapsed_s
                    self._result = {
                        "robot_actions": robot_actions.copy(),
                        "server_timing": dict(out.get("server_timing", {})),
                        "request": dict(req),
                        "client_timing": {
                            "chunk_async_seq": self._seq,
                            "chunk_request_reason": req["reason"],
                            "chunk_request_step": req["client_step"],
                            "chunk_request_local_idx": req["local_chunk_index"],
                            "chunk_request_remaining": req["local_remaining"],
                            "chunk_request_fixed_delay_steps": req["fixed_delay_steps"],
                            "chunk_request_delay_steps": req["fixed_delay_steps"],
                            "chunk_request_delay_mode": req["delay_mode"],
                            "chunk_request_delay_source": req["delay_source"],
                            "chunk_request_delay_elapsed_s": req["delay_elapsed_s"],
                            "chunk_request_prefix_attention_horizon": req["prefix_attention_horizon"],
                            "chunk_payload_kib": payload_kib,
                            "chunk_snap_ms": (t_snap - t0) * 1000,
                            "chunk_obs_ms": (t_obs - t_snap) * 1000,
                            "chunk_ws_ms": (t_infer - t_obs) * 1000,
                            "chunk_total_ms": (t_infer - t0) * 1000,
                            "chunk_response_time_s": t_infer,
                        },
                    }
                    self._last_error = ""
                    self._inflight = False
                    self._seq += 1
                    self._cv.notify_all()
            except Exception as exc:
                with self._cv:
                    self._last_error = str(exc)
                    self._inflight = False
                    self._cv.notify_all()
                print(f"[CHUNK] request failed: {exc}")
                time.sleep(0.05)


def fdict_ms(data: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def summarize_cmd_diag(left: dict, right: dict) -> dict:
    left = left or {}
    right = right or {}
    return {
        "heartbeat": bool(left.get("heartbeat")) or bool(right.get("heartbeat")),
        "motion_ctrl_ms": fdict_ms(left, "motion_ctrl_ms") + fdict_ms(right, "motion_ctrl_ms"),
        "joint_ms": fdict_ms(left, "joint_ms") + fdict_ms(right, "joint_ms"),
        "gripper_ms": fdict_ms(left, "gripper_ms") + fdict_ms(right, "gripper_ms"),
        "left_total_ms": fdict_ms(left, "total_ms"),
        "right_total_ms": fdict_ms(right, "total_ms"),
    }


def classify_stutter(
    *,
    obs_kind: str,
    period_ms: float,
    loop_ms: float,
    ms_snap: float,
    ms_obs: float,
    ms_ws: float,
    ms_action: float,
    ms_metrics: float,
    ms_cmd: float,
    ms_save: float,
    ms_log: float,
    prev_log_ms: float,
    server_timing: dict,
    cmd_diag: dict,
    threshold_ms: float,
) -> str:
    server_get_ms = fdict_ms(server_timing, "get_action_ms")
    server_unpack_ms = fdict_ms(server_timing, "server_unpack_ms")
    server_prepare_ms = fdict_ms(server_timing, "server_prepare_ms")
    ws_extra_ms = max(0.0, ms_ws - server_prepare_ms)

    if period_ms > threshold_ms and loop_ms <= threshold_ms:
        if prev_log_ms > 1.0:
            return "previous-step logging/printing delayed next loop start"
        return "previous-step overrun or OS scheduling jitter"
    if bool(server_timing.get("queue_underrun")):
        return "server action queue underrun; held last action"
    if ms_action >= max(ms_ws, ms_obs, ms_snap, ms_metrics, ms_cmd, ms_save, 1.0):
        return "client action postprocess path"
    if ms_metrics >= max(ms_ws, ms_obs, ms_snap, ms_action, ms_cmd, ms_save, 1.0):
        return "client smoothness metrics path"
    if ms_cmd >= max(ms_ws, ms_obs, ms_snap, ms_action, ms_metrics, ms_save, 1.0):
        return "RobotArmService TCP step path"
    if ms_ws >= max(ms_cmd, ms_obs, ms_snap, ms_action, ms_metrics, ms_save, 1.0):
        if server_get_ms > threshold_ms * 0.5:
            return "server get_action blocked or waited for action queue"
        if server_unpack_ms > 5.0:
            return "server unpacked a large observation payload"
        if ws_extra_ms > threshold_ms * 0.5 and obs_kind == "obs":
            return "client serialization/WebSocket transfer of full observation"
        if bool(server_timing.get("inference_active")):
            return "WebSocket round trip while server inference thread was active"
        return "WebSocket/client scheduling round trip"
    if ms_obs >= max(ms_cmd, ms_ws, ms_snap, ms_action, ms_metrics, ms_save, 1.0):
        return "client observation copy/build path"
    if ms_snap >= max(ms_cmd, ms_ws, ms_obs, ms_action, ms_metrics, ms_save, 1.0):
        return "client observation snapshot lock/copy path"
    if ms_save >= max(ms_cmd, ms_ws, ms_obs, ms_snap, ms_action, ms_metrics, 1.0):
        return "debug image save path"
    if ms_log > 1.0:
        return "logging/printing overhead"
    return "mixed small costs or OS scheduling jitter"


class TimingBottleneckMonitor:
    """Lightweight scalar timing records for control-frequency diagnosis."""

    COMPONENT_KEYS = [
        "snap_ms",
        "obs_ms",
        "ws_ms",
        "action_ms",
        "metrics_ms",
        "cmd_ms",
        "save_ms",
        "log_ms",
    ]

    def __init__(
        self,
        *,
        control_hz: float,
        enabled: bool = True,
        output_dir: str = "/tmp/paper_rtc_smoothness",
        history_limit: int = 9000,
        window_steps: int = 90,
        print_interval_steps: int = 90,
        save_csv: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.control_hz = float(control_hz)
        self.target_dt_ms = 1000.0 / max(self.control_hz, 1e-6)
        self.output_dir = Path(output_dir)
        self.history_limit = max(10, int(history_limit))
        self.window_steps = max(3, int(window_steps))
        self.print_interval_steps = max(1, int(print_interval_steps))
        self.save_csv_enabled = bool(save_csv)
        self.records: list[dict[str, Any]] = []

    def _trim(self) -> None:
        if len(self.records) > self.history_limit:
            del self.records[: len(self.records) - self.history_limit]

    @staticmethod
    def _percentile(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        return float(np.percentile(np.asarray(vals, dtype=np.float64), q))

    def add(
        self,
        *,
        step: int,
        obs_kind: str,
        chunk_step: int,
        period_ms: float,
        loop_ms: float,
        total_ms: float,
        sleep_request_ms: float,
        snap_ms: float,
        obs_ms: float,
        ws_ms: float,
        action_ms: float,
        metrics_ms: float,
        cmd_ms: float,
        save_ms: float,
        log_ms: float,
        payload_kib: float,
        server_timing: dict,
        cmd_diag: dict,
        reason: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        components = {
            "snap_ms": float(snap_ms),
            "obs_ms": float(obs_ms),
            "ws_ms": float(ws_ms),
            "action_ms": float(action_ms),
            "metrics_ms": float(metrics_ms),
            "cmd_ms": float(cmd_ms),
            "save_ms": float(save_ms),
            "log_ms": float(log_ms),
        }
        dominant = max(self.COMPONENT_KEYS, key=lambda key: components.get(key, 0.0))
        server_prepare_ms = fdict_ms(server_timing, "server_prepare_ms")
        rec = {
            "step": int(step),
            "obs_kind": str(obs_kind),
            "chunk_step": int(chunk_step) if isinstance(chunk_step, (int, np.integer)) else -1,
            "target_dt_ms": self.target_dt_ms,
            "period_ms": float(period_ms),
            "loop_ms": float(loop_ms),
            "total_ms": float(total_ms),
            "sleep_request_ms": float(sleep_request_ms),
            "overrun_ms": max(0.0, float(total_ms) - self.target_dt_ms),
            "period_slip_ms": max(0.0, float(period_ms) - self.target_dt_ms),
            "payload_kib": float(payload_kib),
            **components,
            "server_get_ms": fdict_ms(server_timing, "get_action_ms"),
            "server_unpack_ms": fdict_ms(server_timing, "server_unpack_ms"),
            "server_prepare_ms": server_prepare_ms,
            "server_prev_total_ms": fdict_ms(server_timing, "prev_total_ms"),
            "ws_extra_ms": max(0.0, float(ws_ms) - server_prepare_ms),
            "queue_remaining": server_timing.get("queue_remaining", -1),
            "queue_before_request": server_timing.get("queue_before_request", -1),
            "queue_before_get": server_timing.get("queue_before_get", -1),
            "queue_after_get": server_timing.get("queue_after_get", -1),
            "inference_active": int(bool(server_timing.get("inference_active", False))),
            "queue_underrun": int(bool(server_timing.get("queue_underrun", False))),
            "trigger_count": server_timing.get("trigger_count", 0),
            "delay_est_steps": server_timing.get("delay_est_steps", -1),
            "real_delay_steps": server_timing.get("real_delay_steps", -1),
            "skipped_steps": server_timing.get("skipped_steps", -1),
            "boundary_blend_len": server_timing.get("boundary_blend_len", 0),
            "boundary_jump_pre": fdict_ms(server_timing, "boundary_jump_pre"),
            "boundary_jump_post": fdict_ms(server_timing, "boundary_jump_post"),
            "cmd_left_total_ms": fdict_ms(cmd_diag, "left_total_ms"),
            "cmd_right_total_ms": fdict_ms(cmd_diag, "right_total_ms"),
            "cmd_joint_ms": fdict_ms(cmd_diag, "joint_ms"),
            "cmd_gripper_ms": fdict_ms(cmd_diag, "gripper_ms"),
            "cmd_motion_ctrl_ms": fdict_ms(cmd_diag, "motion_ctrl_ms"),
            "cmd_heartbeat": int(bool(cmd_diag.get("heartbeat"))),
            "reason": str(reason),
            "dominant_client": dominant.replace("_ms", ""),
            "dominant_client_ms": float(components[dominant]),
        }
        self.records.append(rec)
        self._trim()
        return rec

    def should_print(self, step: int) -> bool:
        return self.enabled and self.print_interval_steps > 0 and step % self.print_interval_steps == 0

    def rolling_summary(self) -> dict[str, float]:
        if not self.records:
            return {}
        window = self.records[-self.window_steps :]
        period_vals = [float(r["period_ms"]) for r in window if float(r["period_ms"]) > 0.0]
        total_vals = [float(r["total_ms"]) for r in window]
        summary = {
            "actual_hz": 1000.0 / (sum(period_vals) / len(period_vals)) if period_vals else 0.0,
            "period_mean": sum(period_vals) / len(period_vals) if period_vals else 0.0,
            "period_p95": self._percentile(period_vals, 95),
            "period_max": max(period_vals) if period_vals else 0.0,
            "total_mean": sum(total_vals) / len(total_vals) if total_vals else 0.0,
            "total_p95": self._percentile(total_vals, 95),
            "overrun_rate": 100.0
            * sum(float(r["total_ms"]) > self.target_dt_ms for r in window)
            / max(1, len(window)),
        }
        for key in self.COMPONENT_KEYS:
            vals = [float(r.get(key, 0.0)) for r in window]
            summary[f"{key}_mean"] = sum(vals) / len(vals) if vals else 0.0
            summary[f"{key}_p95"] = self._percentile(vals, 95)
        counts: dict[str, int] = {}
        for rec in window:
            name = str(rec.get("dominant_client", "?"))
            counts[name] = counts.get(name, 0) + 1
        if counts:
            summary["dominant_count"] = max(counts.values())
            summary["dominant_name"] = max(counts, key=counts.get)  # type: ignore[assignment]
        return summary

    def format_online(self) -> str:
        if not self.records:
            return ""
        roll = self.rolling_summary()
        dom = roll.get("dominant_name", "?")
        dom_count = float(roll.get("dominant_count", 0.0))
        dom_pct = 100.0 * dom_count / max(1, min(len(self.records), self.window_steps))
        return (
            f"  [TIMING] step={self.records[-1]['step']} "
            f"hz={roll.get('actual_hz', 0.0):.2f}/{self.control_hz:.1f} "
            f"period mean/p95/max={roll.get('period_mean', 0.0):.1f}/"
            f"{roll.get('period_p95', 0.0):.1f}/{roll.get('period_max', 0.0):.1f}ms "
            f"total mean/p95={roll.get('total_mean', 0.0):.1f}/"
            f"{roll.get('total_p95', 0.0):.1f}ms overrun={roll.get('overrun_rate', 0.0):.0f}% "
            f"ws p95={roll.get('ws_ms_p95', 0.0):.1f}ms "
            f"cmd p95={roll.get('cmd_ms_p95', 0.0):.1f}ms "
            f"log p95={roll.get('log_ms_p95', 0.0):.1f}ms "
            f"dom={dom}({dom_pct:.0f}%)"
        )

    def print_final_summary(self) -> None:
        if not self.enabled or not self.records:
            return
        period_vals = [float(r["period_ms"]) for r in self.records if float(r["period_ms"]) > 0.0]
        total_vals = [float(r["total_ms"]) for r in self.records]
        print("\n[TIMING SUMMARY]")
        print(f"  samples={len(self.records)} target={self.control_hz:.1f}Hz ({self.target_dt_ms:.2f}ms)")
        if period_vals:
            actual_hz = 1000.0 / (sum(period_vals) / len(period_vals))
            print(
                f"  actual_hz={actual_hz:.2f} "
                f"period mean/p95/p99/max="
                f"{np.mean(period_vals):.1f}/"
                f"{np.percentile(period_vals, 95):.1f}/"
                f"{np.percentile(period_vals, 99):.1f}/"
                f"{max(period_vals):.1f}ms"
            )
        if total_vals:
            overrun = 100.0 * sum(v > self.target_dt_ms for v in total_vals) / len(total_vals)
            print(
                f"  total mean/p95/max="
                f"{np.mean(total_vals):.1f}/"
                f"{np.percentile(total_vals, 95):.1f}/"
                f"{max(total_vals):.1f}ms overrun_rate={overrun:.1f}%"
            )
        for key in self.COMPONENT_KEYS:
            vals = [float(r.get(key, 0.0)) for r in self.records]
            if vals:
                print(
                    f"  {key:<10} mean/p95/max="
                    f"{np.mean(vals):.1f}/{np.percentile(vals, 95):.1f}/{max(vals):.1f}ms"
                )
        counts: dict[str, int] = {}
        for rec in self.records:
            name = str(rec.get("dominant_client", "?"))
            counts[name] = counts.get(name, 0) + 1
        if counts:
            top = max(counts, key=counts.get)
            print(f"  dominant_client={top} ({100.0 * counts[top] / len(self.records):.1f}% of samples)")

    def save_csv(self) -> None:
        if not self.enabled or not self.save_csv_enabled or not self.records:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "timing_bottleneck_metrics.csv"
        keys = list(self.records[-1].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in self.records:
                writer.writerow({k: rec.get(k) for k in keys})
        print(f"[TIMING] saved timing bottleneck CSV: {path}")


def smooth_action(last: Optional[np.ndarray], cur: np.ndarray,
                  alpha: float = 0.6,
                  max_delta: Optional[float] = None) -> np.ndarray:
    """EMA smooth with optional per-step displacement cap.

    Server-side hard inpainting + seam blend reduce chunk-boundary discontinuity.
    max_delta is a hardware safety cap for Piper position control: the CAN bus
    sends absolute joint targets, so any residual jump arrives as a single
    instantaneous command.  The paper's experimental arms use velocity/impedance
    control that naturally absorbs such jumps; Piper does not.
    """
    if last is None:
        return cur
    blended = alpha * cur + (1.0 - alpha) * last
    if max_delta is not None:
        delta = blended - last
        blended = last + np.clip(delta, -max_delta, max_delta)
    return blended


def smooth_end_pose(
    last: Optional[np.ndarray],
    cur: np.ndarray,
    *,
    max_position_delta_m: float,
    max_rotation_delta_rad: float,
    position_filter_alpha: float,
    rotation_filter_alpha: float,
) -> np.ndarray:
    """Low-pass XYZ and SLERP Euler-XYZ orientations with per-step caps."""
    cur = np.asarray(cur, dtype=np.float32)
    if last is None:
        return cur
    last = np.asarray(last, dtype=np.float32)
    result = cur.copy()
    position_delta = position_filter_alpha * (cur[:3] - last[:3])
    result[:3] = last[:3] + np.clip(
        position_delta, -max_position_delta_m, max_position_delta_m
    )
    q_last = _euler_xyz_to_quaternion(last[3:6])
    q_cur = _euler_xyz_to_quaternion(cur[3:6])
    dot = float(np.dot(q_last, q_cur))
    if dot < 0.0:
        q_cur = -q_cur
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * math.acos(dot)
    fraction = rotation_filter_alpha
    if angle > 1e-8:
        fraction = min(fraction, max_rotation_delta_rad / angle)
    q_filtered = _quaternion_slerp(q_last, q_cur, float(np.clip(fraction, 0.0, 1.0)))
    result[3:6] = _quaternion_to_euler_xyz(q_filtered)
    return result


def _euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in euler)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _quaternion_slerp(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(start, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + fraction * (target - start)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - fraction) * theta) / sin_theta * start
        + math.sin(fraction * theta) / sin_theta * target
    )


def _quaternion_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion / np.linalg.norm(quaternion))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _transform_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    roll, pitch, yaw = rpy
    out[:3, :3] = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return out


def _transform_rz(angle: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rot_z(float(angle))
    return out


_PIPER_JOINT_ORIGINS = [
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),
]
_PIPER_TCP_OFFSET = np.array([0.0, 0.0, 0.1358, 1.0], dtype=np.float64)
_PIPER_ORIGIN_TF = [_transform_xyz_rpy(xyz, rpy) for xyz, rpy in _PIPER_JOINT_ORIGINS]


def piper_local_tcp_xyz(joints: np.ndarray) -> np.ndarray:
    """Approximate Piper TCP xyz in each arm's local base frame.

    Uses the local Piper URDF chain through joint6 plus the gripper center
    offset. This is for trajectory diagnostics/plots, not control.
    """
    q = np.asarray(joints, dtype=np.float64).reshape(-1)
    if q.size < 6 or not np.all(np.isfinite(q[:6])):
        return np.full(3, np.nan, dtype=np.float64)
    tf = np.eye(4, dtype=np.float64)
    for origin_tf, angle in zip(_PIPER_ORIGIN_TF, q[:6]):
        tf = tf @ origin_tf @ _transform_rz(float(angle))
    return (tf @ _PIPER_TCP_OFFSET)[:3]


def piper_batch_local_tcp_xyz(actions: np.ndarray, *, right: bool) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float64)
    offset = 7 if right else 0
    return np.stack([piper_local_tcp_xyz(row[offset : offset + 6]) for row in arr], axis=0)


class SmoothnessMonitor:
    """Collects low-overhead trajectory smoothness metrics and saves plots."""

    JOINT_IDX = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    GRIP_IDX = np.array([6, 13], dtype=np.int64)
    JOINT_NAMES = [
        "L1", "L2", "L3", "L4", "L5", "L6",
        "R1", "R2", "R3", "R4", "R5", "R6",
    ]
    ACTION_NAMES = [
        "L1", "L2", "L3", "L4", "L5", "L6", "GL",
        "R1", "R2", "R3", "R4", "R5", "R6", "GR",
    ]

    def __init__(
        self,
        *,
        control_hz: float,
        joint_delta_cap_rad: float = 0.15,
        enabled: bool = True,
        plot_dir: str = "/tmp/paper_rtc_smoothness",
        history_limit: int = 9000,
        window_steps: int = 60,
        print_interval_steps: int = 30,
        plot_interval_s: float = 0.0,
        save_csv: bool = True,
        save_plots: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.control_hz = float(control_hz)
        self.joint_delta_cap_rad = float(joint_delta_cap_rad)
        if not math.isfinite(self.joint_delta_cap_rad) or self.joint_delta_cap_rad <= 0.0:
            raise ValueError("joint_delta_cap_rad must be finite and > 0")
        self.dt_default = 1.0 / max(self.control_hz, 1e-6)
        self.plot_dir = Path(plot_dir)
        self.history_limit = max(10, int(history_limit))
        self.window_steps = max(3, int(window_steps))
        self.print_interval_steps = max(1, int(print_interval_steps))
        self.plot_interval_s = max(0.0, float(plot_interval_s))
        self.save_csv_enabled = bool(save_csv)
        self.save_plots_enabled = bool(save_plots)

        self.records: list[dict[str, Any]] = []
        self._last_raw: np.ndarray | None = None
        self._last_cmd: np.ndarray | None = None
        self._last_vel: np.ndarray | None = None
        self._last_acc: np.ndarray | None = None
        self._last_t: float | None = None
        self._last_plot_s = 0.0
        self._accel_energy = 0.0
        self._jerk_energy = 0.0
        self._cmd_path_len = 0.0
        self._raw_path_len = 0.0

    @staticmethod
    def _l2(x: np.ndarray) -> float:
        return float(np.linalg.norm(x))

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    @staticmethod
    def _max_abs(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.max(np.abs(x)))

    def _trim(self) -> None:
        if len(self.records) > self.history_limit:
            drop = len(self.records) - self.history_limit
            del self.records[:drop]

    def add(
        self,
        *,
        step: int,
        now_s: float,
        period_ms: float,
        raw_action: np.ndarray,
        cmd_action: np.ndarray,
        observed_action: np.ndarray,
        chunk_step: int,
        obs_kind: str,
        server_timing: dict,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}

        raw = np.asarray(raw_action, dtype=np.float64)
        cmd = np.asarray(cmd_action, dtype=np.float64)
        obs = np.asarray(observed_action, dtype=np.float64)
        if self._last_t is None:
            dt_s = self.dt_default
        else:
            dt_s = max(1e-4, float(now_s - self._last_t))
        self._last_t = float(now_s)

        if self._last_raw is None:
            raw_delta = np.zeros_like(raw)
            cmd_delta = np.zeros_like(cmd)
        else:
            raw_delta = raw - self._last_raw
            cmd_delta = cmd - self._last_cmd

        vel = cmd_delta / dt_s
        if self._last_vel is None:
            acc = np.zeros_like(vel)
        else:
            acc = (vel - self._last_vel) / dt_s
        if self._last_acc is None:
            jerk = np.zeros_like(acc)
        else:
            jerk = (acc - self._last_acc) / dt_s

        joint_raw_delta = raw_delta[self.JOINT_IDX]
        joint_cmd_delta = cmd_delta[self.JOINT_IDX]
        grip_raw_delta = raw_delta[self.GRIP_IDX]
        grip_cmd_delta = cmd_delta[self.GRIP_IDX]
        joint_vel = vel[self.JOINT_IDX]
        joint_acc = acc[self.JOINT_IDX]
        joint_jerk = jerk[self.JOINT_IDX]
        grip_vel = vel[self.GRIP_IDX]
        grip_acc = acc[self.GRIP_IDX]
        grip_jerk = jerk[self.GRIP_IDX]
        track = cmd - obs
        joint_track = track[self.JOINT_IDX]
        grip_track = track[self.GRIP_IDX]
        smoothing_correction = raw - cmd
        joint_smoothing_correction = smoothing_correction[self.JOINT_IDX]

        raw_delta_l2 = self._l2(joint_raw_delta)
        cmd_delta_l2 = self._l2(joint_cmd_delta)
        smooth_ratio = 0.0 if raw_delta_l2 < 1e-9 else cmd_delta_l2 / raw_delta_l2
        joint_cap_hits = int(
            np.count_nonzero(np.abs(joint_cmd_delta) >= 0.99 * self.joint_delta_cap_rad)
        )
        grip_cap_hits = int(np.count_nonzero(np.abs(grip_cmd_delta) >= 0.049))

        self._raw_path_len += raw_delta_l2
        self._cmd_path_len += cmd_delta_l2
        self._accel_energy += float(np.sum(np.square(joint_acc)) * dt_s)
        self._jerk_energy += float(np.sum(np.square(joint_jerk)) * dt_s)

        rec = {
            "step": int(step),
            "t_s": float(now_s),
            "period_ms": float(period_ms),
            "dt_s": float(dt_s),
            "chunk_step": int(chunk_step) if isinstance(chunk_step, (int, np.integer)) else -1,
            "obs_kind": str(obs_kind),
            "queue_remaining": server_timing.get("queue_remaining", -1),
            "inference_active": int(bool(server_timing.get("inference_active", False))),
            "trigger_count": server_timing.get("trigger_count", 0),
            "delay_est_steps": server_timing.get("delay_est_steps", -1),
            "real_delay_steps": server_timing.get("real_delay_steps", -1),
            "skipped_steps": server_timing.get("skipped_steps", -1),
            "boundary_blend_len": server_timing.get("boundary_blend_len", 0),
            "boundary_jump_pre": fdict_ms(server_timing, "boundary_jump_pre"),
            "boundary_jump_post": fdict_ms(server_timing, "boundary_jump_post"),
            "raw_delta_joint_l2": raw_delta_l2,
            "raw_delta_joint_max": self._max_abs(joint_raw_delta),
            "cmd_delta_joint_l2": cmd_delta_l2,
            "cmd_delta_joint_max": self._max_abs(joint_cmd_delta),
            "cmd_delta_left_l2": self._l2(joint_cmd_delta[:6]),
            "cmd_delta_right_l2": self._l2(joint_cmd_delta[6:]),
            "grip_delta_l2": self._l2(grip_cmd_delta),
            "grip_delta_max": self._max_abs(grip_cmd_delta),
            "smooth_ratio": float(smooth_ratio),
            "smoothing_correction_l2": self._l2(joint_smoothing_correction),
            "smoothing_correction_max": self._max_abs(joint_smoothing_correction),
            "joint_vel_rms": self._rms(joint_vel),
            "joint_vel_max": self._max_abs(joint_vel),
            "joint_acc_rms": self._rms(joint_acc),
            "joint_acc_max": self._max_abs(joint_acc),
            "joint_jerk_rms": self._rms(joint_jerk),
            "joint_jerk_max": self._max_abs(joint_jerk),
            "grip_vel_rms": self._rms(grip_vel),
            "grip_acc_rms": self._rms(grip_acc),
            "grip_jerk_rms": self._rms(grip_jerk),
            "tracking_joint_l2": self._l2(joint_track),
            "tracking_joint_max": self._max_abs(joint_track),
            "tracking_grip_l2": self._l2(grip_track),
            "joint_cap_hits": joint_cap_hits,
            "grip_cap_hits": grip_cap_hits,
            "raw_path_len": self._raw_path_len,
            "cmd_path_len": self._cmd_path_len,
            "path_smooth_ratio": (
                0.0 if self._raw_path_len < 1e-9 else self._cmd_path_len / self._raw_path_len
            ),
            "accel_energy": self._accel_energy,
            "jerk_energy": self._jerk_energy,
            "raw": raw.copy(),
            "cmd": cmd.copy(),
            "obs": obs.copy(),
            "vel": vel.copy(),
            "acc": acc.copy(),
            "jerk": jerk.copy(),
        }
        for idx, name in enumerate(self.ACTION_NAMES):
            rec[f"raw_{name}"] = float(raw[idx])
            rec[f"cmd_{name}"] = float(cmd[idx])
            rec[f"obs_{name}"] = float(obs[idx])
            rec[f"delta_cmd_{name}"] = float(cmd_delta[idx])
        for idx in self.JOINT_IDX:
            name = self.ACTION_NAMES[int(idx)]
            rec[f"vel_{name}"] = float(vel[idx])
            rec[f"acc_{name}"] = float(acc[idx])
            rec[f"jerk_{name}"] = float(jerk[idx])
        self.records.append(rec)
        self._trim()

        self._last_raw = raw.copy()
        self._last_cmd = cmd.copy()
        self._last_vel = vel.copy()
        self._last_acc = acc.copy()
        return rec

    def rolling_summary(self) -> dict[str, float]:
        if not self.records:
            return {}
        window = self.records[-self.window_steps :]
        keys = [
            "cmd_delta_joint_l2",
            "cmd_delta_joint_max",
            "joint_vel_rms",
            "joint_vel_max",
            "joint_acc_rms",
            "joint_acc_max",
            "joint_jerk_rms",
            "joint_jerk_max",
            "tracking_joint_l2",
            "smoothing_correction_l2",
            "smooth_ratio",
        ]
        summary: dict[str, float] = {}
        for key in keys:
            vals = np.asarray([float(r.get(key, 0.0)) for r in window], dtype=np.float64)
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_max"] = float(np.max(vals))
            summary[f"{key}_p95"] = float(np.percentile(vals, 95))
        return summary

    def should_print(self, step: int) -> bool:
        return self.enabled and self.print_interval_steps > 0 and step % self.print_interval_steps == 0

    def format_online(self, rec: dict[str, Any]) -> str:
        if not rec:
            return ""
        roll = self.rolling_summary()
        return (
            f"  [SMOOTH] step={rec['step']} "
            f"dq_l2={rec['cmd_delta_joint_l2']:.4f}rad "
            f"dq_max={rec['cmd_delta_joint_max']:.4f}rad "
            f"raw_dq_l2={rec['raw_delta_joint_l2']:.4f}rad "
            f"ratio={rec['smooth_ratio']:.2f} path_ratio={rec['path_smooth_ratio']:.2f} "
            f"vel_rms/max={rec['joint_vel_rms']:.3f}/{rec['joint_vel_max']:.3f}rad/s "
            f"acc_rms/max={rec['joint_acc_rms']:.2f}/{rec['joint_acc_max']:.2f}rad/s^2 "
            f"jerk_rms/max={rec['joint_jerk_rms']:.1f}/{rec['joint_jerk_max']:.1f}rad/s^3 "
            f"track_l2/max={rec['tracking_joint_l2']:.4f}/{rec['tracking_joint_max']:.4f}rad "
            f"smooth_corr={rec['smoothing_correction_l2']:.4f}rad "
            f"cap(j/g)={rec['joint_cap_hits']}/{rec['grip_cap_hits']} "
            f"roll_jerk_p95={roll.get('joint_jerk_rms_p95', 0.0):.1f}"
        )

    def print_final_summary(self) -> None:
        if not self.enabled or not self.records:
            return
        roll = self.rolling_summary()
        all_rec = self.records
        elapsed_s = max(1e-6, all_rec[-1]["t_s"] - all_rec[0]["t_s"])
        vals = {
            key: np.asarray([float(r.get(key, 0.0)) for r in all_rec], dtype=np.float64)
            for key in [
                "cmd_delta_joint_l2",
                "joint_vel_rms",
                "joint_acc_rms",
                "joint_jerk_rms",
                "tracking_joint_l2",
                "smoothing_correction_l2",
                "smooth_ratio",
            ]
        }
        print("\n[SMOOTH SUMMARY]")
        print(
            f"  samples={len(all_rec)} duration={elapsed_s:.1f}s "
            f"cmd_path={all_rec[-1]['cmd_path_len']:.3f}rad "
            f"raw_path={all_rec[-1]['raw_path_len']:.3f}rad "
            f"path_ratio={all_rec[-1]['path_smooth_ratio']:.3f}"
        )
        print(
            f"  dq_l2 mean/p95/max="
            f"{np.mean(vals['cmd_delta_joint_l2']):.4f}/"
            f"{np.percentile(vals['cmd_delta_joint_l2'], 95):.4f}/"
            f"{np.max(vals['cmd_delta_joint_l2']):.4f} rad"
        )
        print(
            f"  vel_rms mean/p95/max="
            f"{np.mean(vals['joint_vel_rms']):.3f}/"
            f"{np.percentile(vals['joint_vel_rms'], 95):.3f}/"
            f"{np.max(vals['joint_vel_rms']):.3f} rad/s"
        )
        print(
            f"  acc_rms mean/p95/max="
            f"{np.mean(vals['joint_acc_rms']):.2f}/"
            f"{np.percentile(vals['joint_acc_rms'], 95):.2f}/"
            f"{np.max(vals['joint_acc_rms']):.2f} rad/s^2"
        )
        print(
            f"  jerk_rms mean/p95/max="
            f"{np.mean(vals['joint_jerk_rms']):.1f}/"
            f"{np.percentile(vals['joint_jerk_rms'], 95):.1f}/"
            f"{np.max(vals['joint_jerk_rms']):.1f} rad/s^3"
        )
        print(
            f"  tracking_l2 mean/p95/max="
            f"{np.mean(vals['tracking_joint_l2']):.4f}/"
            f"{np.percentile(vals['tracking_joint_l2'], 95):.4f}/"
            f"{np.max(vals['tracking_joint_l2']):.4f} rad"
        )
        print(
            f"  smooth_ratio mean/p95/max="
            f"{np.mean(vals['smooth_ratio']):.3f}/"
            f"{np.percentile(vals['smooth_ratio'], 95):.3f}/"
            f"{np.max(vals['smooth_ratio']):.3f}; "
            f"rolling_jerk_p95={roll.get('joint_jerk_rms_p95', 0.0):.1f}"
        )

    def maybe_save_outputs(self, *, force: bool = False) -> None:
        if not self.enabled or not self.records:
            return
        now = time.monotonic()
        if not force and self.plot_interval_s <= 0.0:
            return
        if not force and now - self._last_plot_s < self.plot_interval_s:
            return
        self._last_plot_s = now
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        if self.save_csv_enabled:
            self.save_csv()
        if self.save_plots_enabled:
            self.save_plots()

    def save_csv(self) -> None:
        path = self.plot_dir / "smoothness_metrics.csv"
        metric_keys = [
            k for k in self.records[-1].keys()
            if not isinstance(self.records[-1][k], np.ndarray)
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_keys)
            writer.writeheader()
            for rec in self.records:
                writer.writerow({k: rec.get(k) for k in metric_keys})

        raw = np.stack([r["raw"] for r in self.records])
        cmd = np.stack([r["cmd"] for r in self.records])
        obs = np.stack([r["obs"] for r in self.records])
        tcp_sets = {
            "raw_left": piper_batch_local_tcp_xyz(raw, right=False),
            "raw_right": piper_batch_local_tcp_xyz(raw, right=True),
            "cmd_left": piper_batch_local_tcp_xyz(cmd, right=False),
            "cmd_right": piper_batch_local_tcp_xyz(cmd, right=True),
            "obs_left": piper_batch_local_tcp_xyz(obs, right=False),
            "obs_right": piper_batch_local_tcp_xyz(obs, right=True),
        }
        tcp_path = self.plot_dir / "tcp_xyz_metrics.csv"
        tcp_keys = ["step", "t_s", "obs_kind", "chunk_step"]
        for prefix in tcp_sets:
            tcp_keys.extend([f"{prefix}_x_m", f"{prefix}_y_m", f"{prefix}_z_m"])
        with tcp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=tcp_keys)
            writer.writeheader()
            for i, rec in enumerate(self.records):
                row = {
                    "step": rec.get("step"),
                    "t_s": rec.get("t_s"),
                    "obs_kind": rec.get("obs_kind"),
                    "chunk_step": rec.get("chunk_step"),
                }
                for prefix, xyz in tcp_sets.items():
                    row[f"{prefix}_x_m"] = float(xyz[i, 0])
                    row[f"{prefix}_y_m"] = float(xyz[i, 1])
                    row[f"{prefix}_z_m"] = float(xyz[i, 2])
                writer.writerow(row)
        print(f"[SMOOTH] saved TCP xyz CSV: {tcp_path}")

    def _series(self, key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, 0.0)) for r in self.records], dtype=np.float64)

    def _save_cv2_panel_plot(
        self,
        filename: str,
        title: str,
        panels: list[tuple[str, list[tuple[str, np.ndarray, tuple[int, int, int]]]]],
    ) -> None:
        t = np.asarray([r["t_s"] for r in self.records], dtype=np.float64)
        t = t - t[0]
        width = 1500
        panel_h = 230
        margin_l, margin_r = 95, 25
        margin_t, margin_b = 50, 42
        gap = 28
        height = margin_t + margin_b + len(panels) * panel_h + (len(panels) - 1) * gap
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.putText(img, title, (margin_l, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)

        x0 = margin_l
        x1 = width - margin_r
        plot_w = max(1, x1 - x0)
        t_span = max(1e-9, float(t[-1] - t[0])) if len(t) > 1 else 1.0
        x_coords = x0 + ((t - t[0]) / t_span * plot_w).astype(np.int32)

        for panel_idx, (ylabel, series_list) in enumerate(panels):
            y0 = margin_t + panel_idx * (panel_h + gap)
            y1 = y0 + panel_h
            values = []
            for _, y, _ in series_list:
                yy = np.asarray(y, dtype=np.float64)
                yy = yy[np.isfinite(yy)]
                if yy.size:
                    values.append(yy)
            if values:
                merged = np.concatenate(values)
                y_min = float(np.min(merged))
                y_max = float(np.max(merged))
            else:
                y_min, y_max = 0.0, 1.0
            if abs(y_max - y_min) < 1e-9:
                pad = max(1e-3, abs(y_max) * 0.1)
                y_min -= pad
                y_max += pad
            else:
                pad = (y_max - y_min) * 0.08
                y_min -= pad
                y_max += pad

            cv2.rectangle(img, (x0, y0), (x1, y1), (210, 210, 210), 1)
            for frac in (0.25, 0.5, 0.75):
                yy = int(y1 - frac * panel_h)
                cv2.line(img, (x0, yy), (x1, yy), (235, 235, 235), 1)
            cv2.putText(img, ylabel, (12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(img, f"{y_max:.3g}", (12, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(img, f"{y_min:.3g}", (12, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

            legend_x = x0 + 10
            for label, y, color in series_list:
                yy = np.asarray(y, dtype=np.float64)
                if yy.size != t.size:
                    continue
                y_coords = y1 - ((yy - y_min) / (y_max - y_min) * panel_h).astype(np.int32)
                pts = np.stack([x_coords, np.clip(y_coords, y0, y1)], axis=1).astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
                cv2.line(img, (legend_x, y0 + 18), (legend_x + 22, y0 + 18), color, 2, cv2.LINE_AA)
                cv2.putText(img, label, (legend_x + 28, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
                legend_x += 185

        cv2.putText(img, "time (s)", (width // 2 - 45, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.imwrite(str(self.plot_dir / filename), img)

    def save_cv2_plots(self, reason: Exception | None = None) -> None:
        if reason is not None:
            print(f"[SMOOTH] matplotlib unavailable, using OpenCV plots: {reason}")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        t = np.asarray([r["t_s"] for r in self.records], dtype=np.float64)
        t = t - t[0]
        dt = np.diff(t, prepend=t[0])
        default_dt = float(np.median(np.diff(t))) if len(t) > 1 else self.dt_default
        if not np.isfinite(default_dt) or default_dt <= 0.0:
            default_dt = self.dt_default
        dt[0] = default_dt
        dt = np.maximum(dt, 1e-4)
        obs = np.stack([r["obs"] for r in self.records])
        obs_l_tcp = piper_batch_local_tcp_xyz(obs, right=False)
        obs_r_tcp = piper_batch_local_tcp_xyz(obs, right=True)
        obs_delta = np.diff(obs, axis=0, prepend=obs[:1])
        obs_tcp_l_delta = np.diff(obs_l_tcp, axis=0, prepend=obs_l_tcp[:1])
        obs_tcp_r_delta = np.diff(obs_r_tcp, axis=0, prepend=obs_r_tcp[:1])
        obs_joint_delta = obs_delta[:, self.JOINT_IDX]
        obs_joint_vel = obs_joint_delta / dt[:, None]
        obs_joint_acc = np.diff(obs_joint_vel, axis=0, prepend=obs_joint_vel[:1]) / dt[:, None]
        obs_joint_jerk = np.diff(obs_joint_acc, axis=0, prepend=obs_joint_acc[:1]) / dt[:, None]
        obs_delta_joint_l2 = np.linalg.norm(obs_joint_delta, axis=1)
        obs_left_delta_l2 = np.linalg.norm(obs_delta[:, :6], axis=1)
        obs_right_delta_l2 = np.linalg.norm(obs_delta[:, 7:13], axis=1)
        obs_left_tcp_delta_l2 = np.linalg.norm(obs_tcp_l_delta, axis=1)
        obs_right_tcp_delta_l2 = np.linalg.norm(obs_tcp_r_delta, axis=1)
        obs_joint_vel_rms = np.sqrt(np.mean(np.square(obs_joint_vel), axis=1))
        obs_joint_vel_max = np.max(np.abs(obs_joint_vel), axis=1)
        obs_joint_acc_rms = np.sqrt(np.mean(np.square(obs_joint_acc), axis=1))
        obs_joint_acc_max = np.max(np.abs(obs_joint_acc), axis=1)
        obs_joint_jerk_rms = np.sqrt(np.mean(np.square(obs_joint_jerk), axis=1))
        obs_joint_jerk_max = np.max(np.abs(obs_joint_jerk), axis=1)

        blue = (200, 80, 20)
        orange = (30, 130, 230)
        green = (50, 160, 70)
        red = (40, 40, 220)
        purple = (170, 70, 170)
        gray = (120, 120, 120)

        self._save_cv2_panel_plot(
            "cv2_trajectory_command_vs_observation.png",
            "Paper RTC observed key trajectories",
            [
                ("L1/R1 rad", [
                    ("obs L1", obs[:, 0], blue),
                    ("obs R1", obs[:, 7], green),
                ]),
                ("L2/R2 rad", [
                    ("obs L2", obs[:, 1], blue),
                    ("obs R2", obs[:, 8], green),
                ]),
                ("gripper m", [
                    ("obs GL", obs[:, 6], blue),
                    ("obs GR", obs[:, 13], green),
                ]),
            ],
        )
        self._save_cv2_panel_plot(
            "cv2_trajectory_all_joints.png",
            "Paper RTC observed all joint trajectories",
            [
                (
                    f"{name} rad",
                    [
                        (f"obs {name}", obs[:, int(idx)], blue),
                    ],
                )
                for idx, name in zip(self.JOINT_IDX, self.JOINT_NAMES)
            ],
        )
        self._save_cv2_panel_plot(
            "cv2_trajectory_tcp_xyz.png",
            "Paper RTC observed local TCP xyz trajectories",
            [
                ("left TCP x/y/z m", [
                    ("obs x", obs_l_tcp[:, 0], blue),
                    ("obs y", obs_l_tcp[:, 1], green),
                    ("obs z", obs_l_tcp[:, 2], red),
                ]),
                ("right TCP x/y/z m", [
                    ("obs x", obs_r_tcp[:, 0], blue),
                    ("obs y", obs_r_tcp[:, 1], green),
                    ("obs z", obs_r_tcp[:, 2], red),
                ]),
            ],
        )
        self._save_cv2_panel_plot(
            "cv2_smoothness_delta_velocity_timing.png",
            "Paper RTC observed step smoothness and timing",
            [
                ("rad/step", [
                    ("obs dq L2", obs_delta_joint_l2, blue),
                    ("obs left dq L2", obs_left_delta_l2, green),
                    ("obs right dq L2", obs_right_delta_l2, red),
                ]),
                ("TCP m/step", [
                    ("obs left TCP dL2", obs_left_tcp_delta_l2, blue),
                    ("obs right TCP dL2", obs_right_tcp_delta_l2, green),
                ]),
                ("rad/s", [
                    ("obs vel rms", obs_joint_vel_rms, blue),
                    ("obs vel max", obs_joint_vel_max, red),
                ]),
                ("period ms", [
                    ("period", self._series("period_ms"), gray),
                ]),
            ],
        )
        self._save_cv2_panel_plot(
            "cv2_smoothness_acceleration_jerk.png",
            "Paper RTC observed acceleration and jerk",
            [
                ("rad/s^2", [
                    ("obs acc rms", obs_joint_acc_rms, blue),
                    ("obs acc max", obs_joint_acc_max, red),
                ]),
                ("rad/s^3", [
                    ("obs jerk rms", obs_joint_jerk_rms, blue),
                    ("obs jerk max", obs_joint_jerk_max, red),
                ]),
                ("cost", [
                    ("obs acc energy", np.cumsum(np.sum(np.square(obs_joint_acc), axis=1) * dt), green),
                    ("obs jerk energy", np.cumsum(np.sum(np.square(obs_joint_jerk), axis=1) * dt), purple),
                ]),
            ],
        )
        print(f"[SMOOTH] saved OpenCV CSV/plots under {self.plot_dir}")

    def save_plots(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as exc:
            self.save_cv2_plots(exc)
            return

        t = np.asarray([r["t_s"] for r in self.records], dtype=np.float64)
        t = t - t[0]
        obs = np.stack([r["obs"] for r in self.records])
        obs_l_tcp = piper_batch_local_tcp_xyz(obs, right=False)
        obs_r_tcp = piper_batch_local_tcp_xyz(obs, right=True)
        dt = np.diff(t, prepend=t[0])
        default_dt = float(np.median(np.diff(t))) if len(t) > 1 else self.dt_default
        if not np.isfinite(default_dt) or default_dt <= 0.0:
            default_dt = self.dt_default
        dt[0] = default_dt
        dt = np.maximum(dt, 1e-4)
        obs_delta = np.diff(obs, axis=0, prepend=obs[:1])
        obs_joint_delta = obs_delta[:, self.JOINT_IDX]
        vel = obs_joint_delta / dt[:, None]
        acc = np.diff(vel, axis=0, prepend=vel[:1]) / dt[:, None]
        jerk = np.diff(acc, axis=0, prepend=acc[:1]) / dt[:, None]
        obs_l_tcp_delta = np.diff(obs_l_tcp, axis=0, prepend=obs_l_tcp[:1])
        obs_r_tcp_delta = np.diff(obs_r_tcp, axis=0, prepend=obs_r_tcp[:1])
        obs_left_delta_l2 = np.linalg.norm(obs_delta[:, :6], axis=1)
        obs_right_delta_l2 = np.linalg.norm(obs_delta[:, 7:13], axis=1)
        obs_left_tcp_delta_l2 = np.linalg.norm(obs_l_tcp_delta, axis=1)
        obs_right_tcp_delta_l2 = np.linalg.norm(obs_r_tcp_delta, axis=1)
        obs_vel_rms = np.sqrt(np.mean(np.square(vel), axis=1))
        obs_vel_max = np.max(np.abs(vel), axis=1)
        obs_acc_rms = np.sqrt(np.mean(np.square(acc), axis=1))
        obs_acc_max = np.max(np.abs(acc), axis=1)
        obs_jerk_rms = np.sqrt(np.mean(np.square(jerk), axis=1))
        obs_jerk_max = np.max(np.abs(jerk), axis=1)
        obs_acc_energy = np.cumsum(np.sum(np.square(acc), axis=1) * dt)
        obs_jerk_energy = np.cumsum(np.sum(np.square(jerk), axis=1) * dt)

        def arr(key: str) -> np.ndarray:
            return np.asarray([float(r.get(key, 0.0)) for r in self.records], dtype=np.float64)

        def mark_obs_and_triggers(axis) -> None:
            obs_t = [t[i] for i, r in enumerate(self.records) if r.get("obs_kind") == "obs"]
            stride = max(1, len(obs_t) // 30) if obs_t else 1
            for x in obs_t[::stride]:
                axis.axvline(x, color="tab:green", alpha=0.12, linewidth=0.8)
            trigger = arr("trigger_count")
            for i in np.where(np.diff(trigger, prepend=trigger[0]) > 0)[0]:
                axis.axvline(t[i], color="tab:red", alpha=0.18, linewidth=0.9)

        def finish_time_axis(axis, *, legend: bool = True) -> None:
            axis.grid(True, alpha=0.25)
            mark_obs_and_triggers(axis)
            if legend:
                axis.legend(loc="upper right", ncol=3, fontsize=8)

        def save_arm_overview(
            *,
            side_name: str,
            joint_indices: list[int],
            joint_names: list[str],
            gripper_index: int,
            obs_tcp: np.ndarray,
            obs_joint_delta_l2: np.ndarray,
            obs_tcp_delta_l2: np.ndarray,
            filename: str,
        ) -> None:
            fig = plt.figure(figsize=(16, 18))
            gs = fig.add_gridspec(
                6,
                2,
                height_ratios=[1.2, 1.0, 0.9, 1.15, 0.95, 1.8],
                hspace=0.38,
                wspace=0.22,
            )

            ax = fig.add_subplot(gs[0, :])
            for idx, name in zip(joint_indices, joint_names):
                ax.plot(t, obs[:, idx], label=f"obs {name}", linewidth=1.35)
            ax.set_title(f"{side_name} observed joints")
            ax.set_ylabel("rad")
            finish_time_axis(ax)

            ax = fig.add_subplot(gs[1, :])
            side_delta = obs_delta[:, joint_indices]
            for j, name in enumerate(joint_names):
                ax.plot(t, side_delta[:, j], label=f"{name} step delta", linewidth=1.1)
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
            ax.set_title(f"{side_name} observed joint step delta")
            ax.set_ylabel("rad/step")
            finish_time_axis(ax)

            ax = fig.add_subplot(gs[2, 0])
            ax.plot(t, obs[:, gripper_index], label="obs gripper", linewidth=1.4)
            ax.set_title(f"{side_name} observed gripper")
            ax.set_ylabel("m")
            finish_time_axis(ax, legend=True)

            ax = fig.add_subplot(gs[2, 1])
            ax.plot(t, obs_joint_delta_l2, label="obs joint delta L2", linewidth=1.3)
            ax.plot(t, obs_tcp_delta_l2, label="obs TCP delta L2", linewidth=1.3)
            ax.set_title(f"{side_name} observed compact motion")
            ax.set_ylabel("mixed")
            finish_time_axis(ax, legend=True)

            ax = fig.add_subplot(gs[3, :])
            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                ax.plot(t, obs_tcp[:, axis_idx], label=f"obs {axis_name}", linewidth=1.45)
            ax.set_title(f"{side_name} observed TCP xyz over time")
            ax.set_ylabel("m")
            finish_time_axis(ax)

            ax = fig.add_subplot(gs[4, :])
            ax.plot(t, obs_vel_rms, label="obs joint vel rms", linewidth=1.25)
            ax.plot(t, obs_acc_rms, label="obs joint acc rms", linewidth=1.25)
            ax.plot(t, obs_jerk_rms, label="obs joint jerk rms", linewidth=1.25)
            ax.plot(t, arr("period_ms"), label="period ms", linewidth=1.0, alpha=0.8)
            ax.set_title("Observed smoothness and control timing")
            ax.set_ylabel("mixed")
            finish_time_axis(ax)

            ax3d = fig.add_subplot(gs[5, :], projection="3d")
            ax3d.plot(obs_tcp[:, 0], obs_tcp[:, 1], obs_tcp[:, 2], label="obs TCP", linewidth=1.8)
            ax3d.scatter(obs_tcp[:1, 0], obs_tcp[:1, 1], obs_tcp[:1, 2], marker="o", s=35, label="start")
            ax3d.scatter(obs_tcp[-1:, 0], obs_tcp[-1:, 1], obs_tcp[-1:, 2], marker="x", s=45, label="end")
            ax3d.set_title(f"{side_name} observed TCP 3D trajectory")
            ax3d.set_xlabel("x m")
            ax3d.set_ylabel("y m")
            ax3d.set_zlabel("z m")
            ax3d.legend(loc="upper right", fontsize=8)

            fig.suptitle(f"Paper RTC {side_name} arm overview", fontsize=15)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
            fig.savefig(self.plot_dir / filename, dpi=150)
            plt.close(fig)

        save_arm_overview(
            side_name="left",
            joint_indices=[0, 1, 2, 3, 4, 5],
            joint_names=["L1", "L2", "L3", "L4", "L5", "L6"],
            gripper_index=6,
            obs_tcp=obs_l_tcp,
            obs_joint_delta_l2=obs_left_delta_l2,
            obs_tcp_delta_l2=obs_left_tcp_delta_l2,
            filename="trajectory_left_arm_overview.png",
        )
        save_arm_overview(
            side_name="right",
            joint_indices=[7, 8, 9, 10, 11, 12],
            joint_names=["R1", "R2", "R3", "R4", "R5", "R6"],
            gripper_index=13,
            obs_tcp=obs_r_tcp,
            obs_joint_delta_l2=obs_right_delta_l2,
            obs_tcp_delta_l2=obs_right_tcp_delta_l2,
            filename="trajectory_right_arm_overview.png",
        )

        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        axes[0].plot(t, obs[:, 0], label="obs L1")
        axes[0].plot(t, obs[:, 7], label="obs R1")
        axes[0].set_ylabel("joint rad")
        axes[0].legend(loc="upper right")
        axes[1].plot(t, obs[:, 1], label="obs L2")
        axes[1].plot(t, obs[:, 8], label="obs R2")
        axes[1].set_ylabel("joint rad")
        axes[1].legend(loc="upper right")
        axes[2].plot(t, obs[:, 6], label="obs GL")
        axes[2].plot(t, obs[:, 13], label="obs GR")
        axes[2].set_ylabel("gripper m")
        axes[2].legend(loc="upper right")
        axes[3].plot(t, np.linalg.norm(obs_joint_delta, axis=1), label="obs joint delta L2")
        axes[3].plot(t, np.linalg.norm(obs_delta[:, self.GRIP_IDX], axis=1), label="obs gripper delta L2")
        axes[3].set_ylabel("rad")
        axes[3].set_xlabel("time s")
        axes[3].legend(loc="upper right")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            mark_obs_and_triggers(ax)
        fig.suptitle("Paper RTC observed key trajectories")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "trajectory_command_vs_observation.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
        for ax, idx, name in zip(axes.ravel(), self.JOINT_IDX, self.JOINT_NAMES):
            ax.plot(t, obs[:, idx], label=f"obs {name}")
            ax.set_title(name)
            ax.grid(True, alpha=0.25)
            mark_obs_and_triggers(ax)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("time s")
        axes[-1, 1].set_xlabel("time s")
        axes[-1, 2].set_xlabel("time s")
        fig.suptitle("Paper RTC observed all joint trajectories")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "trajectory_all_joints.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        for col, title, obs_tcp in [
            (0, "left observed local TCP", obs_l_tcp),
            (1, "right observed local TCP", obs_r_tcp),
        ]:
            for row, axis_name in enumerate(["x", "y", "z"]):
                axes[row, col].plot(t, obs_tcp[:, row], label=f"obs {axis_name}")
                axes[row, col].set_title(f"{title} {axis_name}")
                axes[row, col].set_ylabel("m")
                axes[row, col].grid(True, alpha=0.25)
                axes[row, col].legend(loc="upper right")
                mark_obs_and_triggers(axes[row, col])
        axes[-1, 0].set_xlabel("time s")
        axes[-1, 1].set_xlabel("time s")
        fig.suptitle("Paper RTC observed local TCP xyz trajectories")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "trajectory_tcp_xyz.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        axes[0].plot(t, np.linalg.norm(obs_joint_delta, axis=1), label="obs joint delta L2")
        axes[0].plot(t, obs_left_delta_l2, label="obs left delta L2")
        axes[0].plot(t, obs_right_delta_l2, label="obs right delta L2")
        axes[0].set_ylabel("rad/step")
        axes[0].legend(loc="upper right")
        axes[1].plot(t, obs_left_tcp_delta_l2, label="obs left TCP delta L2")
        axes[1].plot(t, obs_right_tcp_delta_l2, label="obs right TCP delta L2")
        axes[1].set_ylabel("m/step")
        axes[1].legend(loc="upper right")
        axes[2].plot(t, obs_vel_rms, label="obs vel rms")
        axes[2].plot(t, obs_vel_max, label="obs vel max")
        axes[2].set_ylabel("rad/s")
        axes[2].legend(loc="upper right")
        axes[3].plot(t, arr("period_ms"), label="control period")
        axes[3].set_ylabel("ms")
        axes[3].set_xlabel("time s")
        axes[3].legend(loc="upper right")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            mark_obs_and_triggers(ax)
        fig.suptitle("Paper RTC observed step smoothness and timing")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "smoothness_delta_velocity_timing.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(t, obs_acc_rms, label="obs acc rms")
        axes[0].plot(t, obs_acc_max, label="obs acc max")
        axes[0].set_ylabel("rad/s^2")
        axes[0].legend(loc="upper right")
        axes[1].plot(t, obs_jerk_rms, label="obs jerk rms")
        axes[1].plot(t, obs_jerk_max, label="obs jerk max")
        axes[1].set_ylabel("rad/s^3")
        axes[1].legend(loc="upper right")
        axes[2].plot(t, obs_acc_energy, label="obs acc energy cumulative")
        axes[2].plot(t, obs_jerk_energy, label="obs jerk energy cumulative")
        axes[2].set_ylabel("cost")
        axes[2].set_xlabel("time s")
        axes[2].legend(loc="upper right")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            mark_obs_and_triggers(ax)
        fig.suptitle("Paper RTC observed acceleration and jerk")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "smoothness_acceleration_jerk.png", dpi=140)
        plt.close(fig)

        extent = [float(t[0]), float(t[-1]) if len(t) > 1 else 1.0, 0, len(self.JOINT_NAMES)]
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for ax, data, title in [
            (axes[0], np.abs(vel).T, "observed |velocity| rad/s"),
            (axes[1], np.abs(acc).T, "observed |acceleration| rad/s^2"),
            (axes[2], np.abs(jerk).T, "observed |jerk| rad/s^3"),
        ]:
            im = ax.imshow(data, aspect="auto", origin="lower", extent=extent)
            ax.set_yticks(np.arange(len(self.JOINT_NAMES)) + 0.5)
            ax.set_yticklabels(self.JOINT_NAMES)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, pad=0.01)
        axes[-1].set_xlabel("time s")
        fig.suptitle("Paper RTC observed derivative heatmaps")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "smoothness_derivative_heatmaps.png", dpi=140)
        plt.close(fig)

        x = np.arange(len(self.JOINT_NAMES))
        width = 0.25
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, np.sqrt(np.mean(np.square(vel), axis=0)), width, label="obs vel rms")
        ax.bar(x, np.sqrt(np.mean(np.square(acc), axis=0)), width, label="obs acc rms")
        ax.bar(x + width, np.sqrt(np.mean(np.square(jerk), axis=0)), width, label="obs jerk rms")
        ax.set_xticks(x)
        ax.set_xticklabels(self.JOINT_NAMES)
        ax.set_ylabel("mixed units by derivative order")
        ax.set_title("Observed per-joint smoothness RMS")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "smoothness_per_joint_rms.png", dpi=140)
        plt.close(fig)

        print(f"[SMOOTH] saved CSV/plots under {self.plot_dir}")


# =====================================================
#              主推理函数（RTC 模式）
# =====================================================

def run_rtc_inference(args):
    """RTC 推理执行：每步调用一次 infer()，获取单步动作后直接执行。"""
    global _EXECUTION_ACTION_SPACE
    max_joint_delta_rad = float(args.max_joint_delta_rad)
    if not math.isfinite(max_joint_delta_rad) or max_joint_delta_rad <= 0.0:
        raise ValueError(
            f"max_joint_delta_rad must be finite and > 0, got {max_joint_delta_rad!r}"
        )
    max_ee_position_delta_m = float(getattr(args, "max_ee_position_delta_m", 0.01))
    max_ee_rotation_delta_rad = float(getattr(args, "max_ee_rotation_delta_rad", 0.0872665))
    ee_position_filter_alpha = float(getattr(args, "ee_position_filter_alpha", 0.2))
    ee_rotation_filter_alpha = float(getattr(args, "ee_rotation_filter_alpha", 0.2))
    ee_pose_log_interval_steps = int(getattr(args, "ee_pose_log_interval_steps", 30))
    if not math.isfinite(max_ee_position_delta_m) or max_ee_position_delta_m <= 0.0:
        raise ValueError("max_ee_position_delta_m must be finite and > 0")
    if not math.isfinite(max_ee_rotation_delta_rad) or max_ee_rotation_delta_rad <= 0.0:
        raise ValueError("max_ee_rotation_delta_rad must be finite and > 0")
    if not 0.0 < ee_position_filter_alpha <= 1.0:
        raise ValueError("ee_position_filter_alpha must be in (0, 1]")
    if not 0.0 < ee_rotation_filter_alpha <= 1.0:
        raise ValueError("ee_rotation_filter_alpha must be in (0, 1]")
    print("\n" + "=" * 60)
    print("  RTC 在线推理执行 - ROS2 AGILEXDroid 平台")
    print(f"  control_hz : {args.control_hz} Hz  (dt={1000/args.control_hz:.1f} ms)")
    print(f"  max_steps  : {args.max_steps if args.max_steps > 0 else '无限'}")
    print(f"  server     : {args.host}:{args.port}")
    print(f"  hardware   : ROS-only service {args.robot_endpoint}")
    print(f"  joint cap  : {max_joint_delta_rad:.3f} rad/step")
    print("=" * 60)

    save_debug_images = bool(getattr(args, 'save_debug_images', False))
    save_dir = Path(getattr(args, 'save_image_dir',
                            '/tmp/rtc_debug_images'))
    if save_debug_images:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)

    # WebSocket 客户端
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port
    )
    server_metadata = client.get_server_metadata()
    print("server metadata:", server_metadata)
    paper_rtc_metadata = server_metadata.get("paper_rtc", server_metadata)
    if not isinstance(paper_rtc_metadata, dict):
        paper_rtc_metadata = {}
    _EXECUTION_ACTION_SPACE = str(
        paper_rtc_metadata.get("execution_action_space", "absolute_joint")
    )
    if _EXECUTION_ACTION_SPACE not in {"absolute_joint", "absolute_end_effector"}:
        raise RuntimeError(
            f"unsupported server execution_action_space: {_EXECUTION_ACTION_SPACE!r}"
        )
    print(f"  action space: {_EXECUTION_ACTION_SPACE}")

    obs_buffer = ObservationBuffer()
    robot_service = None
    chunk_planner = None

    print("\n[RobotService] 连接硬件服务 ...")
    robot_service = RobotArmServiceClient(
        str(args.robot_endpoint),
        prompt=args.prompt,
        recv_timeout_ms=int(getattr(args, 'robot_service_recv_timeout_ms', 30000)),
    )
    robot_service.action_space = _EXECUTION_ACTION_SPACE
    robot_service.connect()
    print("[RobotService] reset：由 start_robot_arm_service.py 运动到初始姿态 ...")
    reset_obs = robot_service.reset()
    reset_snapshot = robot_service_obs_to_snapshot(reset_obs)
    obs_buffer.set_snapshot(reset_snapshot)
    startup_init_hold_action = snapshot_to_robot_service_action(reset_snapshot)
    print("[RobotService] 初始观察已就绪")

    client_chunk_execution = bool(getattr(args, 'client_chunk_execution', True))
    if not client_chunk_execution:
        raise RuntimeError("Paper-RTC single-step obs/tick requests were removed; use client_chunk_execution=true.")
    action_horizon = int(paper_rtc_metadata.get("action_horizon", 50))
    server_execute_horizon = int(
        paper_rtc_metadata.get(
            "execute_horizon",
            paper_rtc_metadata.get("execution_horizon", 10),
        )
    )
    configured_execute_horizon = int(getattr(args, 'client_chunk_execute_horizon', 0))
    client_chunk_s_min = configured_execute_horizon if configured_execute_horizon > 0 else server_execute_horizon
    client_chunk_s_min = max(1, min(client_chunk_s_min, action_horizon - 1))
    client_chunk_delay_mode = str(getattr(args, 'client_chunk_delay_mode', 'fixed')).strip().lower()
    if client_chunk_delay_mode not in {'fixed', 'realtime_ceil'}:
        raise ValueError(f"unsupported client_chunk_delay_mode: {client_chunk_delay_mode!r}")
    configured_max_delay_steps = int(getattr(args, 'client_chunk_max_delay_steps', 0))
    client_chunk_max_delay_steps = action_horizon - 1 if configured_max_delay_steps <= 0 else configured_max_delay_steps
    client_chunk_max_delay_steps = max(0, min(client_chunk_max_delay_steps, action_horizon - 1))
    client_chunk_fixed_delay_steps = max(0, int(getattr(args, 'client_chunk_fixed_delay_steps', 5)))
    client_chunk_fixed_delay_steps = min(client_chunk_fixed_delay_steps, client_chunk_max_delay_steps)
    if not bool(paper_rtc_metadata.get("supports_client_chunk_execution", False)):
        raise RuntimeError(
            "Connected Paper-RTC server does not support client chunk execution. "
            "Restart scripts/deploy/start_agilex_paper_rtc_policy_server.sh after deploying serve_paper_rtc_policy.py."
        )

    dt = 1.0 / args.control_hz
    obs_send_hz = float(getattr(args, 'obs_send_hz', 10.0))
    if obs_send_hz <= 0.0 or obs_send_hz >= args.control_hz:
        full_obs_interval = 1
    else:
        full_obs_interval = max(1, int(round(args.control_hz / obs_send_hz)))
    effective_obs_hz = args.control_hz / full_obs_interval
    max_steps = int(getattr(args, 'max_steps', 0))
    stutter_warn_ms = float(getattr(args, 'stutter_warn_ms', 0.0))
    if stutter_warn_ms <= 0.0:
        stutter_warn_ms = dt * 1000 * 1.5
    stutter_diagnostics = bool(getattr(args, 'stutter_diagnostics', True))
    control_log_interval_steps = int(getattr(args, 'control_log_interval_steps', 300))
    full_log_interval_steps = int(getattr(args, 'full_log_interval_steps', 0))
    startup_full_log_steps = int(getattr(args, 'startup_full_log_steps', 0))
    overrun_warn_interval_steps = int(getattr(args, 'overrun_warn_interval_steps', 300))
    smoothness_monitor = SmoothnessMonitor(
        control_hz=args.control_hz,
        joint_delta_cap_rad=max_joint_delta_rad,
        enabled=(
            bool(getattr(args, 'smoothness_diagnostics', True))
            and _EXECUTION_ACTION_SPACE == "absolute_joint"
        ),
        plot_dir=str(getattr(args, 'smoothness_plot_dir', '/tmp/paper_rtc_smoothness')),
        history_limit=int(getattr(args, 'smoothness_history_limit', 9000)),
        window_steps=int(getattr(args, 'smoothness_window_steps', 60)),
        print_interval_steps=int(getattr(args, 'smoothness_print_interval_steps', 30)),
        plot_interval_s=float(getattr(args, 'smoothness_plot_interval_s', 0.0)),
        save_csv=bool(getattr(args, 'smoothness_save_csv', True)),
        save_plots=bool(getattr(args, 'smoothness_save_plots', True)),
    )
    timing_monitor = TimingBottleneckMonitor(
        control_hz=args.control_hz,
        enabled=bool(getattr(args, 'timing_diagnostics', True)),
        output_dir=str(getattr(args, 'smoothness_plot_dir', '/tmp/paper_rtc_smoothness')),
        history_limit=int(getattr(args, 'timing_history_limit', 9000)),
        window_steps=int(getattr(args, 'timing_window_steps', 90)),
        print_interval_steps=int(getattr(args, 'timing_print_interval_steps', 90)),
        save_csv=bool(getattr(args, 'timing_save_csv', True)),
    )

    print("[CHUNK] running no-motion warmup request; robot remains at reset/init pose ...")
    warmup_start_s = time.monotonic()
    warmup_obs = make_paper_rtc_client_chunk_request(
        obs_buffer.get_snapshot(copy_images=False),
        args.prompt,
        client_step=0,
        local_chunk_id=-1,
        local_chunk_index=0,
        local_remaining=0,
        prev_leftover_robot=None,
        fixed_delay_steps=0,
        prefix_attention_horizon=0,
    )
    warmup_out = client.infer(warmup_obs)
    warmup_elapsed_s = time.monotonic() - warmup_start_s
    if not isinstance(warmup_out, dict):
        raise RuntimeError(f"warmup response must be a dict, got {type(warmup_out).__name__}")
    warmup_robot_actions = np.asarray(warmup_out.get("actions"), dtype=np.float32)
    if warmup_robot_actions.ndim != 2:
        raise RuntimeError(
            "warmup response must contain rank-2 actions, "
            f"got {warmup_robot_actions.shape}"
        )
    warmup_timing = dict(warmup_out.get("server_timing", {}))
    warmup_infer_ms = float(warmup_timing.get("last_infer_ms", 0.0))
    print(
        "[CHUNK] warmup complete; discarded action chunk "
        f"elapsed={warmup_elapsed_s * 1000:.1f}ms server_infer={warmup_infer_ms:.1f}ms"
    )

    chunk_planner = AsyncClientChunkPlanner(
        client=client,
        obs_buffer=obs_buffer,
        prompt=args.prompt,
        fixed_delay_steps=client_chunk_fixed_delay_steps,
        delay_mode=client_chunk_delay_mode,
        control_hz=args.control_hz,
        max_delay_steps=client_chunk_max_delay_steps,
    )
    chunk_planner.start()

    def current_hold_policy_prefix() -> Optional[np.ndarray]:
        if client_chunk_fixed_delay_steps <= 0:
            return None
        hold_action = snapshot_to_policy_action(obs_buffer.get_snapshot(copy_images=False))
        return np.repeat(hold_action[None, :], client_chunk_fixed_delay_steps, axis=0).astype(np.float32)

    chunk_planner.request(
        client_step=0,
        local_chunk_id=-1,
        local_chunk_index=0,
        prev_leftover_robot=current_hold_policy_prefix(),
        reason="initial",
    )
    startup_hold_min_s = max(0.0, float(getattr(args, 'startup_init_hold_min_s', 0.0)))
    startup_hold_timeout_s = max(0.0, float(getattr(args, 'startup_init_hold_timeout_s', 0.0)))
    startup_hold_log_interval_s = max(
        0.2,
        float(getattr(args, 'startup_init_hold_log_interval_s', 1.0)),
    )

    print(
        "[CHUNK] requesting first action chunk; holding fixed init pose via ROS-only service "
        f"(min_hold={startup_hold_min_s:.1f}s, "
        f"timeout={'none' if startup_hold_timeout_s <= 0 else f'{startup_hold_timeout_s:.1f}s'}) ..."
    )
    first_request_start_s = time.monotonic()
    hold_deadline_s = (
        None if startup_hold_timeout_s <= 0.0 else first_request_start_s + startup_hold_timeout_s
    )
    hold_steps = 0
    last_hold_log_s = first_request_start_s
    while True:
        now_s = time.monotonic()
        elapsed_s = now_s - first_request_start_s
        if chunk_planner.ready() and elapsed_s >= startup_hold_min_s:
            break
        if hold_deadline_s is not None and now_s >= hold_deadline_s:
            print("[CHUNK] first chunk wait timed out; RTC loop will continue holding until a chunk arrives")
            break
        if (not chunk_planner.pending()) and (not chunk_planner.ready()):
            err = chunk_planner.last_error()
            if err:
                print(f"[CHUNK] initial request failed; retrying while holding init pose: {err}")
            chunk_planner.request(
                client_step=0,
                local_chunk_id=-1,
                local_chunk_index=0,
                prev_leftover_robot=current_hold_policy_prefix(),
                reason="initial_retry",
            )

        step_start_s = time.monotonic()
        init_action = startup_init_hold_action.copy()
        service_obs, service_done, service_info = robot_service.step(init_action)
        obs_buffer.set_snapshot(robot_service_obs_to_snapshot(service_obs))
        hold_steps += 1
        if service_done:
            raise RuntimeError(
                "RobotArmService returned done=True during startup init hold "
                f"at service step {service_info.get('step', '?')}"
            )
        now_after_step_s = time.monotonic()
        if now_after_step_s - last_hold_log_s >= startup_hold_log_interval_s:
            print(
                f"[startup-hold] steps={hold_steps} elapsed={now_after_step_s - first_request_start_s:.1f}s "
                f"chunk_ready={int(chunk_planner.ready())}"
            )
            last_hold_log_s = now_after_step_s
        sleep_s = dt - (now_after_step_s - step_start_s)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    print(
        f"[CHUNK] first chunk ready={int(chunk_planner.ready())}; "
        f"held init pose for {hold_steps} steps / {time.monotonic() - first_request_start_s:.1f}s"
    )

    print(f"\n[系统] 就绪，开始 RTC 控制循环 (dt={dt*1000:.1f}ms)")
    print(f"  full obs upload: every {full_obs_interval} steps ({effective_obs_hz:.1f} Hz)")
    print(f"  client chunks   : 1")
    print(
        f"  chunk horizon   : H={action_horizon}, s_min={client_chunk_s_min}, "
        f"delay_mode={client_chunk_delay_mode}, d_fallback={client_chunk_fixed_delay_steps}, "
        f"d_max={client_chunk_max_delay_steps}"
    )
    print(f"  stutter warn    : loop/period > {stutter_warn_ms:.1f} ms")
    print(
        "  control logs    : "
        f"short every {control_log_interval_steps} steps, "
        f"full every {full_log_interval_steps} steps"
    )
    if smoothness_monitor.enabled:
        print(f"  smoothness      : print every {smoothness_monitor.print_interval_steps} steps")
        print(f"  smoothness plots: {smoothness_monitor.plot_dir}")
    if timing_monitor.enabled:
        print(f"  timing          : print every {timing_monitor.print_interval_steps} steps")
        print(f"  timing CSV      : {timing_monitor.output_dir / 'timing_bottleneck_metrics.csv'}")
    print("  按 Ctrl+C 停止")
    print("-" * 60)

    # ──────────────────────────────────────────────────────────────
    # RTC control loop:
    #   - execute local current_robot_chunk[current_chunk_index]
    #   - request the next H-step chunk in AsyncClientChunkPlanner
    #   - no per-step obs/tick WebSocket request in the control loop
    # ──────────────────────────────────────────────────────────────
    def rtc_loop():
        step = 0
        seed_snapshot = obs_buffer.get_snapshot(copy_images=False)
        if _EXECUTION_ACTION_SPACE == "absolute_end_effector":
            last_jl = seed_snapshot['end_pose_left'].copy()
            last_jr = seed_snapshot['end_pose_right'].copy()
        else:
            last_jl = seed_snapshot['joint_left'].copy()
            last_jr = seed_snapshot['joint_right'].copy()
        last_gl = np.array([float(seed_snapshot['gripper_position'][0])], dtype=np.float32)
        last_gr = np.array([float(seed_snapshot['gripper_position'][1])], dtype=np.float32)
        last_server_timing: dict[str, Any] = {}
        current_robot_chunk: Optional[np.ndarray] = None
        current_chunk_index = 0
        current_chunk_id = -1
        installed_chunks = 0
        prev_step_start = None
        prev_log_ms = 0.0

        def hold_policy_action() -> np.ndarray:
            gl_norm = float(last_gl[0]) / 0.105
            gr_norm = float(last_gr[0]) / 0.105
            return np.concatenate(
                [
                    last_jl,
                    np.array([gl_norm], dtype=np.float32),
                    last_jr,
                    np.array([gr_norm], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)

        def install_chunk_packet(packet: dict[str, Any], now_step: int) -> dict[str, Any]:
            nonlocal current_robot_chunk
            nonlocal current_chunk_index, current_chunk_id, installed_chunks
            req = dict(packet.get("request", {}))
            robot_chunk = np.asarray(packet.get("robot_actions"), dtype=np.float32)
            if robot_chunk.ndim != 2:
                raise RuntimeError(
                    f"invalid chunk packet shape robot={robot_chunk.shape}"
                )
            request_reason = str(req.get("reason", ""))
            request_step = int(req.get("client_step", now_step))
            elapsed_steps = max(0, int(now_step) - request_step)
            request_delay_steps = int(req.get("fixed_delay_steps", client_chunk_fixed_delay_steps))
            skip = 0 if request_reason == "initial" else request_delay_steps
            skip = max(0, min(skip, robot_chunk.shape[0] - 1))
            current_robot_chunk = robot_chunk.copy()
            current_chunk_index = skip
            installed_chunks += 1
            current_chunk_id = installed_chunks

            server_timing = dict(packet.get("server_timing", {}))
            server_timing.update(dict(packet.get("client_timing", {})))
            response_time_s = server_timing.get("chunk_response_time_s")
            chunk_age_ms = 0.0
            if response_time_s is not None:
                chunk_age_ms = (time.monotonic() - float(response_time_s)) * 1000
            server_timing.update({
                "chunk_mode": "client_execution",
                "chunk_id": current_chunk_id,
                "chunk_request_step": request_step,
                "chunk_install_step": int(now_step),
                "chunk_elapsed_steps": elapsed_steps,
                "chunk_fixed_delay_steps": client_chunk_fixed_delay_steps,
                "chunk_delay_steps": request_delay_steps,
                "chunk_delay_mode": str(req.get("delay_mode", client_chunk_delay_mode)),
                "chunk_delay_source": str(req.get("delay_source", "")),
                "chunk_skip_steps": skip,
                "chunk_len": int(current_robot_chunk.shape[0] - current_chunk_index),
                "queue_remaining": int(current_robot_chunk.shape[0] - current_chunk_index),
                "queue_underrun": False,
                "action_stale": False,
                "chunk_response_age_ms": chunk_age_ms,
            })
            return server_timing

        def maybe_request_next_chunk(request_step: int) -> None:
            if chunk_planner.pending() or current_robot_chunk is None:
                return
            if current_chunk_index < client_chunk_s_min:
                return
            prefix_end = min(len(current_robot_chunk), action_horizon)
            if current_chunk_index >= prefix_end:
                return
            prev_leftover_robot = current_robot_chunk[current_chunk_index:prefix_end].copy()
            if len(prev_leftover_robot) <= 0:
                return
            chunk_planner.request(
                client_step=request_step,
                local_chunk_id=current_chunk_id,
                local_chunk_index=current_chunk_index,
                prev_leftover_robot=prev_leftover_robot,
                reason="replan",
            )

        while True:
            if max_steps > 0 and step >= max_steps:
                print(f"[系统] 已执行 {step} 步，达到上限，退出")
                return

            t_start = time.monotonic()
            if prev_step_start is None:
                period_ms = 0.0
            else:
                period_ms = (t_start - prev_step_start) * 1000
            prev_step_start = t_start

            # 构造观察
            snapshot = obs_buffer.get_snapshot(copy_images=False)
            snapshot = obs_buffer.get_snapshot(copy_images=False)
            t_snap = time.monotonic()
            packet = chunk_planner.pop_ready()
            t_obs = t_snap
            t_infer = t_obs
            if packet is not None:
                server_timing = install_chunk_packet(packet, step)
                last_server_timing = server_timing.copy()
                obs_kind = "chunk"
                payload_kib = float(server_timing.get("chunk_payload_kib", 0.0))
            else:
                server_timing = dict(last_server_timing)
                server_timing.update({
                    "chunk_mode": "client_execution",
                    "chunk_id": current_chunk_id,
                    "chunk_idx": current_chunk_index,
                    "chunk_pending": chunk_planner.pending(),
                    "queue_remaining": (
                        0 if current_robot_chunk is None
                        else max(0, len(current_robot_chunk) - current_chunk_index)
                    ),
                })
                obs_kind = "local"
                payload_kib = 0.0

            if current_robot_chunk is not None and current_chunk_index < len(current_robot_chunk):
                action = current_robot_chunk[current_chunk_index].copy()
                chunk_step = current_chunk_index
                server_timing.update({
                    "chunk_step": chunk_step,
                    "queue_before_get": len(current_robot_chunk) - current_chunk_index,
                    "queue_after_get": max(0, len(current_robot_chunk) - current_chunk_index - 1),
                    "queue_remaining": max(0, len(current_robot_chunk) - current_chunk_index - 1),
                    "action_stale": False,
                    "queue_underrun": False,
                })
            else:
                action = hold_policy_action()
                chunk_step = -1
                server_timing.update({
                    "chunk_step": -1,
                    "action_stale": True,
                    "queue_underrun": True,
                })
                obs_kind = "hold"
            out = {"actions": action, "server_timing": server_timing}

            # ── 调用 RTC 服务端 ────────────────────────────────────
            # 返回 {"actions": np.ndarray(14,), "server_timing": {...}}
            # ──────────────────────────────────────────────────────

            # step=0 时打印原始响应结构，确认协议正常
            if step == 0:
                print(f"  [DIAG] server response keys : {list(out.keys()) if out else 'None'}")
                if out:
                    a0 = out.get("actions")
                    print(f"  [DIAG] actions shape/dtype: "
                          f"{getattr(a0,'shape','?')} / {getattr(a0,'dtype','?')}")
                    print(f"  [DIAG] server_timing       : {out.get('server_timing')}")
                    print(f"  [DIAG] request kind        : {obs_kind}")

            action = out.get("actions") if out else None
            if action is None:
                print("[警告] 推理返回空，跳过本步")
                continue

            # 解析服务端计时（chunk_step 反映服务端 chunk 内位置）
            server_timing = out.get("server_timing", {})
            chunk_step = server_timing.get("chunk_step", -1)
            get_ms = server_timing.get("get_action_ms", 0.0)

            # 兼容旧 serve_policy.py（返回 (H,14) 的 chunk）：取首步
            if action.ndim == 2:
                action = action[0]

            # 14 dims: [left 6D target, left gripper, right 6D target, right gripper].
            # A 6D target is joints or xyz+EulerXYZ according to server metadata.
            act_jl = action[:6]
            act_gl = float(action[6]) * 0.105    # 0~1 → 米
            act_jr = action[7:13]
            act_gr = float(action[13]) * 0.105   # 0~1 → 米

            # ΠiGDM (server) reduces chunk-boundary discontinuity.
            # max_delta caps any residual jump for Piper position control safety.
            if _EXECUTION_ACTION_SPACE == "absolute_end_effector":
                s_jl = smooth_end_pose(
                    last_jl,
                    act_jl,
                    max_position_delta_m=max_ee_position_delta_m,
                    max_rotation_delta_rad=max_ee_rotation_delta_rad,
                    position_filter_alpha=ee_position_filter_alpha,
                    rotation_filter_alpha=ee_rotation_filter_alpha,
                )
                s_jr = smooth_end_pose(
                    last_jr,
                    act_jr,
                    max_position_delta_m=max_ee_position_delta_m,
                    max_rotation_delta_rad=max_ee_rotation_delta_rad,
                    position_filter_alpha=ee_position_filter_alpha,
                    rotation_filter_alpha=ee_rotation_filter_alpha,
                )
            else:
                s_jl = smooth_action(last_jl, act_jl, max_delta=max_joint_delta_rad)
                s_jr = smooth_action(last_jr, act_jr, max_delta=max_joint_delta_rad)
            s_gl = smooth_action(last_gl, np.array([act_gl]), max_delta=0.05)
            s_gr = smooth_action(last_gr, np.array([act_gr]), max_delta=0.05)
            t_action = time.monotonic()

            smooth_rec = {}
            if smoothness_monitor.enabled:
                raw_exec = np.concatenate(
                    [act_jl, np.array([act_gl]), act_jr, np.array([act_gr])],
                    axis=0,
                )
                smooth_exec = np.concatenate([s_jl, s_gl, s_jr, s_gr], axis=0)
                observed_exec = np.concatenate(
                    [
                        snapshot['joint_left'],
                        np.array([float(snapshot['gripper_position'][0])]),
                        snapshot['joint_right'],
                        np.array([float(snapshot['gripper_position'][1])]),
                    ],
                    axis=0,
                )
                smooth_rec = smoothness_monitor.add(
                    step=step,
                    now_s=t_infer,
                    period_ms=period_ms,
                    raw_action=raw_exec,
                    cmd_action=smooth_exec,
                    observed_action=observed_exec,
                    chunk_step=chunk_step,
                    obs_kind=obs_kind,
                    server_timing=server_timing,
                )
            t_metrics = time.monotonic()

            last_jl, last_jr = s_jl.copy(), s_jr.copy()
            last_gl, last_gr = s_gl.copy(), s_gr.copy()

            # 发送动作。CAN 只由 piper_ros 持有；本客户端仅访问 ROS-only TCP 适配服务。
            service_action = np.concatenate([s_jl, s_gl, s_jr, s_gr], axis=0).astype(np.float32)
            service_t0 = time.monotonic()
            service_obs, service_done, service_info = robot_service.step(service_action)
            service_snapshot = robot_service_obs_to_snapshot(service_obs)
            obs_buffer.set_snapshot(service_snapshot)
            t_cmd = time.monotonic()
            if (
                _EXECUTION_ACTION_SPACE == "absolute_end_effector"
                and ee_pose_log_interval_steps > 0
                and step % ee_pose_log_interval_steps == 0
            ):
                print(
                    "[EE POSE] "
                    f"step={step} "
                    f"cmd_left_xyz={np.array2string(s_jl[:3], precision=4, suppress_small=True)} "
                    f"obs_left_xyz={np.array2string(service_snapshot['end_pose_left'][:3], precision=4, suppress_small=True)} "
                    f"cmd_right_xyz={np.array2string(s_jr[:3], precision=4, suppress_small=True)} "
                    f"obs_right_xyz={np.array2string(service_snapshot['end_pose_right'][:3], precision=4, suppress_small=True)}"
                )
            service_ms = (t_cmd - service_t0) * 1000
            cmd_diag = {
                "heartbeat": False,
                "motion_ctrl_ms": 0.0,
                "joint_ms": 0.0,
                "gripper_ms": 0.0,
                "left_total_ms": service_ms,
                "right_total_ms": 0.0,
                "robot_service_ms": service_ms,
                "robot_service_step": int(service_info.get("step", step + 1)),
            }
            if service_done:
                print(f"[系统] RobotArmService 返回 done=True，step={cmd_diag['robot_service_step']}，退出")
                return
            if (
                chunk_step >= 0
                and current_robot_chunk is not None
                and current_chunk_index < len(current_robot_chunk)
            ):
                current_chunk_index += 1

            # 定期保存调试图像（每 50 步一次，避免 50Hz 写盘压力）
            t_save_start = time.monotonic()
            if save_debug_images and step % 50 == 0:
                for cam in ('cam_top', 'cam_left_wrist', 'cam_right_wrist'):
                    img = snapshot['images'].get(cam)
                    if img is not None:
                        path = str(save_dir / f"step{step:06d}_{cam}.jpg")
                        cv2.imwrite(path, cv2.cvtColor(
                            np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR
                        ))
            t_save = time.monotonic()

            # 日志
            ms_snap  = (t_snap  - t_start) * 1000  # obs_buffer snapshot
            ms_obs   = (t_obs   - t_snap)  * 1000  # make_agilex_observation
            ms_infer = (t_infer - t_obs)   * 1000  # WebSocket round trip
            ms_action = (t_action - t_infer) * 1000  # action parse / smoothing
            ms_metrics = (t_metrics - t_action) * 1000  # smoothness metrics
            ms_cmd   = (t_cmd   - t_metrics) * 1000  # CAN send both arms
            ms_save  = (t_save  - t_save_start) * 1000
            elapsed = t_save - t_start
            q_remaining = server_timing.get("queue_remaining", -1)
            infer_active = bool(server_timing.get("inference_active", False))
            async_ws_ms = fdict_ms(server_timing, "async_ws_ms")
            async_q = server_timing.get("async_queue_len_after", -1)
            async_age_ms = fdict_ms(server_timing, "async_response_age_ms")
            stale_flag = "Y" if bool(server_timing.get("action_stale", False)) else "n"
            heartbeat_flag = "Y" if cmd_diag.get("heartbeat") else "n"
            smooth_tag = ""
            if smooth_rec:
                smooth_tag = (
                    f" sm_dq={smooth_rec['cmd_delta_joint_l2']:.3f}"
                    f" jerk={smooth_rec['joint_jerk_rms']:.1f}"
                )
            boundary_pre = fdict_ms(server_timing, "boundary_jump_pre")
            boundary_post = fdict_ms(server_timing, "boundary_jump_post")
            boundary_blend_len = server_timing.get("boundary_blend_len", 0)
            if boundary_pre > 0.0 or boundary_post > 0.0:
                smooth_tag += (
                    f" bj={boundary_pre:.3f}->{boundary_post:.3f}"
                    f"/{boundary_blend_len}"
                )
            t_log_start = time.monotonic()
            do_full_log = (
                (startup_full_log_steps > 0 and step < startup_full_log_steps)
                or (full_log_interval_steps > 0 and step % full_log_interval_steps == 0)
            )
            do_short_log = (
                control_log_interval_steps > 0
                and step % control_log_interval_steps == 0
            )
            if do_full_log:
                # 完整诊断：前10步 + 每50步
                np.set_printoptions(precision=4, suppress=True, linewidth=120)
                delta_jl = s_jl - snapshot['joint_left']
                delta_jr = s_jr - snapshot['joint_right']
                print(
                    f"\n[步 {step:5d}] chunk_step={chunk_step}  "
                    f"get={get_ms:.1f}ms  loop={elapsed*1000:.1f}ms  "
                    f"[tx={obs_kind} snap={ms_snap:.1f} obs={ms_obs:.1f} "
                    f"ws={ms_infer:.1f} aws={async_ws_ms:.1f} act={ms_action:.1f} met={ms_metrics:.1f} "
                    f"cmd={ms_cmd:.1f} save={ms_save:.1f}]"
                    f" q={q_remaining} active={int(infer_active)} hb={heartbeat_flag} "
                    f"payload={payload_kib:.0f}KiB aq={async_q} age={async_age_ms:.1f}ms stale={stale_flag} "
                    f"period={period_ms:.1f}ms{smooth_tag}"
                    f"\n  obs.state  left ={snapshot['joint_left']}"
                    f"\n             right={snapshot['joint_right']}"
                    f"\n  obs.gripper={snapshot['gripper_position']}"
                    f"\n  raw_action ={action}"
                    f"\n  smooth jl  ={s_jl}  gl={float(s_gl[0]):.4f}m"
                    f"\n  smooth jr  ={s_jr}  gr={float(s_gr[0]):.4f}m"
                    f"\n  Δjl        ={delta_jl}"
                    f"\n  Δjr        ={delta_jr}"
                )
            elif do_short_log:
                print(
                    f"  [步 {step:5d}] chunk={chunk_step:3d} "
                    f"get={get_ms:5.1f}ms  exec={elapsed*1000:5.1f}ms"
                    f"[tx={obs_kind} obs={ms_obs:.0f} ws={ms_infer:.0f} aws={async_ws_ms:.0f} "
                    f"cmd={ms_cmd:.0f} q={q_remaining} aq={async_q} stale={stale_flag} hb={heartbeat_flag}] | "
                    f"L1={s_jl[0]:+.3f} GL={float(s_gl[0]):.3f}m | "
                    f"R1={s_jr[0]:+.3f} GR={float(s_gr[0]):.3f}m |{smooth_tag}"
                )
            if smoothness_monitor.should_print(step):
                smooth_line = smoothness_monitor.format_online(smooth_rec)
                if smooth_line:
                    print(smooth_line)
            smoothness_monitor.maybe_save_outputs(force=False)
            ms_log = (time.monotonic() - t_log_start) * 1000
            total_elapsed = time.monotonic() - t_start
            sleep_t = dt - total_elapsed
            sleep_request_ms = max(0.0, sleep_t * 1000)
            if timing_monitor.enabled:
                reason = classify_stutter(
                    obs_kind=obs_kind,
                    period_ms=period_ms,
                    loop_ms=total_elapsed * 1000,
                    ms_snap=ms_snap,
                    ms_obs=ms_obs,
                    ms_ws=ms_infer,
                    ms_action=ms_action,
                    ms_metrics=ms_metrics,
                    ms_cmd=ms_cmd,
                    ms_save=ms_save,
                    ms_log=ms_log,
                    prev_log_ms=prev_log_ms,
                    server_timing=server_timing,
                    cmd_diag=cmd_diag,
                    threshold_ms=stutter_warn_ms,
                )
                timing_monitor.add(
                    step=step,
                    obs_kind=obs_kind,
                    chunk_step=chunk_step,
                    period_ms=period_ms,
                    loop_ms=elapsed * 1000,
                    total_ms=total_elapsed * 1000,
                    sleep_request_ms=sleep_request_ms,
                    snap_ms=ms_snap,
                    obs_ms=ms_obs,
                    ws_ms=ms_infer,
                    action_ms=ms_action,
                    metrics_ms=ms_metrics,
                    cmd_ms=ms_cmd,
                    save_ms=ms_save,
                    log_ms=ms_log,
                    payload_kib=payload_kib,
                    server_timing=server_timing,
                    cmd_diag=cmd_diag,
                    reason=reason,
                )
                if timing_monitor.should_print(step):
                    timing_line = timing_monitor.format_online()
                    if timing_line:
                        print(timing_line)

            if stutter_diagnostics and (
                total_elapsed * 1000 > stutter_warn_ms or period_ms > stutter_warn_ms
            ):
                reason = classify_stutter(
                    obs_kind=obs_kind,
                    period_ms=period_ms,
                    loop_ms=total_elapsed * 1000,
                    ms_snap=ms_snap,
                    ms_obs=ms_obs,
                    ms_ws=ms_infer,
                    ms_action=ms_action,
                    ms_metrics=ms_metrics,
                    ms_cmd=ms_cmd,
                    ms_save=ms_save,
                    ms_log=ms_log,
                    prev_log_ms=prev_log_ms,
                    server_timing=server_timing,
                    cmd_diag=cmd_diag,
                    threshold_ms=stutter_warn_ms,
                )
                server_prepare_ms = fdict_ms(server_timing, "server_prepare_ms")
                ws_extra_ms = max(0.0, ms_infer - server_prepare_ms)
                print(
                    f"  [STUTTER] step={step} reason={reason} "
                    f"period={period_ms:.1f}ms exec={elapsed*1000:.1f}ms "
                    f"total={total_elapsed*1000:.1f}ms tx={obs_kind} payload={payload_kib:.0f}KiB "
                    f"segments snap/obs/ws/action/metrics/cmd/save/log="
                    f"{ms_snap:.1f}/{ms_obs:.1f}/{ms_infer:.1f}/{ms_action:.1f}/"
                    f"{ms_metrics:.1f}/{ms_cmd:.1f}/{ms_save:.1f}/{ms_log:.1f}ms "
                    f"ws_extra~{ws_extra_ms:.1f}ms "
                    f"server get/unpack/prepare/prev_total="
                    f"{get_ms:.1f}/{fdict_ms(server_timing, 'server_unpack_ms'):.1f}/"
                    f"{server_prepare_ms:.1f}/{fdict_ms(server_timing, 'prev_total_ms'):.1f}ms "
                    f"queue before/get/after/rem="
                    f"{server_timing.get('queue_before_request', '?')}/"
                    f"{server_timing.get('queue_before_get', '?')}/"
                    f"{server_timing.get('queue_after_get', '?')}/"
                    f"{q_remaining} "
                    f"infer_active={int(infer_active)} underrun={int(bool(server_timing.get('queue_underrun')))} "
                    f"cmd L/R={cmd_diag['left_total_ms']:.1f}/{cmd_diag['right_total_ms']:.1f}ms "
                    f"joint/grip/motion={cmd_diag['joint_ms']:.1f}/{cmd_diag['gripper_ms']:.1f}/"
                    f"{cmd_diag['motion_ctrl_ms']:.1f}ms prev_log={prev_log_ms:.1f}ms "
                    f"sleep_req={sleep_request_ms:.1f}ms"
                )
            prev_log_ms = ms_log

            step += 1
            maybe_request_next_chunk(step)

            # 精确控制循环频率
            sleep_t = dt - total_elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif (
                total_elapsed > dt * 1.5
                and overrun_warn_interval_steps > 0
                and step % overrun_warn_interval_steps == 0
            ):
                print(f"  [警告] 控制循环超时: {total_elapsed*1000:.1f}ms > {dt*1000:.1f}ms")

    try:
        rtc_loop()
    except KeyboardInterrupt:
        print("\n\n[系统] 用户中断")
    finally:
        try:
            print("\n[SMOOTH] final trajectory diagnostics before go_zero ...")
            smoothness_monitor.print_final_summary()
            if bool(getattr(args, 'smoothness_save_on_exit', False)):
                smoothness_monitor.maybe_save_outputs(force=True)
        except Exception as e:
            print(f"[SMOOTH] final diagnostics failed before go_zero: {e}")

        try:
            print("\n[TIMING] saving timing diagnostics before go_zero ...")
            timing_monitor.print_final_summary()
            timing_monitor.save_csv()
        except Exception as e:
            print(f"[TIMING] save failed before go_zero: {e}")

        if chunk_planner is not None:
            print("\n[CHUNK] stopping chunk request thread ...")
            chunk_planner.stop()

        if robot_service is not None:
            print("\n[RobotService] stop：由 start_robot_arm_service.py 回零 ...")
            robot_service.stop()
            robot_service.close()
        print("[完成]")


# =====================================================
#              参数解析
# =====================================================

def load_yaml_config(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("未安装 PyYAML: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with p.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是 key-value 字典")
    # 支持可选命名空间（兼容 online_inference_execution.py 的 YAML）
    for ns in ('rtc_online_inference', 'online_inference_execution'):
        if ns in data and isinstance(data[ns], dict):
            data = data[ns]
            break
    return data


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='RTC 在线机械臂推理执行脚本 (配合 serve_rtc_policy.py)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python3 paper_rtc_online_inference_execution.py \\
    --host 127.0.0.1 --port 8001 \\
    --prompt "pick up the cube" \\
    --control_hz 30.0

  # YAML 配置
  python3 paper_rtc_online_inference_execution.py --config paper_rtc_online_inference.yaml
""",
    )

    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径（命令行参数会覆盖同名配置）')

    # 服务端连接
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='RTC 服务端地址 (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8001,
                        help='RTC 服务端端口 (default: 8001)')

    # 任务
    parser.add_argument('--prompt', type=str, default='do something',
                        help='任务提示词')

    # RTC 控制频率
    parser.add_argument('--control_hz', type=float, default=30.0,
                        help='控制循环频率 Hz，应与服务端 --control-hz 一致 (default: 30.0)')
    parser.add_argument('--max_joint_delta_rad', type=float, default=0.20,
                        help='每个控制步单关节最大目标变化量 rad (default: 0.20)')
    parser.add_argument('--max_ee_position_delta_m', type=float, default=0.01,
                        help='每个控制步末端单轴最大位置变化量 m (default: 0.01)')
    parser.add_argument('--max_ee_rotation_delta_rad', type=float, default=0.0872665,
                        help='每个控制步末端单轴最大姿态变化量 rad (default: 5 degrees)')
    parser.add_argument('--ee_position_filter_alpha', type=float, default=0.2,
                        help='末端位置低通系数，越小越平滑 (default: 0.2)')
    parser.add_argument('--ee_rotation_filter_alpha', type=float, default=0.2,
                        help='末端四元数 SLERP 系数，越小越平滑 (default: 0.2)')
    parser.add_argument('--ee_pose_log_interval_steps', type=int, default=30,
                        help='末端模式每 N 步打印目标/实测 XYZ；<=0 关闭 (default: 30)')
    parser.add_argument('--obs_send_hz', type=float, default=10.0,
                        help='Legacy full-observation frequency; client chunk mode requests full obs per chunk')
    parser.add_argument('--client_chunk_execution', action=argparse.BooleanOptionalAction, default=True,
                        help='Required: request full Paper-RTC chunks and execute them locally')
    parser.add_argument('--client_chunk_execute_horizon', type=int, default=0,
                        help='Minimum executed chunk steps s before replan; 0 uses server execution_horizon')
    parser.add_argument('--client_chunk_fixed_delay_steps', type=int, default=5,
                        help='Fixed client-side delay d, or warmup fallback for realtime_ceil mode')
    parser.add_argument('--client_chunk_delay_mode', type=str, default='fixed', choices=('fixed', 'realtime_ceil'),
                        help='Client delay mode: fixed or realtime_ceil based on previous request latency')
    parser.add_argument('--client_chunk_max_delay_steps', type=int, default=0,
                        help='Maximum client delay steps; <=0 caps at action_horizon - 1')
    parser.add_argument('--async_first_action_timeout_s', type=float, default=5.0,
                        help='Legacy; startup init hold now waits for the first chunk')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='最大执行步数，0 表示无限循环 (default: 0)')

    # 硬件后端：Paper-RTC 不直接打开 CAN，只连接 ROS-only 适配服务。
    parser.add_argument('--robot_endpoint', type=str, default='tcp://127.0.0.1:9901',
                        help='ROS-only robot control service endpoint')
    parser.add_argument('--robot_service_recv_timeout_ms', type=int, default=30000,
                        help='TCP receive timeout for the ROS-only robot control service')
    parser.add_argument('--startup_init_hold_min_s', type=float, default=0.0,
                        help='Minimum seconds to keep sending init pose before the first chunk is executed')
    parser.add_argument('--startup_init_hold_ramp_s', type=float, default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument('--startup_init_hold_timeout_s', type=float, default=0.0,
                        help='Max seconds to wait for first chunk while holding init pose; <=0 waits indefinitely')
    parser.add_argument('--startup_init_hold_log_interval_s', type=float, default=1.0,
                        help='Seconds between startup-hold progress logs')

    # 调试
    parser.add_argument('--save_image_dir', type=str,
                        default='/tmp/rtc_debug_images',
                        help='调试图像保存目录 (default: /tmp/rtc_debug_images)')
    parser.add_argument('--save_debug_images', action='store_true',
                        help='Enable periodic cv2.imwrite debug images; disabled by default')
    parser.add_argument('--stutter_warn_ms', type=float, default=0.0,
                        help='Print STUTTER diagnostics when loop/period exceeds this many ms; <=0 uses 1.5x control dt')
    parser.add_argument('--stutter_diagnostics', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable detailed intermittent-stutter diagnosis logs (default: true)')
    parser.add_argument('--execution_diagnostics', action=argparse.BooleanOptionalAction, default=False,
                        help='Reserved execution diagnostics switch; disabled by default')
    parser.add_argument('--chunk_diagnostics', action=argparse.BooleanOptionalAction, default=False,
                        help='Reserved chunk diagnostics switch; disabled by default')
    parser.add_argument('--control_log_interval_steps', type=int, default=300,
                        help='Print compact control log every N steps; <=0 disables')
    parser.add_argument('--full_log_interval_steps', type=int, default=0,
                        help='Print full array diagnostics every N steps; <=0 disables')
    parser.add_argument('--startup_full_log_steps', type=int, default=0,
                        help='Print full array diagnostics for the first N steps; <=0 disables')
    parser.add_argument('--overrun_warn_interval_steps', type=int, default=300,
                        help='Print overrun warning at most every N steps; <=0 disables')
    parser.add_argument('--timing_diagnostics', action=argparse.BooleanOptionalAction, default=True,
                        help='Record lightweight control-loop timing bottleneck metrics (default: true)')
    parser.add_argument('--timing_history_limit', type=int, default=9000,
                        help='Maximum timing samples kept in memory and written to CSV')
    parser.add_argument('--timing_window_steps', type=int, default=90,
                        help='Rolling window size for online timing summaries')
    parser.add_argument('--timing_print_interval_steps', type=int, default=90,
                        help='Print online timing metrics every N control steps')
    parser.add_argument('--timing_save_csv', action=argparse.BooleanOptionalAction, default=True,
                        help='Save timing_bottleneck_metrics.csv on exit (default: true)')
    parser.add_argument('--smoothness_diagnostics', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable trajectory smoothness metrics (default: true)')
    parser.add_argument('--smoothness_plot_dir', type=str, default='/tmp/paper_rtc_smoothness',
                        help='Directory for smoothness CSV and PNG plots')
    parser.add_argument('--smoothness_history_limit', type=int, default=9000,
                        help='Maximum smoothness samples kept in memory and written to CSV')
    parser.add_argument('--smoothness_window_steps', type=int, default=60,
                        help='Rolling window size for online smoothness summaries')
    parser.add_argument('--smoothness_print_interval_steps', type=int, default=30,
                        help='Print online smoothness metrics every N control steps')
    parser.add_argument('--smoothness_plot_interval_s', type=float, default=0.0,
                        help='Save smoothness plots every N seconds; <=0 disables periodic saves')
    parser.add_argument('--smoothness_save_on_exit', action=argparse.BooleanOptionalAction, default=False,
                        help='Force-save smoothness CSV/plots on Ctrl+C/exit (default: false)')
    parser.add_argument('--smoothness_save_csv', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable smoothness_metrics.csv when smoothness saving is requested (default: true)')
    parser.add_argument('--smoothness_save_plots', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable smoothness PNG plots when smoothness saving is requested (default: true)')

    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = create_arg_parser()

    if pre_args.config:
        try:
            parser.set_defaults(**load_yaml_config(pre_args.config))
        except Exception as e:
            print(f"[错误] 读取 YAML 失败: {e}")
            sys.exit(1)

    args = parser.parse_args()
    run_rtc_inference(args)


if __name__ == '__main__':
    main()
