#!/usr/bin/env python3
"""ROS-only observation and dual-arm control adapter.

The TCP protocol is compatible with the existing online inference client:
4-byte big-endian payload length followed by msgpack-numpy data.  Robot
observations are read from ROS topics and commands are published as either
``sensor_msgs/msg/JointState`` or ``piper_msgs/msg/PosCmd``. This process never
opens a hardware device.
"""

from __future__ import annotations

import argparse
import math
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import cv_bridge
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Pose
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState


OPENPI_ROOT = Path(__file__).resolve().parents[2]
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

from openpi_client import msgpack_numpy  # noqa: E402


DEFAULT_INIT_JOINTS_LEFT = np.array(
    [-0.47566299, 0.12493393, -0.49851463, 0.09187755, 0.84711553, -0.11246147],
    dtype=np.float32,
)
DEFAULT_INIT_JOINTS_RIGHT = np.array(
    [0.05784430, 0.45746890, -0.47620376, 0.25457774, 0.83889940, -0.19685554],
    dtype=np.float32,
)
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
MAX_GRIPPER_M = 0.070
DEFAULT_INIT_GRIPPER_LEFT_M = 0.000
DEFAULT_INIT_GRIPPER_RIGHT_M = 0.060
MAX_FRAME_BYTES = 256 * 1024 * 1024


def _tcp_send(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)


def _tcp_recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("connection closed")
        data.extend(chunk)
    return bytes(data)


def _tcp_recv(sock: socket.socket) -> bytes:
    (size,) = struct.unpack(">I", _tcp_recv_exact(sock, 4))
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise ValueError(f"invalid TCP frame length: {size}")
    return _tcp_recv_exact(sock, size)


def _validate_vector(values: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinity")
    return array


class ObservationBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.images: dict[str, np.ndarray | None] = {
            "cam_top": None,
            "cam_left_wrist": None,
            "cam_right_wrist": None,
        }
        self.joint_left = np.zeros(6, dtype=np.float32)
        self.joint_right = np.zeros(6, dtype=np.float32)
        self.end_pose_left = np.zeros(6, dtype=np.float32)
        self.end_pose_right = np.zeros(6, dtype=np.float32)
        self.gripper_position = np.zeros(2, dtype=np.float32)
        self._updated_at: dict[str, float] = {
            "cam_top": 0.0,
            "cam_left_wrist": 0.0,
            "cam_right_wrist": 0.0,
            "joint_left": 0.0,
            "joint_right": 0.0,
            "end_pose_left": 0.0,
            "end_pose_right": 0.0,
        }

    def update_image(self, camera: str, image_hwc: np.ndarray) -> None:
        array = np.asarray(image_hwc)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{camera} expected HWC RGB image, got {array.shape}")
        image_chw = np.transpose(array, (2, 0, 1)).astype(np.uint8, copy=True)
        with self._lock:
            self.images[camera] = image_chw
            self._updated_at[camera] = time.monotonic()

    def update_joint(self, side: str, position: Any) -> None:
        values = np.asarray(position, dtype=np.float32)
        if values.ndim != 1 or values.size < 7:
            raise ValueError(f"{side} JointState.position requires 7 values, got {values.shape}")
        if not np.all(np.isfinite(values[:7])):
            raise ValueError(f"{side} JointState.position contains NaN or infinity")
        with self._lock:
            if side == "left":
                self.joint_left = values[:6].copy()
                self.gripper_position[0] = float(values[6])
            else:
                self.joint_right = values[:6].copy()
                self.gripper_position[1] = float(values[6])
            self._updated_at[f"joint_{side}"] = time.monotonic()

    def update_end_pose(self, side: str, pose: Pose) -> None:
        quaternion = np.asarray(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"{side} end-pose quaternion is invalid")
        x, y, z, w = quaternion / norm
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        values = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z, roll, pitch, yaw],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{side} end pose contains NaN or infinity")
        with self._lock:
            if side == "left":
                self.end_pose_left = values
            else:
                self.end_pose_right = values
            self._updated_at[f"end_pose_{side}"] = time.monotonic()

    def ready(self) -> bool:
        with self._lock:
            return all(timestamp > 0.0 for timestamp in self._updated_at.values())

    def missing(self) -> list[str]:
        with self._lock:
            return [name for name, timestamp in self._updated_at.items() if timestamp <= 0.0]

    def stale(self, threshold_s: float) -> list[str]:
        if threshold_s <= 0.0:
            return []
        now = time.monotonic()
        with self._lock:
            return [
                name
                for name, timestamp in self._updated_at.items()
                if timestamp <= 0.0 or now - timestamp > threshold_s
            ]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "images": {
                    name: None if image is None else image.copy()
                    for name, image in self.images.items()
                },
                "joint_left": self.joint_left.copy(),
                "joint_right": self.joint_right.copy(),
                "end_pose_left": self.end_pose_left.copy(),
                "end_pose_right": self.end_pose_right.copy(),
                "gripper_position": self.gripper_position.copy(),
            }


class RosOnlyControlNode(Node):
    def __init__(self, observations: ObservationBuffer) -> None:
        super().__init__("ros_only_robot_control_service")
        self._observations = observations
        self._bridge = cv_bridge.CvBridge()

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        arm_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Image,
            "/camera/top/camera/color/image_raw",
            lambda message: self._on_image(message, "cam_top"),
            camera_qos,
        )
        self.create_subscription(
            Image,
            "/camera/left/camera/color/image_raw",
            lambda message: self._on_image(message, "cam_left_wrist"),
            camera_qos,
        )
        self.create_subscription(
            Image,
            "/camera/right/camera/color/image_raw",
            lambda message: self._on_image(message, "cam_right_wrist"),
            camera_qos,
        )
        self.create_subscription(
            JointState,
            "/joint_states_left",
            lambda message: self._on_joint(message, "left"),
            arm_qos,
        )
        self.create_subscription(
            JointState,
            "/joint_states_right",
            lambda message: self._on_joint(message, "right"),
            arm_qos,
        )
        self.create_subscription(
            Pose,
            "/end_pose_left",
            lambda message: self._on_end_pose(message, "left"),
            arm_qos,
        )
        self.create_subscription(
            Pose,
            "/end_pose_right",
            lambda message: self._on_end_pose(message, "right"),
            arm_qos,
        )

        self._left_command = self.create_publisher(
            JointState, "/joint_ctrl_cmd_left", command_qos
        )
        self._right_command = self.create_publisher(
            JointState, "/joint_ctrl_cmd_right", command_qos
        )
        self._left_end_command = self.create_publisher(PosCmd, "/pos_cmd_left", command_qos)
        self._right_end_command = self.create_publisher(PosCmd, "/pos_cmd_right", command_qos)

        self._command_topics = {
            "left": ("/joint_ctrl_cmd_left", "piper_left_ctrl_node"),
            "right": ("/joint_ctrl_cmd_right", "piper_right_ctrl_node"),
        }
        self._end_command_topics = {
            "left": ("/pos_cmd_left", "piper_left_ctrl_node"),
            "right": ("/pos_cmd_right", "piper_right_ctrl_node"),
        }

    def _on_image(self, message: Image, camera: str) -> None:
        try:
            rgb = self._bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
            self._observations.update_image(camera, rgb)
        except Exception as exc:
            self.get_logger().error(f"image decode failed [{camera}]: {exc}")

    def _on_joint(self, message: JointState, side: str) -> None:
        try:
            self._observations.update_joint(side, message.position)
        except Exception as exc:
            self.get_logger().error(f"joint decode failed [{side}]: {exc}")

    def _on_end_pose(self, message: Pose, side: str) -> None:
        try:
            self._observations.update_end_pose(side, message)
        except Exception as exc:
            self.get_logger().error(f"end-pose decode failed [{side}]: {exc}")

    def controller_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for side, (topic, expected_node) in self._command_topics.items():
            endpoints = self.get_subscriptions_info_by_topic(topic)
            counts[side] = sum(
                endpoint.node_name == expected_node
                and endpoint.node_namespace == "/"
                for endpoint in endpoints
            )
        return counts

    def command_subscription_totals(self) -> dict[str, int]:
        return {
            "left": self._left_command.get_subscription_count(),
            "right": self._right_command.get_subscription_count(),
        }

    def end_controller_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for side, (topic, expected_node) in self._end_command_topics.items():
            endpoints = self.get_subscriptions_info_by_topic(topic)
            counts[side] = sum(
                endpoint.node_name == expected_node
                and endpoint.node_namespace == "/"
                for endpoint in endpoints
            )
        return counts

    def end_command_subscription_totals(self) -> dict[str, int]:
        return {
            "left": self._left_end_command.get_subscription_count(),
            "right": self._right_end_command.get_subscription_count(),
        }

    def require_one_controller_per_side(self) -> None:
        counts = self.controller_counts()
        if counts != {"left": 1, "right": 1}:
            raise RuntimeError(
                "expected exactly one piper_ros command subscriber per side; "
                f"observed {counts}"
            )

    def publish_pair(
        self,
        left_joints: Any,
        left_gripper_m: float,
        right_joints: Any,
        right_gripper_m: float,
        speed_pct: int,
    ) -> None:
        self.require_one_controller_per_side()
        left = self._make_command(left_joints, left_gripper_m, speed_pct)
        right = self._make_command(right_joints, right_gripper_m, speed_pct)
        self._left_command.publish(left)
        self._right_command.publish(right)

    def publish_end_pair(
        self,
        left_pose: Any,
        left_gripper_m: float,
        right_pose: Any,
        right_gripper_m: float,
    ) -> None:
        counts = self.end_controller_counts()
        if counts != {"left": 1, "right": 1}:
            raise RuntimeError(
                "expected exactly one piper_ros end-pose subscriber per side; "
                f"observed {counts}"
            )
        self._left_end_command.publish(self._make_end_command(left_pose, left_gripper_m))
        self._right_end_command.publish(self._make_end_command(right_pose, right_gripper_m))

    @staticmethod
    def _make_end_command(pose: Any, gripper_m: float) -> PosCmd:
        values = _validate_vector(pose, 6, "end-effector pose")
        gripper = float(gripper_m)
        if not np.isfinite(gripper):
            raise ValueError("gripper command contains NaN or infinity")
        message = PosCmd()
        message.x, message.y, message.z = (float(value) for value in values[:3])
        message.roll, message.pitch, message.yaw = (float(value) for value in values[3:6])
        message.gripper = float(np.clip(gripper, 0.0, MAX_GRIPPER_M))
        message.mode1 = 0
        message.mode2 = 0
        return message

    def _make_command(self, joints: Any, gripper_m: float, speed_pct: int) -> JointState:
        joint_array = _validate_vector(joints, 6, "joint command")
        gripper = float(gripper_m)
        if not np.isfinite(gripper):
            raise ValueError("gripper command contains NaN or infinity")

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [float(value) for value in joint_array]
        message.position.append(float(np.clip(gripper, 0.0, MAX_GRIPPER_M)))
        message.velocity = [0.0] * 6 + [float(np.clip(speed_pct, 1, 100))]
        message.effort = [0.0] * 6 + [1.0]
        return message


class RosOnlyRobotControlService:
    def __init__(
        self,
        *,
        prompt: str,
        max_steps: int,
        topic_wait_timeout: float,
        stale_observation_timeout: float,
        init_ramp_s: float,
        init_ramp_hz: float,
        init_hold_s: float,
        init_speed_pct: int,
        speed_pct: int,
    ) -> None:
        self._prompt = prompt
        self._max_steps = max_steps
        self._topic_wait_timeout = topic_wait_timeout
        self._stale_observation_timeout = stale_observation_timeout
        self._init_ramp_s = init_ramp_s
        self._init_ramp_hz = init_ramp_hz
        self._init_hold_s = init_hold_s
        self._init_speed_pct = init_speed_pct
        self._speed_pct = speed_pct

        self._observations = ObservationBuffer()
        self._node: RosOnlyControlNode | None = None
        self._spin_thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._step_count = 0

    def serve_forever(self, host: str, port: int) -> None:
        self._setup_ros()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.settimeout(0.5)
        print(f"[ROS-only service] listening on {host}:{port}", flush=True)

        packer = msgpack_numpy.Packer()
        try:
            while not self._shutdown.is_set():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                print(f"[ROS-only service] client connected from {address}", flush=True)
                with connection:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    try:
                        self._serve_connection(connection, packer)
                    except (EOFError, ConnectionError, OSError) as exc:
                        print(f"[ROS-only service] connection ended: {exc}", flush=True)
                print("[ROS-only service] client disconnected", flush=True)
        finally:
            server.close()
            self._shutdown_ros()

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def _serve_connection(self, connection: socket.socket, packer: Any) -> None:
        while not self._shutdown.is_set():
            message = msgpack_numpy.unpackb(_tcp_recv(connection))
            try:
                reply = self._handle(message)
            except Exception as exc:
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            _tcp_send(connection, packer.pack(reply))

    def _handle(self, message: Any) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise TypeError("request must be a dictionary")
        command = message.get("cmd")
        if command == "ping":
            assert self._node is not None
            return {
                "pong": True,
                "transport": "ros_topics_only",
                "piper_controller_subscriptions": self._node.controller_counts(),
                "command_subscription_totals": self._node.command_subscription_totals(),
                "end_effector_controller_subscriptions": self._node.end_controller_counts(),
                "end_effector_command_subscription_totals": self._node.end_command_subscription_totals(),
                "supported_action_spaces": ["absolute_joint", "absolute_end_effector"],
            }
        if command == "reset":
            return {"obs": self._reset()}
        if command == "observe":
            self._require_fresh_observations()
            return {"obs": self._observation()}
        if command == "step":
            observation, done, info = self._step(
                message.get("action"),
                str(message.get("action_space", "absolute_joint")),
            )
            return {"obs": observation, "done": done, "info": info}
        if command == "stop":
            self._go_zero()
            self._step_count = 0
            return {"status": "ok"}
        raise ValueError(f"unknown command: {command!r}")

    def _reset(self) -> dict[str, Any]:
        assert self._node is not None
        self._require_fresh_observations()
        self._step_count = 0
        snapshot = self._observations.snapshot()
        start_left = snapshot["joint_left"]
        start_right = snapshot["joint_right"]
        start_gripper = snapshot["gripper_position"]

        frequency = max(1.0, float(self._init_ramp_hz))
        duration = max(0.1, float(self._init_ramp_s))
        steps = max(2, int(round(duration * frequency)))
        period = 1.0 / frequency
        deadline = time.monotonic()
        print(
            f"[ROS-only service] smooth reset {duration:.2f}s @ {frequency:.1f}Hz",
            flush=True,
        )
        for index in range(steps + 1):
            x = index / steps
            alpha = x * x * x * (10.0 + x * (-15.0 + 6.0 * x))
            left = (1.0 - alpha) * start_left + alpha * DEFAULT_INIT_JOINTS_LEFT
            right = (1.0 - alpha) * start_right + alpha * DEFAULT_INIT_JOINTS_RIGHT
            left_gripper = (
                (1.0 - alpha) * float(start_gripper[0])
                + alpha * DEFAULT_INIT_GRIPPER_LEFT_M
            )
            right_gripper = (
                (1.0 - alpha) * float(start_gripper[1])
                + alpha * DEFAULT_INIT_GRIPPER_RIGHT_M
            )
            self._node.publish_pair(
                left,
                left_gripper,
                right,
                right_gripper,
                self._init_speed_pct,
            )
            deadline += period
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
        if self._init_hold_s > 0.0:
            time.sleep(self._init_hold_s)
        return self._observation()

    def _step(self, action: Any, action_space: str) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        assert self._node is not None
        command = _validate_vector(action, 14, "action")
        self._require_fresh_observations()
        if action_space == "absolute_joint":
            self._node.publish_pair(
                command[:6],
                float(command[6]),
                command[7:13],
                float(command[13]),
                self._speed_pct,
            )
        elif action_space == "absolute_end_effector":
            self._node.publish_end_pair(
                command[:6],
                float(command[6]),
                command[7:13],
                float(command[13]),
            )
        else:
            raise ValueError(f"unsupported action_space: {action_space!r}")
        self._step_count += 1
        done = self._max_steps > 0 and self._step_count >= self._max_steps
        return self._observation(), done, {
            "step": self._step_count,
            "transport": "ros_topics_only",
            "action_space": action_space,
            "piper_controller_subscriptions": self._node.controller_counts(),
            "command_subscription_totals": self._node.command_subscription_totals(),
        }

    def _go_zero(self) -> None:
        assert self._node is not None
        print("[ROS-only service] stop request: publish zero pose", flush=True)
        self._node.publish_pair(np.zeros(6), 0.0, np.zeros(6), 0.0, 30)
        time.sleep(2.0)

    def _observation(self) -> dict[str, Any]:
        snapshot = self._observations.snapshot()
        return {
            "state": np.concatenate([snapshot["joint_left"], snapshot["joint_right"]]),
            "end_effector_state": np.concatenate(
                [snapshot["end_pose_left"], snapshot["end_pose_right"]]
            ),
            "gripper_position": snapshot["gripper_position"],
            "images": snapshot["images"],
            "prompt": self._prompt,
        }

    def _require_fresh_observations(self) -> None:
        stale = self._observations.stale(self._stale_observation_timeout)
        if stale:
            raise RuntimeError(f"stale ROS observations: {stale}")

    def _setup_ros(self) -> None:
        rclpy.init(args=None)
        self._node = RosOnlyControlNode(self._observations)
        deadline = time.monotonic() + self._topic_wait_timeout
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)
            counts = self._node.controller_counts()
            if counts["left"] > 1 or counts["right"] > 1:
                raise RuntimeError(f"duplicate piper_ros command subscribers detected: {counts}")
            if self._observations.ready() and counts == {"left": 1, "right": 1}:
                break
            if self._topic_wait_timeout > 0.0 and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ROS readiness timeout: missing={self._observations.missing()}, "
                    f"controller_subscriptions={counts}"
                )
        self._spin_thread = threading.Thread(target=self._spin, name="ros-only-spin", daemon=True)
        self._spin_thread.start()
        print(
            "[ROS-only service] READY: observations live; one controller per side",
            flush=True,
        )

    def _spin(self) -> None:
        assert self._node is not None
        try:
            while not self._shutdown.is_set() and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.01)
        except (ExternalShutdownException, RCLError):
            if not self._shutdown.is_set() and rclpy.ok():
                raise

    def _shutdown_ros(self) -> None:
        self._shutdown.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS-only TCP adapter for Training Paper RTC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9901)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--topic-wait-timeout", type=float, default=45.0)
    parser.add_argument("--stale-observation-timeout", type=float, default=2.0)
    parser.add_argument("--init-ramp-s", type=float, default=4.0)
    parser.add_argument("--init-ramp-hz", type=float, default=20.0)
    parser.add_argument("--init-hold-s", type=float, default=0.5)
    parser.add_argument("--init-speed-pct", type=int, default=10)
    parser.add_argument("--speed-pct", type=int, default=100)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    service = RosOnlyRobotControlService(
        prompt=args.prompt,
        max_steps=args.max_steps,
        topic_wait_timeout=args.topic_wait_timeout,
        stale_observation_timeout=args.stale_observation_timeout,
        init_ramp_s=args.init_ramp_s,
        init_ramp_hz=args.init_ramp_hz,
        init_hold_s=args.init_hold_s,
        init_speed_pct=args.init_speed_pct,
        speed_pct=args.speed_pct,
    )

    def request_shutdown(signum: int, _frame: Any) -> None:
        print(f"[ROS-only service] received signal {signum}", flush=True)
        service.request_shutdown()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    service.serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
