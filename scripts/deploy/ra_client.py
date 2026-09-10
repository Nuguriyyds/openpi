"""Training-paper RTC AgileX client for the ROS-only supervisor.

The training-paper RTC server intentionally speaks the same client chunk protocol
as ``paper_rtc_online_inference_execution.py``:

* request ``_paper_rtc_client_chunk_request=True``;
* send ``_paper_rtc_prev_leftover_robot`` as a robot-space absolute-action prefix;
* receive only ``actions`` robot-space chunks.

The model/prefix/action-chunk logic remains in the existing Paper RTC client.
This entrypoint replaces only its hardware transport and refuses any endpoint
that does not identify itself as the ROS-only adapter.
"""

from __future__ import annotations

import atexit
import os
import socket

import cv2
import numpy as np

import paper_rtc_online_inference_execution as _paper_rtc


_ROS_ONLY_DEFAULT_ENDPOINT = "tcp://127.0.0.1:9901"
_ROS_ONLY_TRANSPORT = "ros_topics_only"
_active_ros_only_client = None


class RosOnlyRobotControlServiceClient(_paper_rtc.RobotArmServiceClient):
    """Paper RTC hardware client restricted to the ROS-only adapter."""

    def connect(self) -> None:
        global _active_ros_only_client

        if self._sock is not None:
            raise RuntimeError("ROS-only service client is already connected")
        if not self.endpoint.startswith("tcp://"):
            raise ValueError(
                f"ROS-only service endpoint must use tcp://, got {self.endpoint!r}"
            )
        address = self.endpoint[len("tcp://") :]
        try:
            host, port_text = address.rsplit(":", 1)
            port = int(port_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid ROS-only service endpoint: {self.endpoint!r}") from exc
        if not host or not (1 <= port <= 65535):
            raise ValueError(f"invalid ROS-only service endpoint: {self.endpoint!r}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.recv_timeout_ms / 1000)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((host, port))
            self._sock = sock
            _active_ros_only_client = self
            reply = self._request({"cmd": "ping"})
            if not reply.get("pong"):
                raise RuntimeError(f"unexpected ROS-only service ping reply: {reply}")
            transport = reply.get("transport")
            if transport != _ROS_ONLY_TRANSPORT:
                raise RuntimeError(
                    "refusing non-ROS-only hardware service: "
                    f"expected transport={_ROS_ONLY_TRANSPORT!r}, got {transport!r}"
                )
            supported = set(reply.get("supported_action_spaces", ()))
            if self.action_space not in supported:
                raise RuntimeError(
                    f"ROS-only service does not support {self.action_space!r}: {sorted(supported)!r}"
                )
            controller_key = (
                "end_effector_controller_subscriptions"
                if self.action_space == "absolute_end_effector"
                else "piper_controller_subscriptions"
            )
            controllers = reply.get(controller_key)
            if controllers != {"left": 1, "right": 1}:
                raise RuntimeError(
                    "ROS-only service does not have exactly one Piper controller per side: "
                    f"{controllers!r}"
                )
        except BaseException:
            if _active_ros_only_client is self:
                _active_ros_only_client = None
            self._sock = None
            sock.close()
            raise

        print(f"  [ROS-only service] connected to {self.endpoint}")

    def close(self) -> None:
        global _active_ros_only_client

        try:
            super().close()
        finally:
            if _active_ros_only_client is self:
                _active_ros_only_client = None


_paper_rtc.RobotArmServiceClient = RosOnlyRobotControlServiceClient


_original_create_arg_parser = _paper_rtc.create_arg_parser


def create_ros_only_arg_parser():
    parser = _original_create_arg_parser()
    parser.set_defaults(robot_endpoint=_ROS_ONLY_DEFAULT_ENDPOINT)
    for action in parser._actions:
        if action.dest == "robot_endpoint":
            action.help = (
                "ROS-only control service endpoint; the peer must report "
                "transport=ros_topics_only"
            )
            break
    return parser


_paper_rtc.create_arg_parser = create_ros_only_arg_parser


def _cleanup_active_ros_only_client() -> None:
    client = _active_ros_only_client
    if client is None:
        return
    try:
        client.stop()
    finally:
        client.close()


atexit.register(_cleanup_active_ros_only_client)


_GRIPPER_OPEN_WIDTH_M = 0.105
_START_GRIPPER_WIDTH_M = 0.060
_CAMERAS = ("cam_top", "cam_left_wrist", "cam_right_wrist")
_IMAGE_SIZE = 224
_JPEG_QUALITY = 85
_TRANSPORT_KEY = "_ra_image_transport"
_TRANSPORT_VERSION = 1
_original_make_agilex_observation = _paper_rtc.make_agilex_observation


def _encode_client_image(image: np.ndarray) -> bytes:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise TypeError(f"camera image must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"camera image must have shape (3,H,W), got {image.shape}")

    image_hwc = np.transpose(image, (1, 2, 0))
    source_height, source_width = image_hwc.shape[:2]
    ratio = max(source_width / _IMAGE_SIZE, source_height / _IMAGE_SIZE)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    resized = cv2.resize(
        image_hwc,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
    pad_height = (_IMAGE_SIZE - resized_height) // 2
    pad_width = (_IMAGE_SIZE - resized_width) // 2
    padded[
        pad_height : pad_height + resized_height,
        pad_width : pad_width + resized_width,
    ] = resized

    encoded, payload = cv2.imencode(
        ".jpg",
        cv2.cvtColor(padded, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
    )
    if not encoded:
        raise RuntimeError("OpenCV failed to encode camera image as JPEG")
    return payload.tobytes()


def make_training_paper_rtc_observation(snapshot, prompt: str) -> dict:
    """Build a network-ready observation with 10470 gripper units.

    RobotArmService reports gripper width in meters, while the 10470 LeRobot
    dataset stores ``observation.gripper_position`` as a 0..1 open percentage.
    Images are resized to 224x224 and JPEG-encoded before network transfer.
    """

    obs = _original_make_agilex_observation(snapshot, prompt)
    if set(obs["images"]) != set(_CAMERAS):
        raise ValueError(f"observation must contain exactly {_CAMERAS}")
    obs["images"] = {
        camera: _encode_client_image(obs["images"][camera])
        for camera in _CAMERAS
    }
    obs[_TRANSPORT_KEY] = _TRANSPORT_VERSION
    gripper = np.asarray(obs["gripper_position"], dtype=np.float32)
    if gripper.size and float(np.nanmax(np.abs(gripper))) <= _GRIPPER_OPEN_WIDTH_M + 1e-3:
        gripper = gripper / _GRIPPER_OPEN_WIDTH_M
    obs["gripper_position"] = np.clip(gripper, 0.0, 1.0).astype(np.float32)
    return obs


_paper_rtc.make_agilex_observation = make_training_paper_rtc_observation


def jpeg_observation_payload_kib(obs: dict) -> float:
    images = obs.get("images")
    if not isinstance(images, dict):
        return 0.0
    return sum(len(images[camera]) for camera in _CAMERAS) / 1024.0


_paper_rtc.obs_payload_kib = jpeg_observation_payload_kib


_OriginalWebsocketClientPolicy = _paper_rtc.websocket_client_policy.WebsocketClientPolicy


class JpegWebsocketClientPolicy(_OriginalWebsocketClientPolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        transport = self.get_server_metadata().get("ra_image_transport")
        valid = (
            isinstance(transport, dict)
            and transport.get("version") == _TRANSPORT_VERSION
            and transport.get("encoding") == "jpeg"
            and transport.get("image_size") == _IMAGE_SIZE
            and tuple(transport.get("cameras", ())) == _CAMERAS
        )
        if not valid:
            self._ws.close()
            raise RuntimeError(f"RA JPEG server metadata mismatch: {transport!r}")


_paper_rtc.websocket_client_policy.WebsocketClientPolicy = JpegWebsocketClientPolicy

# The shared paper client starts with the left gripper closed. The 10470 dataset
# was collected with gripper_position in 0..1 percent and both grippers usually
# open near reset, so keep the training-paper RTC startup state in-distribution.
_paper_rtc._DEFAULT_INIT_ACTION[6] = _START_GRIPPER_WIDTH_M
_paper_rtc._DEFAULT_INIT_ACTION[13] = _START_GRIPPER_WIDTH_M


def training_paper_rtc_smooth_action(last, cur, alpha: float = 1.0, max_delta=None):
    """Disable EMA smoothing while keeping the hard per-step safety cap."""

    del alpha
    cur = np.asarray(cur, dtype=np.float32)
    if last is None:
        return cur
    last = np.asarray(last, dtype=np.float32)
    if max_delta is None:
        return cur
    delta = cur - last
    return last + np.clip(delta, -max_delta, max_delta)


_paper_rtc.smooth_action = training_paper_rtc_smooth_action


_original_pop_ready = _paper_rtc.AsyncClientChunkPlanner.pop_ready
_chunk_stats_count = 0


def _format_range(values: np.ndarray) -> str:
    return np.array2string(values, precision=4, suppress_small=True, separator=", ")


def training_paper_rtc_pop_ready_with_chunk_stats(self):
    """Add compact whole-chunk diagnostics without editing the shared client."""

    global _chunk_stats_count
    packet = _original_pop_ready(self)
    if packet is None:
        return None

    robot_chunk = np.asarray(packet.get("robot_actions"), dtype=np.float32)
    if robot_chunk.ndim == 2 and robot_chunk.shape[1] >= 14:
        targets = robot_chunk[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
        grippers_m = robot_chunk[:, [6, 13]] * _GRIPPER_OPEN_WIDTH_M
        target_range = np.ptp(targets, axis=0)
        target_step_max = np.max(np.abs(np.diff(targets, axis=0)), axis=0) if len(targets) > 1 else np.zeros(12)
        gripper_range = np.ptp(grippers_m, axis=0)
        first = robot_chunk[0]
        last = robot_chunk[-1]

        timing = dict(packet.get("client_timing", {}))
        timing.update(
            {
                "tp_rtc_chunk_target_range_max": float(np.max(target_range)),
                "tp_rtc_chunk_target_step_max": float(np.max(target_step_max)),
                "tp_rtc_chunk_left_gripper_range_m": float(gripper_range[0]),
                "tp_rtc_chunk_right_gripper_range_m": float(gripper_range[1]),
                "tp_rtc_chunk_first_right_gripper_m": float(first[13] * _GRIPPER_OPEN_WIDTH_M),
                "tp_rtc_chunk_last_right_gripper_m": float(last[13] * _GRIPPER_OPEN_WIDTH_M),
            }
        )
        packet["client_timing"] = timing

        first_n = int(os.environ.get("TRAINING_PAPER_RTC_CHUNK_STATS_FIRST_N", "5"))
        interval = int(os.environ.get("TRAINING_PAPER_RTC_CHUNK_STATS_INTERVAL", "20"))
        should_print = _chunk_stats_count < first_n or (interval > 0 and _chunk_stats_count % interval == 0)
        if should_print:
            print(
                "[TP-RTC CHUNK] "
                f"space={_paper_rtc._EXECUTION_ACTION_SPACE} "
                f"seq={_chunk_stats_count} len={len(robot_chunk)} "
                f"target_range_max={np.max(target_range):.4f} "
                f"target_step_max={np.max(target_step_max):.4f} "
                f"gripper_range_m={_format_range(gripper_range)} "
                f"first_gripper_m={_format_range(first[[6, 13]] * _GRIPPER_OPEN_WIDTH_M)} "
                f"last_gripper_m={_format_range(last[[6, 13]] * _GRIPPER_OPEN_WIDTH_M)}"
            )
        _chunk_stats_count += 1

    return packet


_paper_rtc.AsyncClientChunkPlanner.pop_ready = training_paper_rtc_pop_ready_with_chunk_stats
main = _paper_rtc.main


if __name__ == "__main__":
    main()
