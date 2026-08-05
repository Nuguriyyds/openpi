#!/usr/bin/env python3
"""Record the live ROS2 top-camera stream with frame timestamps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

import cv2
import cv_bridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopCameraRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("breakfast_top_camera_recorder")
        self.args = args
        self.output_dir = args.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_dir / args.video_name
        self.frames_path = self.output_dir / args.frames_name
        self.metadata_path = self.output_dir / args.metadata_name
        self.ready_path = self.output_dir / args.ready_name

        for path in (self.video_path, self.frames_path, self.metadata_path, self.ready_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing recording output: {path}")

        self.bridge = cv_bridge.CvBridge()
        self.frames_stream = self.frames_path.open("x", encoding="utf-8")
        self.writer: cv2.VideoWriter | None = None
        self.codec: str | None = None
        self.frame_size: tuple[int, int] | None = None
        self.frame_count = 0
        self.start_timestamp_utc = utc_now()
        self.first_receipt_timestamp_unix: float | None = None
        self.last_receipt_timestamp_unix: float | None = None
        self.first_ros_timestamp_ns: int | None = None
        self.last_ros_timestamp_ns: int | None = None
        self.closed = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
        )
        self.subscription = self.create_subscription(Image, args.topic, self._on_image, qos)
        self.get_logger().info(f"Recording {args.topic} into {self.video_path}")

    def _open_writer(self, width: int, height: int) -> None:
        candidates = [self.args.codec] if self.args.codec != "auto" else ["avc1", "H264", "mp4v"]
        for codec in candidates:
            writer = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter_fourcc(*codec),
                self.args.fps,
                (width, height),
            )
            if writer.isOpened():
                self.writer = writer
                self.codec = codec
                self.frame_size = (width, height)
                return
            writer.release()
            self.video_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not open an MP4 writer with codecs: {candidates}")

    def _on_image(self, msg: Image) -> None:
        receipt_timestamp_unix = time.time()
        ros_timestamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width = frame.shape[:2]

        if self.writer is None:
            self._open_writer(width, height)
        if self.frame_size != (width, height):
            raise ValueError(f"Top-camera resolution changed from {self.frame_size} to {(width, height)}")

        assert self.writer is not None
        self.writer.write(frame)

        if self.first_receipt_timestamp_unix is None:
            self.first_receipt_timestamp_unix = receipt_timestamp_unix
            self.first_ros_timestamp_ns = ros_timestamp_ns
        self.last_receipt_timestamp_unix = receipt_timestamp_unix
        self.last_ros_timestamp_ns = ros_timestamp_ns

        record = {
            "frame_index": self.frame_count,
            "video_time_s": self.frame_count / self.args.fps,
            "timestamp_utc": datetime.fromtimestamp(receipt_timestamp_unix, timezone.utc).isoformat(),
            "timestamp_unix": receipt_timestamp_unix,
            "ros_timestamp_ns": ros_timestamp_ns,
        }
        self.frames_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.frame_count += 1
        if self.frame_count == 1:
            self.frames_stream.flush()
            self.ready_path.write_text(json.dumps(record), encoding="utf-8")
            self.get_logger().info(f"First frame received: {width}x{height} at {self.args.fps:g} fps")
        elif self.frame_count % max(1, round(self.args.fps)) == 0:
            self.frames_stream.flush()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.writer is not None:
            self.writer.release()
        self.frames_stream.flush()
        self.frames_stream.close()
        metadata = {
            "topic": self.args.topic,
            "video_file": self.video_path.name,
            "frames_file": self.frames_path.name,
            "fps": self.args.fps,
            "codec": self.codec,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "frame_count": self.frame_count,
            "start_timestamp_utc": self.start_timestamp_utc,
            "end_timestamp_utc": utc_now(),
            "first_receipt_timestamp_unix": self.first_receipt_timestamp_unix,
            "last_receipt_timestamp_unix": self.last_receipt_timestamp_unix,
            "first_ros_timestamp_ns": self.first_ros_timestamp_ns,
            "last_ros_timestamp_ns": self.last_ros_timestamp_ns,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.ready_path.unlink(missing_ok=True)
        self.get_logger().info(f"Recorded {self.frame_count} top-camera frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topic", default="/camera/top/camera/color/image_raw")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", default="auto", help="FourCC codec or 'auto'")
    parser.add_argument("--video-name", default="top.mp4")
    parser.add_argument("--frames-name", default="top_frames.jsonl")
    parser.add_argument("--metadata-name", default="top_recording.json")
    parser.add_argument("--ready-name", default="top_recorder.ready")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    recorder = TopCameraRecorder(args)

    def stop_recording(_signum, _frame):
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, stop_recording)
    signal.signal(signal.SIGINT, stop_recording)
    try:
        rclpy.spin(recorder)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
