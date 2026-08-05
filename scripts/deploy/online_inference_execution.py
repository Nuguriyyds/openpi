#!/usr/bin/env python3
"""
在线机械臂推理执行脚本
=====================

集成以下功能:
  1. 读取机械臂实时状态 (关节角、夹爪位置)
  2. 采集相机图像
  3. 调用 OpenPI 模型进行推理
  4. 执行推理的动作指令

平台: ROS2 (AGILEXDroid) — 通过订阅话题获取观察和发送动作

流程:
  [读取状态] → [采集图像] → [推理] → [执行动作] → 循环...

使用方法:
  python3 online_inference_execution.py \
    --model_config pi05_agilex \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --prompt "pick up the cube"
"""

import os
import sys
import json
import time
import threading
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import shutil
from openpi_client import websocket_client_policy

try:
    import yaml
except ImportError:
    yaml = None

# ========== ROS2 可选导入 ==========
HAS_ROS2 = False
try:
    import rclpy
    import cv_bridge
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, JointState
    HAS_ROS2 = True
except ImportError:
    print("[警告] ROS2 模块未安装，无法使用 --platform ros2")

# ========== Piper SDK 可选导入 ==========
HAS_PIPER = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'piper_sdk'))
    from piper_sdk import C_PiperInterface_V2
    HAS_PIPER = True
except ImportError:
    print("[警告] Piper SDK 未安装，无法使用 --platform piper")


# =====================================================
#              常量定义
# =====================================================

# 弧度 → SDK 0.001° 单位
RAD_TO_DEG_001 = 1000.0 * 180.0 / math.pi

# 夹爪: 米 → SDK 0.001mm 单位
METER_TO_GRIPPER_RAW = 1_000_000.0

# 夹爪行程范围 (SDK 0.001mm 单位)
GRIPPER_MIN, GRIPPER_MAX = 0, 70000

# 回零点等待判断阈值
ZERO_THRESHOLD = 300
DEFAULT_BREAKFAST_SUBTASKS = (
    {
        "name": "put_bread_in_toaster",
        "prompt": "Pick up all bread pieces from the bread rack and insert them into the toaster.",
    },
    {
        "name": "activate_toaster",
        "prompt": (
            "Push down the toaster's front lever to activate the toaster, then return the right "
            "gripper to its initial resting configuration."
        ),
    },
    {
        "name": "pour_drink",
        "prompt": (
            "Pour drink from the water bottle into the cup, return the bottle upright, and return "
            "the left gripper to its initial resting configuration."
        ),
    },
    {
        "name": "plate_toast",
        "prompt": "Remove all toasted bread from the toaster and place it on the plate.",
    },
)


class JSONLRunLogger:
    """Write deployment results as one JSON object per line."""

    def __init__(self, root: str):
        shared_run_dir = os.environ.get("BREAKFAST_RUN_DIR")
        if shared_run_dir:
            self.run_dir = Path(shared_run_dir).expanduser().resolve()
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            self.run_dir = Path(root).expanduser() / timestamp
            self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / "events.jsonl"
        if self.path.exists():
            raise FileExistsError(f"JSONL event log already exists: {self.path}")
        self._lock = threading.Lock()

    @staticmethod
    def _default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    def write(self, event: str, **fields):
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=self._default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


class ProgressSubtaskManager:
    """Sequential subtask manager with the FurnitureVLA anti-spike filter."""

    def __init__(self, subtasks, threshold: float):
        if not subtasks:
            raise ValueError("subtasks cannot be empty")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"progress threshold must be in [0, 1], got {threshold}")

        self.subtasks = []
        for index, subtask in enumerate(subtasks):
            name = str(subtask.get("name", "")).strip()
            prompt = str(subtask.get("prompt", "")).strip()
            if not name or not prompt:
                raise ValueError(f"subtask {index} requires name and prompt")
            self.subtasks.append({"name": name, "prompt": prompt})

        self.threshold = float(threshold)
        self.index = 0
        self.finished = False
        self._history = deque(maxlen=4)

    @property
    def current(self):
        return self.subtasks[self.index]

    def update(self, raw_progress: float):
        if self.finished:
            raise RuntimeError("all subtasks are already complete")
        if not np.isfinite(raw_progress):
            raise ValueError(f"progress must be finite, got {raw_progress}")

        progress = float(np.clip(raw_progress, 0.0, 1.0))
        self._history.append(progress >= self.threshold)
        history = tuple(self._history)

        reason = None
        if len(history) >= 2 and history[-2:] == (True, True):
            reason = "two_consecutive_high"
        elif len(history) == 4 and history == (True, False, False, True):
            reason = "high_low_low_high"

        completed_index = None
        completed_name = None
        if reason is not None:
            completed_index = self.index
            completed_name = self.current["name"]
            if self.index == len(self.subtasks) - 1:
                self.finished = True
            else:
                self.index += 1
            self._history.clear()

        return {
            "progress_raw": float(raw_progress),
            "progress": progress,
            "history": [int(value) for value in history],
            "triggered": reason is not None,
            "trigger_reason": reason,
            "completed_index": completed_index,
            "completed_name": completed_name,
            "all_finished": self.finished,
        }


# =====================================================
#              状态观察对象
# =====================================================

class ObservationBuffer:
    """线程安全的观察缓冲"""

    def __init__(self, history_len: int = 1):
        self.history_len = history_len
        self._lock = threading.Lock()
        self.images = {
            'cam_top': None,
            'cam_left_wrist': None,
            'cam_right_wrist': None,
        }
        self.image_timestamps = {name: 0.0 for name in self.images}
        self.image_ros_timestamps_ns = {name: 0 for name in self.images}
        self.joint_left = np.zeros(6)
        self.joint_right = np.zeros(6)
        self.gripper_position = np.zeros(2)
        self.timestamp = 0.0
        self._last_update = 0.0

    def update_images(
        self,
        cam_name: str,
        image: np.ndarray,
        *,
        timestamp: float | None = None,
        ros_timestamp_ns: int = 0,
    ):
        """更新相机图像 (HxWxC 格式)"""
        with self._lock:
            if cam_name in self.images:
                # 转为 3xHxW 格式供模型使用
                if len(image.shape) == 3 and image.shape[2] == 3:
                    self.images[cam_name] = np.transpose(image, (2, 0, 1))
                else:
                    self.images[cam_name] = image
                self.image_timestamps[cam_name] = float(timestamp if timestamp is not None else time.time())
                self.image_ros_timestamps_ns[cam_name] = int(ros_timestamp_ns)

    def update_joints(self, side: str, joints: np.ndarray):
        """更新关节角度 (弧度)"""
        with self._lock:
            if side == 'left':
                self.joint_left = np.array(joints, dtype=np.float32)
            elif side == 'right':
                self.joint_right = np.array(joints, dtype=np.float32)

    def update_gripper(self, gripper_pos: np.ndarray):
        """更新夹爪位置 (米)"""
        with self._lock:
            self.gripper_position = np.array(gripper_pos, dtype=np.float32)

    def update_gripper_one_side(self, side: str, value: float):
        """在锁内安全地更新单侧夹爪，避免读-改-写竞争"""
        with self._lock:
            if side == 'left':
                self.gripper_position[0] = value
            elif side == 'right':
                self.gripper_position[1] = value

    def update_timestamp(self, ts: float):
        """更新时间戳"""
        with self._lock:
            self.timestamp = ts
            self._last_update = time.time()

    def is_fresh(self, threshold_s: float = 1.0) -> bool:
        """检查观察是否为新鲜数据"""
        with self._lock:
            return (time.time() - self._last_update) < threshold_s

    def get_snapshot(self) -> Dict[str, Any]:
        """获取当前观察的快照"""
        with self._lock:
            return {
                'images': {k: v.copy() if v is not None else v
                          for k, v in self.images.items()},
                'image_timestamps': self.image_timestamps.copy(),
                'image_ros_timestamps_ns': self.image_ros_timestamps_ns.copy(),
                'joint_left': self.joint_left.copy(),
                'joint_right': self.joint_right.copy(),
                'gripper_position': self.gripper_position.copy(),
                'timestamp': self.timestamp,
            }

# =====================================================
#              Piper 机械臂封装
# =====================================================

class PiperArm:
    """单个 Piper 机械臂控制器"""

    def __init__(self, can_name: str, name: str = ''):
        if not HAS_PIPER:
            raise RuntimeError("Piper SDK 未安装")
        self.can_name = can_name
        self.name = name or can_name
        self.piper = C_PiperInterface_V2(can_name=can_name)
        self._enabled = False

    def connect(self):
        print(f"  [{self.name}] 连接 CAN 端口 {self.can_name} ...")
        self.piper.ConnectPort()
        time.sleep(0.2)

    def enable(self, timeout: float = 5.0):
        print(f"  [{self.name}] 使能机械臂 ...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.piper.EnablePiper():
                print(f"  [{self.name}] 使能成功")
                self._enabled = True
                return
            time.sleep(0.01)
        raise RuntimeError(f"[{self.name}] 使能超时 ({timeout}s)")

    def read_state(self) -> Tuple[np.ndarray, float]:
        """
        读取机械臂当前状态
        返回: (关节角度[弧度], 夹爪宽度[米])
        """
        if not self._enabled:
            return np.zeros(6), 0.0

        try:
            msgs = self.piper.GetArmJointMsgs()
            js = msgs.joint_state

            # SDK 单位: 0.001° → 弧度
            joints_rad = np.array([
                js.joint_1 / RAD_TO_DEG_001,
                js.joint_2 / RAD_TO_DEG_001,
                js.joint_3 / RAD_TO_DEG_001,
                js.joint_4 / RAD_TO_DEG_001,
                js.joint_5 / RAD_TO_DEG_001,
                js.joint_6 / RAD_TO_DEG_001,
            ], dtype=np.float32)

            # SDK 单位: 0.001mm → 米
            gripper_meter = msgs.gripper_state.cur_pos / METER_TO_GRIPPER_RAW

            return joints_rad, gripper_meter
        except Exception as e:
            print(f"[{self.name}] 读取状态异常: {e}")
            return np.zeros(6), 0.0

    def send_command(self, joints_rad: np.ndarray, gripper_meter: float, speed_percent: int = 100):
        """
        发送关节和夹爪指令
        
        Args:
            joints_rad: 6个关节角度 (弧度)
            gripper_meter: 夹爪宽度 (米)
            speed_percent: 速度百分比 (1-100)
        """
        if not self._enabled:
            print(f"[{self.name}] 未使能，无法发送指令")
            return

        try:
            # 设置运动模式
            self.piper.MotionCtrl_2(
                ctrl_mode=0x01,           # CAN 命令控制
                move_mode=0x01,           # MOVE J
                move_spd_rate_ctrl=speed_percent,
                is_mit_mode=0xAD          # 高跟随模式
            )

            # 转换单位并发送关节指令
            raw_joints = [round(j * RAD_TO_DEG_001) for j in joints_rad]
            self.piper.JointCtrl(*raw_joints)

            # 转换单位并发送夹爪指令
            raw_gripper = round(gripper_meter * METER_TO_GRIPPER_RAW)
            raw_gripper = max(GRIPPER_MIN, min(GRIPPER_MAX, raw_gripper))
            self.piper.GripperCtrl(raw_gripper, 1000, 0x01, 0x00)
        except Exception as e:
            print(f"[{self.name}] 发送指令异常: {e}")

    def go_zero(self, speed_percent: int = 30):
        """回零点"""
        if not self._enabled:
            return

        self.piper.MotionCtrl_2(
            ctrl_mode=0x01, move_mode=0x01,
            move_spd_rate_ctrl=speed_percent,
            is_mit_mode=0x00
        )
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        self.piper.GripperCtrl(0, 1000, 0x01, 0x00)

    def disconnect(self):
        print(f"  [{self.name}] 断开连接")


# =====================================================
#              ROS2 观察采集节点
# =====================================================

class ROS2ObservationCollector(Node):
    """ROS2 观察采集节点"""

    def __init__(self, obs_buffer: ObservationBuffer):
        if not HAS_ROS2:
            raise RuntimeError("ROS2 未安装")

        super().__init__('online_inference_collector')
        self.obs_buffer = obs_buffer
        self.bridge = cv_bridge.CvBridge()
        self._ready_lock = threading.Lock()
        self._topic_ready = {
            'cam_top': False,
            'cam_left_wrist': False,
            'cam_right_wrist': False,
            'joint_left': False,
            'joint_right': False,
        }

        # QoS 配置
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        arm_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # 订阅相机
        self.create_subscription(
            Image, '/camera/top/camera/color/image_raw',
            lambda msg: self._on_image(msg, 'cam_top'),
            cam_qos
        )
        self.create_subscription(
            Image, '/camera/left/camera/color/image_raw',
            lambda msg: self._on_image(msg, 'cam_left_wrist'),
            cam_qos
        )
        self.create_subscription(
            Image, '/camera/right/camera/color/image_raw',
            lambda msg: self._on_image(msg, 'cam_right_wrist'),
            cam_qos
        )

        # 订阅关节状态
        self.create_subscription(
            JointState, '/joint_states_left',
            lambda msg: self._on_joint(msg, 'left'),
            arm_qos
        )
        self.create_subscription(
            JointState, '/joint_states_right',
            lambda msg: self._on_joint(msg, 'right'),
            arm_qos
        )

        self.get_logger().info("[ROS2] 观察采集节点已启动")

    def _on_image(self, msg: Image, cam_name: str):
        """相机回调"""
        try:
            # ROS Image → RGB numpy (模型使用 RGB 格式训练)
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            receipt_timestamp = time.time()
            ros_timestamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            self.obs_buffer.update_images(
                cam_name,
                image,
                timestamp=receipt_timestamp,
                ros_timestamp_ns=ros_timestamp_ns,
            )
            self.obs_buffer.update_timestamp(receipt_timestamp)
            with self._ready_lock:
                self._topic_ready[cam_name] = True
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    def _on_joint(self, msg: JointState, side: str):
        """关节状态回调"""
        try:
            joints = np.array(msg.position[:6], dtype=np.float32)
            gripper = np.array(msg.position[6:7], dtype=np.float32)
            self.obs_buffer.update_joints(side, joints)
            if side == 'left':
                self.obs_buffer.update_gripper_one_side('left', gripper[0])
                with self._ready_lock:
                    self._topic_ready['joint_left'] = True
            else:
                self.obs_buffer.update_gripper_one_side('right', gripper[0])
                with self._ready_lock:
                    self._topic_ready['joint_right'] = True
        except Exception as e:
            self.get_logger().error(f"关节解析失败: {e}")

    def get_missing_topics(self) -> list[str]:
        """返回尚未收到数据的话题键名。"""
        with self._ready_lock:
            return [name for name, ready in self._topic_ready.items() if not ready]

    def all_topics_ready(self) -> bool:
        """检查所有关键话题是否都已收到首帧数据。"""
        with self._ready_lock:
            return all(self._topic_ready.values())


# =====================================================
#              在线推理执行器
# =====================================================


def make_agilex_observation(obs_snapshot, prompt) -> dict:
    state_arm = np.concatenate([obs_snapshot['joint_left'], obs_snapshot['joint_right']], axis=0)
    return {
        "state": state_arm,
        "gripper_position": obs_snapshot['gripper_position'],
        "images": {
            "cam_top": obs_snapshot['images']['cam_top'],
            "cam_left_wrist": obs_snapshot['images']['cam_left_wrist'],
            "cam_right_wrist": obs_snapshot['images']['cam_right_wrist'],
        },
        "prompt": prompt,
    }

# =====================================================
#              主程序
# =====================================================

def smooth_action(last_action, action: np.ndarray) -> np.ndarray:
        """
        平滑动作 (简单的指数加权移动平均)
        """
        if last_action is None:
            smoothed = action
        else:
            alpha = 0.6
            smoothed = alpha * action + (1 - alpha) * last_action
            # smoothed = last_action + alpha * (action - last_action)
        return smoothed

def run_ros2_inference(args):
    """ROS2 平台推理执行"""
    if not HAS_ROS2:
        print("[错误] ROS2 未安装")
        return
    if not HAS_PIPER:
        print("[错误] ROS2 模式下执行动作需要 Piper SDK")
        return

    print("\n" + "="*60)
    print("  在线推理执行 - ROS2 AGILEXDroid 平台")
    print("="*60)

    save_image_path = '/home/geekplus/develop/openpi/scripts/deploy/save_images'
    if os.path.exists(save_image_path):
        shutil.rmtree(save_image_path)
    os.makedirs(save_image_path)


    # 初始化 ROS2
    rclpy.init()

    # 创建client
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = client.get_server_metadata()
    print("server metadata:", server_metadata)

    subtask_manager = ProgressSubtaskManager(
        getattr(args, 'subtasks', DEFAULT_BREAKFAST_SUBTASKS),
        float(getattr(args, 'progress_threshold', 0.95)),
    )
    run_logger = JSONLRunLogger(getattr(args, 'log_dir', 'logs/breakfast_progress_online'))
    print(f"JSONL log: {run_logger.path}")
    run_logger.write(
        "session_start",
        run_dir=str(run_logger.run_dir),
        server_host=args.host,
        server_port=args.port,
        server_metadata=server_metadata,
        progress_threshold=subtask_manager.threshold,
        subtasks=subtask_manager.subtasks,
    )

    # 观察缓冲
    obs_buffer = ObservationBuffer()

    # ROS2 节点
    ros2_node = ROS2ObservationCollector(obs_buffer)

    # 推理前检查所有话题是否收到数据
    print("\n[检查] 等待所有关键话题首帧数据...")
    topic_wait_timeout = max(float(getattr(args, 'topic_wait_timeout', 10.0)), 0.0)
    wait_start = time.time()
    while rclpy.ok() and not ros2_node.all_topics_ready():
        rclpy.spin_once(ros2_node, timeout_sec=0.1)
        if topic_wait_timeout > 0 and (time.time() - wait_start) > topic_wait_timeout:
            missing_topics = ros2_node.get_missing_topics()
            print(f"[错误] 以下话题在 {topic_wait_timeout:.1f}s 内未收到数据: {missing_topics}")
            print("[提示] 请检查 ROS2 topic 名称、发布节点状态和 QoS 配置")
            return

    print("[检查通过] 所有关键话题已收到数据，开始推理流程")

    # 机械臂执行器 (直接使用 Piper SDK 发送动作)
    print("\n[机械臂] 初始化执行通道")
    arm_left = PiperArm(args.can_left, name='左臂')
    arm_right = PiperArm(args.can_right, name='右臂')
    arm_left.connect()
    arm_right.connect()
    arm_left.enable()
    arm_right.enable()

    # 运动到初始姿态
    print("\n[机械臂] 运动到初始姿态...")
    # _init_joints_left  = np.array([-0.070493,    0.,         -0.5154659,  0.05713056,  1.0897719,  -0.078029  ], dtype=np.float32)
    # _init_joints_right = np.array([ 0.028731,   -0.00573922, -0.50383043, 0.00861756,  1.0970463,   0.01107722], dtype=np.float32)
    _init_joints_left  = np.array([-0.47566299200000006, 0.12493392800000001, -0.498514632, 0.09187754800000002, 0.847115528, -0.11246146800000001], dtype=np.float32)
    _init_joints_right  = np.array([0.057844304, 0.45746890000000007, -0.47620375600000003, 0.254577736, 0.8388994040000001, -0.19685554000000002], dtype=np.float32)
    _init_gripper_left  = 0.060    # 米
    _init_gripper_right = 0.060  # 米
    arm_left.send_command(_init_joints_left,  _init_gripper_left,  speed_percent=10)
    arm_right.send_command(_init_joints_right, _init_gripper_right, speed_percent=10)
    time.sleep(3.0)
    print("[机械臂] 初始姿态到位")

    action_horizon = int(getattr(args, 'action_horizon', 10))
    max_inferences = int(getattr(args, 'max_inferences', 0))
    speed_percent = int(getattr(args, 'speed_percent', 100))
    # dt 控制每帧执行间隔，使用 args.fps（对应数据集采集频率）
    # args.fps 应与训练数据集的 fps (默认 30Hz) 保持一致，而非推理触发频率
    dt = 1.0 / args.fps

    print("\n[系统] 准备就绪，开始推理执行")
    print("  按 Ctrl+C 停止")
    print("-" * 60)

    action_chunk = None
    action_idx = 0
    inference_count = 0
    _stop_spin = threading.Event()

    def ros_spin_thread():
        """后台持续 spin，确保推理期间相机/关节回调不被阻塞"""
        while not _stop_spin.is_set() and rclpy.ok():
            rclpy.spin_once(ros2_node, timeout_sec=0.005)

    spin_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    spin_thread.start()

    def inference_loop():
        nonlocal action_chunk, action_idx, inference_count
        last_action_left = None
        last_action_right = None
        last_gripper_left = None
        last_gripper_right = None
        while rclpy.ok():
            # 检查观察是否新鲜
            if not obs_buffer.is_fresh(threshold_s=1.0):
                print("[警告] 观察数据超时，等待...")
                time.sleep(0.1)
                continue

            # 推理上限检查
            if max_inferences > 0 and inference_count >= max_inferences:
                print(f"[系统] 已完成 {inference_count} 次推理，达到上限，退出")
                return

            print(f"[推理] 第 {inference_count} 次" + (f" / {max_inferences}" if max_inferences > 0 else ""))

            obs_snapshot = obs_buffer.get_snapshot()
            
            # cv2.imwrite(save_image_path + f"/debug_obs_{inference_count}_top.jpg", cv2.cvtColor(np.transpose(obs_snapshot['images']['cam_top'], (1, 2, 0)), cv2.COLOR_RGB2BGR))
            # cv2.imwrite(save_image_path + f"/debug_obs_{inference_count}_left_wrist.jpg", cv2.cvtColor(np.transpose(obs_snapshot['images']['cam_left_wrist'], (1, 2, 0)), cv2.COLOR_RGB2BGR))
            # cv2.imwrite(save_image_path + f"/debug_obs_{inference_count}_right_wrist.jpg", cv2.cvtColor(np.transpose(obs_snapshot['images']['cam_right_wrist'], (1, 2, 0)), cv2.COLOR_RGB2BGR))
            # print(f"Joint Left: {obs_snapshot['joint_left']}")
            # print(f"Joint Right: {obs_snapshot['joint_right']}")
            # print(f"Gripper Position: {obs_snapshot['gripper_position']}")

            current_subtask_index = subtask_manager.index
            current_subtask = subtask_manager.current
            obs = make_agilex_observation(obs_snapshot, current_subtask["prompt"])

            start = time.time()
            print("start infer...")
            out = client.infer(obs)
            print("end infer.")
            dt_ms = (time.time() - start) * 1000
            server_timing = out.get("server_timing", {})
            print(f"infer time: {dt_ms:.1f} ms")
            print('server_time: ', server_timing)

            action_chunk = out.get("actions") if out else None
            progress_chunk = out.get("progress") if out else None
            inference_count += 1

            if action_chunk is None:
                print(
                    "\u005b\u8b66\u544a\u005d "
                    "\u63a8\u7406\u8fd4\u56de actions \u4e3a\u7a7a\uff0c"
                    "\u8df3\u8fc7\u672c\u6b21\u6267\u884c"
                )
                continue
            if progress_chunk is None:
                raise KeyError("Policy response is missing progress; check the progress config/checkpoint")

            action_chunk = np.asarray(action_chunk)
            progress_chunk = np.asarray(progress_chunk).reshape(-1)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 14:
                raise ValueError(f"Expected actions shape (H, 14), got {action_chunk.shape}")
            if progress_chunk.shape != (action_chunk.shape[0],):
                raise ValueError(
                    f"Expected progress shape ({action_chunk.shape[0]},), got {progress_chunk.shape}"
                )
            if not np.isfinite(action_chunk).all() or not np.isfinite(progress_chunk).all():
                raise ValueError("actions/progress contain NaN or Inf")

            # FurnitureVLA uses the newest prediction only; progress is not ensembled.
            progress_decision = subtask_manager.update(float(progress_chunk[0]))
            print(
                f"[progress] subtask {current_subtask_index + 1}/{len(subtask_manager.subtasks)} "
                f"{current_subtask['name']}: {progress_decision['progress_raw']:.4f}, "
                f"history={progress_decision['history']}"
            )
            run_logger.write(
                "inference",
                inference_index=inference_count,
                subtask_index=current_subtask_index,
                subtask_name=current_subtask["name"],
                prompt=current_subtask["prompt"],
                progress_raw=progress_decision["progress_raw"],
                progress_clipped=progress_decision["progress"],
                progress_history=progress_decision["history"],
                observation_timestamp_unix=float(
                    obs_snapshot.get("image_timestamps", {}).get("cam_top", obs_snapshot["timestamp"])
                ),
                observation_top_ros_timestamp_ns=int(
                    obs_snapshot.get("image_ros_timestamps_ns", {}).get("cam_top", 0)
                ),
                actions_shape=list(action_chunk.shape),
                progress_shape=list(progress_chunk.shape),
                inference_ms=dt_ms,
                server_timing=server_timing,
            )

            if progress_decision["triggered"]:
                # Discard the current chunk before changing prompts.
                action_chunk = None
                run_logger.write(
                    "subtask_complete",
                    completed_index=progress_decision["completed_index"],
                    completed_name=progress_decision["completed_name"],
                    trigger_reason=progress_decision["trigger_reason"],
                    all_finished=progress_decision["all_finished"],
                )
                if progress_decision["all_finished"]:
                    print("[progress] all four breakfast subtasks completed")
                    return
                print(f"[progress] switching to: {subtask_manager.current['name']}")
                continue

            if action_chunk is None:
                print("[警告] 推理返回 actions 为空，跳过本次执行")
                continue

            # 逐帧执行动作
            for action_idx in range(len(action_chunk)):
                if not rclpy.ok():
                    return
                t_start = time.time()

                current_action = action_chunk[action_idx]
                print(f"  [动作] 原始输出: {current_action}")

                # 解析动作: [left_j0-j5, left_g, right_j0-j5, right_g]
                act_joints_left = current_action[:6]
                act_gripper_left = current_action[6] * 0.105   # 0~1 → 米
                act_joints_right = current_action[7:13]
                act_gripper_right = current_action[13] * 0.105  # 0~1 → 米

                # if act_gripper_left < 0.02:
                #     act_gripper_left = 0
                # else:
                #     act_gripper_left = 0.06
                
                # if act_gripper_right < 0.02:
                #     act_gripper_right = 0
                # else:
                #     act_gripper_right = 0.06
                
                smooth_action_left = smooth_action(last_action_left, act_joints_left)
                smooth_action_right = smooth_action(last_action_right, act_joints_right)
                smooth_gripper_left = smooth_action(last_gripper_left, act_gripper_left)
                smooth_gripper_right = smooth_action(last_gripper_right, act_gripper_right)


                last_action_left = smooth_action_left.copy()
                last_action_right = smooth_action_right.copy()
                last_gripper_left = smooth_gripper_left.copy()
                last_gripper_right = smooth_gripper_right.copy()

                arm_left.send_command(smooth_action_left, float(smooth_gripper_left), speed_percent=speed_percent)
                arm_right.send_command(smooth_action_right, float(smooth_gripper_right), speed_percent=speed_percent)

                print(
                    f"  [执行] #frame {action_idx:3d}/{len(action_chunk)-1} | "
                    f"L1={smooth_action_left[0]:.3f}, LG={smooth_gripper_left:.3f}m | "
                    f"R1={smooth_action_right[0]:.3f}, RG={smooth_gripper_right:.3f}m"
                )

                # 控制每帧执行间隔
                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                print (action_idx, ' ', action_horizon)
                if action_idx + 1 >= action_horizon:
                    break

    try:
        inference_loop()
    except KeyboardInterrupt:
        print("\n\n[系统] 用户中断")
    finally:
        _stop_spin.set()
        spin_thread.join(timeout=1.0)
        print("\n[系统] 回零点...")
        arm_left.go_zero(speed_percent=30)
        arm_right.go_zero(speed_percent=30)
        time.sleep(2.0)
        arm_left.disconnect()
        arm_right.disconnect()

        ros2_node.destroy_node()
        rclpy.shutdown()
        run_logger.write("session_end", inference_count=inference_count)
        print(f"\n[统计] 总推理次数: {inference_count}")
        print("[完成]")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件读取参数配置。"""
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，请先执行: pip install pyyaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是 key-value 字典")

    # 支持可选命名空间 online_inference_execution:
    if "online_inference_execution" in data and isinstance(data["online_inference_execution"], dict):
        data = data["online_inference_execution"]

    return data


def replay_action(args):
    """从 LeRobot v2.1 数据集重放指定 episode 的动作到 Piper 双臂。

    数据格式 (lerobot v2.1):
      - {replay_path}/meta/info.json  — 包含 fps、chunks_size 等元信息
      - {replay_path}/data/chunk-{CCC:03d}/episode_{EEE:06d}.parquet
        actions 列: float32[14] = [left_j0-j5, left_gripper(0~1), right_j0-j5, right_gripper(0~1)]
    """
    if not HAS_PIPER:
        print("[错误] Piper SDK 未安装，无法重放动作")
        return

    try:
        import pandas as pd
    except ImportError:
        print("[错误] 未安装 pandas，请执行: pip install pandas")
        return

    replay_path = Path(args.replay_path)
    episode_id = int(args.replay_id)
    speed_percent = int(getattr(args, 'speed_percent', 100))
    can_left = getattr(args, 'can_left', 'can0')
    can_right = getattr(args, 'can_right', 'can1')

    print("\n" + "=" * 60)
    print("  动作重放 - PIPER 双臂平台")
    print("=" * 60)
    print(f"  数据集路径: {replay_path}")
    print(f"  Episode ID: {episode_id}")

    # 读取 meta/info.json 获取 fps 和分块大小
    info_path = replay_path / 'meta' / 'info.json'
    if not info_path.exists():
        print(f"[错误] 找不到 meta/info.json: {info_path}")
        return
    with info_path.open('r', encoding='utf-8') as f:
        info = json.load(f)

    fps = float(info.get('fps', 30))
    chunks_size = int(info.get('chunks_size', 1000))
    dt = 1.0 / fps
    print(f"  FPS: {fps}  |  帧间隔: {dt * 1000:.1f}ms")

    # 计算 parquet 文件路径
    episode_chunk = episode_id // chunks_size
    parquet_path = replay_path / f'data/chunk-{episode_chunk:03d}/episode_{episode_id:06d}.parquet'
    if not parquet_path.exists():
        print(f"[错误] 找不到 episode 文件: {parquet_path}")
        return

    # 加载动作序列及观测字段
    df = pd.read_parquet(parquet_path)
    if 'actions' not in df.columns:
        print(f"[错误] parquet 文件中没有 'actions' 列，实际列: {df.columns.tolist()}")
        return
    actions = np.stack(df['actions'].values).astype(np.float32)           # (T, 14)

    # observation.state.joint: (T, 12) — [left_j0-j5, right_j0-j5]
    if 'observation.state.joint' in df.columns:
        obs_joints = np.stack(df['observation.state.joint'].values).astype(np.float32)
    else:
        obs_joints = None
        print("[警告] 未找到 observation.state.joint 字段")

    # observation.gripper_position: (T, 2) — [left_gripper%, right_gripper%]
    if 'observation.gripper_position' in df.columns:
        obs_gripper = np.stack(df['observation.gripper_position'].values).astype(np.float32)
    else:
        obs_gripper = None
        print("[警告] 未找到 observation.gripper_position 字段")

    total_frames = len(actions)
    print(f"  总帧数: {total_frames}")

    # 初始化机械臂
    print("\n[机械臂] 初始化")
    arm_left = PiperArm(can_left, name='左臂')
    arm_right = PiperArm(can_right, name='右臂')
    arm_left.connect()
    arm_right.connect()
    arm_left.enable()
    arm_right.enable()

    # 回零点
    print("\n[机械臂] 回零点 ...")
    arm_left.go_zero(speed_percent=30)
    arm_right.go_zero(speed_percent=30)
    time.sleep(2.0)

    print(f"\n[重放] 开始，共 {total_frames} 帧，速度百分比 {speed_percent}%")
    print("  按 Ctrl+C 中断")
    print("-" * 60)

    try:
        for i, action in enumerate(actions):
            t_start = time.time()
 
            # 重放actions动作
            # 解析 14 维动作: [left_j0-j5, left_gripper(0~1), right_j0-j5, right_gripper(0~1)]
            # act_joints_left = action[:6]            # 弧度
            # act_gripper_left = float(action[6])     # 0~1 → 米
            # act_joints_right = action[7:13]         # 弧度
            # act_gripper_right = float(action[13])   # 0~1 → 米

            # # 百分比 (0~1) → 米 (最大行程 0.105 m)
            # gripper_left_m = np.clip(act_gripper_left, 0.0, 1.0) * 0.105
            # gripper_right_m = np.clip(act_gripper_right, 0.0, 1.0) * 0.105
            
            # print(
            #         f"  [原始动作：帧 {i:4d}/{total_frames - 1}] "
            #         f"L1={act_joints_left}, LG={gripper_left_m}m | "
            #         f"R1={act_joints_right}, RG={gripper_right_m}m"
            #     )
            
            # 重放observation动作
            act_joints_left = obs_joints[i, :6]
            act_joints_right = obs_joints[i, 6:12]
            gripper_left_m = np.clip(float(obs_gripper[i, 0]), 0.0, 1.0) * 0.105
            gripper_right_m = np.clip(float(obs_gripper[i, 1]), 0.0, 1.0) * 0.105
            
            arm_left.send_command(act_joints_left, gripper_left_m, speed_percent=speed_percent)
            arm_right.send_command(act_joints_right, gripper_right_m, speed_percent=speed_percent)

 
            if i % 30 == 0 or i == total_frames - 1:
                obs_j_str = ''
                obs_g_str = ''
                if obs_joints is not None:
                    obs_j_str = f" | obs_jL={obs_joints[i, 0]:.3f}, obs_jR={obs_joints[i, 6]:.3f}"
                if obs_gripper is not None:
                    obs_g_str = f" | obs_gL={obs_gripper[i, 0]:.3f}, obs_gR={obs_gripper[i, 1]:.3f}"
                print(
                    f"  [帧 {i:4d}/{total_frames - 1}] "
                    f"L1={act_joints_left[0]:.3f}, LG={gripper_left_m:.4f}m | "
                    f"R1={act_joints_right[0]:.3f}, RG={gripper_right_m:.4f}m"
                    f"{obs_j_str}{obs_g_str}"
                )

            # 控制帧率
            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[系统] 用户中断重放")
    finally:
        print("\n[系统] 回零点...")
        arm_left.go_zero(speed_percent=30)
        arm_right.go_zero(speed_percent=30)
        time.sleep(2.0)
        arm_left.disconnect()
        arm_right.disconnect()
        print("[重放完成]")


def create_arg_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='在线机械臂推理执行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # YAML 配置 + 命令行覆盖
  python3 online_inference_execution.py --config scripts/online_inference.yaml --fps 15

  # ROS2 平台
  python3 online_inference_execution.py \\
    --model_config pi05_agilex \\
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \\
    --prompt "mobile manipulation"
        """
    )

    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径，命令行参数会覆盖同名配置')

    # 通用参数
    parser.add_argument('--model_config', type=str, default='pi05_agilex',
                        help='模型配置 (default: pi05_agilex)')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='gs://openpi-assets/checkpoints/pi05_base',
                        help='检查点目录')
    parser.add_argument('--prompt', type=str, default='do something',
                        help='任务提示词')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='推理/控制频率 (Hz)')

    # Piper 平台参数
    parser.add_argument('--can_left', type=str, default='can0',
                        help='左臂 CAN 端口')
    parser.add_argument('--can_right', type=str, default='can1',
                        help='右臂 CAN 端口')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='动作执行速度倍率')
    parser.add_argument('--topic_wait_timeout', type=float, default=10.0,
                        help='ROS2 模式下等待关键话题首帧的超时时间(秒)，设为 0 表示无限等待')

    # ROS2/WebSocket 参数
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='推理服务器地址 (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                        help='推理服务器端口 (default: 8000)')
    parser.add_argument('--action_horizon', type=int, default=10,
                        help='每轮推理执行的动作步数 (default: 10)')
    parser.add_argument('--speed_percent', type=int, default=100,
                        help='机械臂运动速度百分比 1-100 (default: 100)')
    parser.add_argument('--max_inferences', type=int, default=0,
                        help='最大推理次数，达到后自动停止；0 表示无限循环 (default: 0)')
    
    parser.add_argument('--replay_flag', type=bool, default=False,
                        help='是否启用重放模式 (default: False)')
    parser.add_argument('--replay_path', type=str, default=None,
                        help='重放模式下的路径 (default: None)')
    parser.add_argument('--replay_id', type=int, default=0,
                        help='重放模式下的 ID (default: 0)')
    return parser


def main():
    # 先预解析 --config, 再用 YAML 设置默认值，最后解析完整参数。
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = create_arg_parser()

    if pre_args.config:
        try:
            config_data = load_yaml_config(pre_args.config)
            parser.set_defaults(**config_data)
        except Exception as e:
            print(f"[错误] 读取 YAML 配置失败: {e}")
            sys.exit(1)

    args = parser.parse_args()
    
    if args.replay_flag:
        if not args.replay_path:
            parser.error('启用重放模式时必须提供 --replay_path 参数')
            return
        replay_action(args)
        return

    run_ros2_inference(args)


if __name__ == '__main__':
    main()
