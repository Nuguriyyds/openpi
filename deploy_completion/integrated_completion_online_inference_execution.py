"""30 Hz AgileX client for joint Training-Paper RTC/completion responses.

The hardware and action conventions are imported from the existing, tested
Training-Paper RTC client. This file adds the prompt state machine,
stale-generation handling, and session artifacts. Completion is evaluated from
the prefix produced by each action request; no independent completion request
or second VLA forward is used.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import datetime as dt
import json
import pathlib
import queue
import subprocess
import sys
import threading
import time
from typing import Any

np: Any = None


_DEFAULT_HEAD_METADATA = "/home/geekplus/develop/ra_ttrtc/openpi/checkpoints/done_head_h768/metadata.json"
_DEFAULT_CONFIG = pathlib.Path(__file__).with_name("integrated_completion_online_inference.yaml")
_TASK_PROMPT_ORDER = (
    "load_bread_into_toaster",
    "activate_toaster",
    "pour_drink_into_cup",
    "place_toasted_bread_on_plate",
)


def _load_reference_client() -> Any:
    """Import the original hardware-tested client and its Training-Paper patches."""

    root = pathlib.Path(__file__).resolve().parents[1]
    deploy_dir = root / "scripts" / "deploy"
    for path in (deploy_dir,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # ra_client is the existing real-robot adapter: it keeps the shared
    # Training-Paper RTC action code, adds the ROS-only service guard, and
    # applies the JPEG transport expected by serve_training_paper_rtc_base.
    import ra_client as wrapper  # noqa: PLC0415

    return wrapper._paper_rtc  # noqa: SLF001 - the adapter exports the patched reference client here.


def _json_safe(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonlEventLogger:
    """Asynchronously append compact, JSON-serializable deployment events."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=12000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="completion-event-log", daemon=True)
        self._thread.start()

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "wall_time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wall_time_unix_s": time.time(),
            "monotonic_s": time.monotonic(),
            **fields,
        }
        # The control loop remains real-time oriented; the bounded logger
        # cannot be allowed to block it.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(_json_safe(record))

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8", buffering=1) as handle:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    continue
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._queue.task_done()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


class TopVideoRecorder:
    """Background H.264 MP4 writer that records only the top camera."""

    def __init__(self, path: pathlib.Path, *, fps: float):
        self.path = path
        self.frame_index_path = path.with_name(f"{path.stem}_frames.jsonl")
        self.ffmpeg_log_path = path.with_name(f"{path.stem}_ffmpeg.log")
        self.fps = float(fps)
        self.error: str | None = None
        self._queue: queue.Queue[tuple[np.ndarray, dict[str, Any]] | None] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="completion-top-video", daemon=True)
        self._thread.start()

    def push(self, snapshot: dict[str, Any], overlay: dict[str, Any]) -> None:
        image = snapshot.get("images", {}).get("cam_top")
        if image is None:
            return
        image = np.asarray(image)
        # Drop a video frame rather than delaying RobotArmService.step.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait((image.copy(), dict(overlay)))

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if image.ndim != 3:
            raise ValueError(f"top camera image must be rank 3, got {image.shape}")
        if image.shape[0] == 3:
            rgb = np.transpose(image, (1, 2, 0))
        elif image.shape[-1] == 3:
            rgb = image
        else:
            raise ValueError(f"top camera image must have three channels, got {image.shape}")
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    def _run(self) -> None:
        import cv2  # noqa: PLC0415

        encoder = None
        encoder_input = None
        encoder_log = None
        frame_index_handle = None
        video_frame_index = 0
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    self._queue.task_done()
                    continue
                image, overlay = item
                try:
                    frame = self._to_bgr(image)
                    if encoder is None:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        height, width = frame.shape[:2]
                        encoder_log = self.ffmpeg_log_path.open("w", encoding="utf-8")
                        encoder = subprocess.Popen(
                            [
                                "ffmpeg",
                                "-hide_banner",
                                "-loglevel",
                                "warning",
                                "-y",
                                "-f",
                                "rawvideo",
                                "-pix_fmt",
                                "bgr24",
                                "-video_size",
                                f"{width}x{height}",
                                "-framerate",
                                f"{self.fps:g}",
                                "-i",
                                "pipe:0",
                                "-an",
                                "-c:v",
                                "libx264",
                                "-preset",
                                "veryfast",
                                "-crf",
                                "20",
                                "-threads",
                                "2",
                                "-pix_fmt",
                                "yuv420p",
                                "-movflags",
                                "+faststart",
                                str(self.path),
                            ],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=encoder_log,
                        )
                        encoder_input = encoder.stdin
                        if encoder_input is None:
                            raise RuntimeError("ffmpeg did not expose its input pipe")
                        frame_index_handle = self.frame_index_path.open("w", encoding="utf-8", buffering=1)
                    text_lines = [
                        f"task={overlay.get('task_index', '?')} gen={overlay.get('prompt_generation', '?')}",
                        f"score={overlay.get('score', 'n/a')} threshold={overlay.get('threshold', '?')}",
                        f"history={overlay.get('history_size', 0)}/3 ready={int(bool(overlay.get('history_ready', False)))}",
                        f"chunk={overlay.get('chunk_id', '?')} idx={overlay.get('chunk_index', '?')}",
                        f"prompt={overlay.get('prompt', '')}",
                    ]
                    if overlay.get("switch_label"):
                        text_lines.append(str(overlay["switch_label"]))
                    for line_index, line in enumerate(text_lines):
                        cv2.putText(
                            frame,
                            line[:120],
                            (8, 20 + line_index * 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (0, 255, 255) if line_index < 2 else (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    encoder_input.write(frame.tobytes())
                    frame_index_handle.write(
                        json.dumps(
                            {
                                "video_frame_index": video_frame_index,
                                "control_step": int(overlay["control_step"]),
                                "capture_monotonic_s": overlay.get("capture_monotonic_s"),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    video_frame_index += 1
                except Exception as exc:
                    # Video is optional; never take down the robot control loop.
                    self.error = f"{type(exc).__name__}: {exc}"
                finally:
                    self._queue.task_done()
        finally:
            if encoder_input is not None:
                with contextlib.suppress(BrokenPipeError):
                    encoder_input.close()
            if encoder is not None:
                return_code = encoder.wait()
                if return_code != 0 and self.error is None:
                    self.error = f"ffmpeg exited with status {return_code}; see {self.ffmpeg_log_path}"
            if frame_index_handle is not None:
                frame_index_handle.close()
            if encoder_log is not None:
                encoder_log.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        if self.error is not None:
            print(f"  [top video] recording failed: {self.error}", file=sys.stderr)


class InferenceBroker:
    """Own the WebSocket used by the asynchronous action planner."""

    def __init__(self, client: Any):
        self.client = client
        self._condition = threading.Condition()
        self._actions: collections.deque[tuple[dict[str, Any], Any]] = collections.deque()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="completion-websocket", daemon=True)
        self._thread.start()

    def submit(self, observation: dict[str, Any]) -> Any:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        with self._condition:
            if self._closed:
                future.set_exception(RuntimeError("inference broker is closed"))
                return future
            self._actions.append((observation, future))
            self._condition.notify()
        return future

    def request_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.submit(observation).result()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._actions)
                if self._closed and not self._actions:
                    return
                observation, future = self._actions.popleft()
            if future.cancelled():
                continue
            try:
                result = self.client.infer(observation)
                if not future.cancelled():
                    future.set_result(result)
            except BaseException as exc:
                if not future.cancelled():
                    future.set_exception(exc)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            while self._actions:
                _, future = self._actions.popleft()
                if not future.done():
                    future.set_exception(RuntimeError("inference broker stopped"))
            self._condition.notify_all()
        self._thread.join(timeout=30.0)
        websocket = getattr(self.client, "_ws", None)
        if websocket is not None:
            with contextlib.suppress(Exception):
                websocket.close()


class ActionChunkPlanner:
    """Generation-aware copy of the original asynchronous chunk planner."""

    def __init__(
        self,
        *,
        paper_rtc: Any,
        broker: InferenceBroker,
        obs_buffer: Any,
        logger: JsonlEventLogger,
        fixed_delay_steps: int,
        delay_mode: str,
        control_hz: float,
        max_delay_steps: int,
    ):
        self.paper_rtc = paper_rtc
        self.broker = broker
        self.obs_buffer = obs_buffer
        self.logger = logger
        self.fixed_delay_steps = max(0, int(fixed_delay_steps))
        self.delay_mode = str(delay_mode or "fixed").lower()
        self.control_hz = float(control_hz)
        self.max_delay_steps = max(0, int(max_delay_steps))
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._request: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None
        self._inflight = False
        self._seq = 0
        self._last_error = ""
        self._last_request_elapsed_s: float | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="completion-action-chunk", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def pending(self) -> bool:
        with self._condition:
            return self._request is not None or self._inflight or self._result is not None

    def last_error(self) -> str:
        with self._condition:
            return self._last_error

    def ready(self) -> bool:
        with self._condition:
            return self._result is not None

    def invalidate_generation(self, generation: int) -> None:
        with self._condition:
            if self._request is not None and int(self._request["prompt_generation"]) != int(generation):
                self._request = None
            self._condition.notify_all()

    def request(
        self,
        *,
        client_step: int,
        local_chunk_id: int,
        local_chunk_index: int,
        prev_leftover_robot: np.ndarray | None,
        prompt: str,
        task_index: int,
        prompt_generation: int,
        reason: str,
    ) -> bool:
        prev_robot = None if prev_leftover_robot is None else np.asarray(prev_leftover_robot, dtype=np.float32).copy()
        prefix_horizon = 0 if prev_robot is None else len(prev_robot)
        cap = min(self.max_delay_steps, prefix_horizon)
        fixed_delay = max(0, min(self.fixed_delay_steps, cap))
        if self.delay_mode == "realtime_ceil" and self._last_request_elapsed_s is not None:
            fixed_delay = max(0, min(int(np.ceil(self._last_request_elapsed_s * self.control_hz)), cap))
        elif self.delay_mode not in {"fixed", "realtime_ceil"}:
            raise ValueError(f"unsupported client chunk delay mode {self.delay_mode!r}")
        request = {
            "client_step": int(client_step),
            "local_chunk_id": int(local_chunk_id),
            "local_chunk_index": int(local_chunk_index),
            "local_remaining": 0 if prev_robot is None else len(prev_robot),
            "prev_leftover_robot": prev_robot,
            "prefix_attention_horizon": prefix_horizon,
            "fixed_delay_steps": fixed_delay,
            "delay_mode": self.delay_mode,
            "prompt": prompt,
            "task_index": int(task_index),
            "prompt_generation": int(prompt_generation),
            "reason": str(reason),
            "request_monotonic_s": time.monotonic(),
        }
        with self._condition:
            if self._request is not None or self._inflight or self._result is not None:
                return False
            self._request = request
            self._condition.notify_all()
            return True

    def pop_ready(self) -> dict[str, Any] | None:
        with self._condition:
            result = self._result
            self._result = None
            return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._stop.is_set() or self._request is not None)
                if self._stop.is_set():
                    return
                request = self._request
                self._request = None
                self._inflight = True
            if request is None:
                continue
            started = time.monotonic()
            try:
                snapshot = self.obs_buffer.get_snapshot(copy_images=True)
                observation_monotonic_s = float(
                    snapshot.pop("_completion_observation_monotonic_s", time.monotonic())
                )
                observation = self.paper_rtc.make_paper_rtc_client_chunk_request(
                    snapshot,
                    request["prompt"],
                    client_step=request["client_step"],
                    local_chunk_id=request["local_chunk_id"],
                    local_chunk_index=request["local_chunk_index"],
                    local_remaining=request["local_remaining"],
                    prev_leftover_robot=request["prev_leftover_robot"],
                    fixed_delay_steps=request["fixed_delay_steps"],
                    prefix_attention_horizon=request["prefix_attention_horizon"],
                )
                observation["_completion_observation_monotonic_s"] = observation_monotonic_s
                observation["_completion_prompt_generation"] = request["prompt_generation"]
                observation["_completion_task_index"] = request["task_index"]
                observation["_completion_request_sequence"] = self._seq
                self.logger.log(
                    "action_chunk_request",
                    client_step=request["client_step"],
                    task_index=request["task_index"],
                    prompt_generation=request["prompt_generation"],
                    prompt=request["prompt"],
                    local_chunk_id=request["local_chunk_id"],
                    local_chunk_index=request["local_chunk_index"],
                    reason=request["reason"],
                    fixed_delay_steps=request["fixed_delay_steps"],
                    prefix_attention_horizon=request["prefix_attention_horizon"],
                )
                output = self.broker.request_action(observation)
                actions = np.asarray(output.get("actions"), dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 14 or actions.shape[0] <= 0:
                    raise RuntimeError(f"action chunk must have shape [H,14], got {actions.shape}")
                finished = time.monotonic()
                self._last_request_elapsed_s = finished - started
                packet = {
                    "robot_actions": actions.copy(),
                    "completion": dict(output.get("completion", {})),
                    "server_timing": dict(output.get("server_timing", {})),
                    "request": dict(request),
                    "client_timing": {
                        "chunk_async_seq": self._seq,
                        "chunk_total_ms": (finished - started) * 1000.0,
                        "chunk_response_time_s": finished,
                    },
                }
                with self._condition:
                    self._result = packet
                    self._last_error = ""
                    self._inflight = False
                    self._seq += 1
                    self._condition.notify_all()
                self.logger.log(
                    "action_chunk_response",
                    client_step=request["client_step"],
                    task_index=request["task_index"],
                    prompt_generation=request["prompt_generation"],
                    prompt=request["prompt"],
                    local_chunk_id=request["local_chunk_id"],
                    action_shape=list(actions.shape),
                    total_ms=(finished - started) * 1000.0,
                    completion=output.get("completion", {}),
                    server_timing=output.get("server_timing", {}),
                )
            except Exception as exc:
                with self._condition:
                    self._last_error = str(exc)
                    self._inflight = False
                    self._condition.notify_all()
                self.logger.log(
                    "action_chunk_error",
                    client_step=request["client_step"],
                    task_index=request["task_index"],
                    prompt_generation=request["prompt_generation"],
                    prompt=request["prompt"],
                    error=str(exc),
                )
                time.sleep(0.05)


class KeyboardReader:
    """Read ``n`` + Enter without blocking the 30 Hz loop."""

    def __init__(self):
        self.commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="completion-keyboard", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import select  # noqa: PLC0415

        terminal_state = None
        try:
            if sys.stdin.isatty():
                import termios  # noqa: PLC0415
                import tty  # noqa: PLC0415

                terminal_state = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not readable:
                    continue
                try:
                    character = sys.stdin.read(1)
                except (EOFError, OSError):
                    return
                if not character:
                    return
                if character.lower() == "n":
                    self.commands.put("n")
        finally:
            if terminal_state is not None:
                import termios  # noqa: PLC0415

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_state)

    def pop(self) -> str | None:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def _load_yaml_defaults(path: pathlib.Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 - keep --help independent of the optional YAML dependency.

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data, dict) and isinstance(data.get("integrated_completion_online_inference"), dict):
        data = data["integrated_completion_online_inference"]
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
    return dict(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--control-hz", dest="control_hz", type=float, default=30.0)
    parser.add_argument("--max-joint-delta-rad", dest="max_joint_delta_rad", type=float, default=0.20)
    parser.add_argument("--obs-send-hz", dest="obs_send_hz", type=float, default=3.75)
    parser.add_argument(
        "--client-chunk-execution",
        dest="client_chunk_execution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--client-chunk-execute-horizon", dest="client_chunk_execute_horizon", type=int, default=0)
    parser.add_argument("--client-chunk-fixed-delay-steps", dest="client_chunk_fixed_delay_steps", type=int, default=5)
    parser.add_argument(
        "--client-chunk-delay-mode",
        dest="client_chunk_delay_mode",
        choices=("fixed", "realtime_ceil"),
        default="fixed",
    )
    parser.add_argument("--client-chunk-max-delay-steps", dest="client_chunk_max_delay_steps", type=int, default=0)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=0)
    parser.add_argument("--robot-endpoint", "--robot_endpoint", dest="robot_endpoint", default="tcp://127.0.0.1:9901")
    parser.add_argument(
        "--robot-service-recv-timeout-ms",
        "--robot_service_recv_timeout_ms",
        dest="robot_service_recv_timeout_ms",
        type=int,
        default=30000,
    )
    parser.add_argument("--startup-init-hold-min-s", dest="startup_init_hold_min_s", type=float, default=0.0)
    parser.add_argument(
        "--startup-init-hold-log-interval-s", dest="startup_init_hold_log_interval_s", type=float, default=1.0
    )
    parser.add_argument("--completion-threshold", "--threshold", dest="completion_threshold", type=float, default=0.6)
    parser.add_argument(
        "--completion-head-metadata",
        dest="completion_head_metadata",
        type=pathlib.Path,
        default=pathlib.Path(_DEFAULT_HEAD_METADATA),
    )
    parser.add_argument(
        "--log-root", dest="log_root", type=pathlib.Path, default=pathlib.Path("logs/completion_deploy")
    )
    parser.add_argument(
        "--record-top-video", dest="record_top_video", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def _server_prompts(server_metadata: dict[str, Any], local_metadata_path: pathlib.Path) -> list[str]:
    completion = server_metadata.get("completion")
    if not isinstance(completion, dict):
        raise ValueError("server metadata does not advertise completion support")
    if completion.get("variant") != "token_query_attention":
        raise ValueError(f"unsupported server completion variant: {completion.get('variant')!r}")
    prompts = completion.get("task_prompts")
    if not isinstance(prompts, list) or len(prompts) != 4 or not all(isinstance(item, str) for item in prompts):
        raise ValueError("server completion metadata must contain exactly four task prompts")
    prompts = [str(item) for item in prompts]
    if local_metadata_path.is_file():
        with local_metadata_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        cache = raw.get("cache_metadata", {})
        local_prompts = cache.get("task_prompts") if isinstance(cache, dict) else None
        if isinstance(local_prompts, dict):
            local_prompts = [local_prompts[key] for key in _TASK_PROMPT_ORDER]
        if isinstance(local_prompts, list) and list(local_prompts) != prompts:
            raise ValueError("local head metadata prompts differ from server head metadata")
    return prompts


def _new_session_dir(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("session_%Y%m%d_%H%M%S_%f")
    path = root / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: pathlib.Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(args: argparse.Namespace) -> None:
    global np  # noqa: PLW0603 - numerical dependencies are intentionally loaded only for execution.
    import numpy as np  # noqa: PLC0415

    if not args.client_chunk_execution:
        raise ValueError("completion deployment requires client chunk execution")
    if args.control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if not 0.0 <= args.completion_threshold <= 1.0:
        raise ValueError("completion threshold must be in [0, 1]")
    if abs(args.control_hz - 30.0) > 1.0e-6:
        raise ValueError("this real-robot deployment requires control_hz=30.0")

    paper_rtc = _load_reference_client()
    client = paper_rtc.websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = client.get_server_metadata()
    prompts = _server_prompts(server_metadata, args.completion_head_metadata)
    completion_metadata = dict(server_metadata["completion"])
    if int(completion_metadata.get("temporal_steps", 0)) != 3:
        raise ValueError("server completion head must use temporal_steps=3")

    session_dir = _new_session_dir(args.log_root)
    event_logger = JsonlEventLogger(session_dir / "events.jsonl")
    session_start = time.time()
    _write_json(
        session_dir / "session.json",
        {
            "session_start_wall_time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "session_start_unix_s": session_start,
            "vla": {
                "config": server_metadata.get("training_paper_rtc", {}).get("config_name"),
                "checkpoint": server_metadata.get("training_paper_rtc", {}).get("checkpoint_dir"),
            },
            "completion_head": completion_metadata,
            "completion_head_metadata": str(args.completion_head_metadata),
            "completion_threshold": float(args.completion_threshold),
            "control_hz": float(args.control_hz),
            "completion_schedule": "each_action_inference",
            "prompts": prompts,
            "server_metadata": server_metadata,
        },
    )
    event_logger.log(
        "session_start",
        session_dir=str(session_dir),
        vla_config=server_metadata.get("training_paper_rtc", {}).get("config_name"),
        completion_head_variant=completion_metadata.get("variant"),
        completion_threshold=args.completion_threshold,
        control_hz=args.control_hz,
        completion_schedule="each_action_inference",
        prompts=prompts,
    )

    class TimestampedObservationBuffer(paper_rtc.ObservationBuffer):
        def __init__(self) -> None:
            super().__init__()
            self._completion_timestamp_lock = threading.Lock()
            self._completion_observation_monotonic_s = 0.0

        def set_snapshot(self, snapshot: dict[str, Any]) -> None:
            with self._completion_timestamp_lock:
                super().set_snapshot(snapshot)
                self._completion_observation_monotonic_s = time.monotonic()

        def get_snapshot(self, copy_images: bool = False) -> dict[str, Any]:
            with self._completion_timestamp_lock:
                snapshot = super().get_snapshot(copy_images=copy_images)
                snapshot["_completion_observation_monotonic_s"] = self._completion_observation_monotonic_s
                return snapshot

    broker = InferenceBroker(client)
    obs_buffer = TimestampedObservationBuffer()
    robot_service = None
    chunk_planner = None
    keyboard = KeyboardReader()
    video = TopVideoRecorder(session_dir / "top_camera.mp4", fps=args.control_hz) if args.record_top_video else None
    task_index = 0
    prompt_generation = 0
    active_prompt = prompts[0]
    episode_complete = False
    last_completion: dict[str, Any] = {"score": None, "history_size": 0, "history_ready": False}
    last_switch_label = ""
    current_robot_chunk: np.ndarray | None = None
    current_chunk_index = 0
    current_chunk_id = -1
    installed_chunk_count = 0
    first_executed_chunk_id = -1
    last_service_action: np.ndarray | None = None
    planner_lock = threading.Lock()
    stop_status = "finished"

    def hold_policy_prefix() -> np.ndarray | None:
        if args.client_chunk_fixed_delay_steps <= 0:
            return None
        snapshot = obs_buffer.get_snapshot(copy_images=False)
        policy_action = paper_rtc.snapshot_to_policy_action(snapshot)
        return np.repeat(policy_action[None, :], args.client_chunk_fixed_delay_steps, axis=0).astype(np.float32)

    def queue_action_request(reason: str, step: int) -> bool:
        with planner_lock:
            queued = chunk_planner.request(
                client_step=step,
                local_chunk_id=current_chunk_id,
                local_chunk_index=current_chunk_index,
                prev_leftover_robot=hold_policy_prefix() if reason in {"initial", "initial_retry", "switch"} else None,
                prompt=active_prompt,
                task_index=task_index,
                prompt_generation=prompt_generation,
                reason=reason,
            )
        if queued:
            event_logger.log(
                "action_chunk_request_queued",
                control_step=step,
                task_index=task_index,
                prompt_generation=prompt_generation,
                reason=reason,
            )
        return queued

    def switch_prompt(source: str, step: int, *, score: float | None = None) -> bool:
        nonlocal task_index, prompt_generation, active_prompt, episode_complete, last_completion
        nonlocal current_robot_chunk, current_chunk_index, current_chunk_id, last_switch_label
        old_task = task_index
        old_generation = prompt_generation
        if task_index >= len(prompts) - 1:
            if episode_complete:
                return False
            prompt_generation += 1
            episode_complete = True
            current_robot_chunk = None
            current_chunk_index = 0
            current_chunk_id = -1
            chunk_planner.invalidate_generation(prompt_generation)
            event_logger.log(
                "episode_complete",
                control_step=step,
                source=source,
                task_index=task_index,
                prompt_generation=prompt_generation,
                score=score,
            )
            last_switch_label = "EPISODE COMPLETE"
            return True
        task_index += 1
        prompt_generation += 1
        active_prompt = prompts[task_index]
        current_robot_chunk = None
        current_chunk_index = 0
        current_chunk_id = -1
        chunk_planner.invalidate_generation(prompt_generation)
        last_completion = {"score": None, "history_size": 0, "history_ready": False}
        event_logger.log(
            f"{source}_prompt_switch",
            control_step=step,
            old_task_index=old_task,
            new_task_index=task_index,
            old_prompt_generation=old_generation,
            prompt_generation=prompt_generation,
            score=score,
            prompt=active_prompt,
        )
        last_switch_label = f"{source.upper()} -> task {task_index}"
        queue_action_request("switch", step)
        return True

    try:
        event_logger.log("robot_service_connect_request", endpoint=args.robot_endpoint)
        robot_service = paper_rtc.RobotArmServiceClient(
            str(args.robot_endpoint),
            prompt=active_prompt,
            recv_timeout_ms=int(args.robot_service_recv_timeout_ms),
        )
        robot_service.connect()
        reset_obs = robot_service.reset()
        obs_buffer.set_snapshot(paper_rtc.robot_service_obs_to_snapshot(reset_obs))
        last_service_action = paper_rtc._DEFAULT_INIT_ACTION.copy()  # noqa: SLF001 - preserve the tested init pose.
        event_logger.log("robot_service_reset", task_index=task_index, prompt_generation=prompt_generation)

        action_horizon = int(server_metadata.get("training_paper_rtc", {}).get("action_horizon", 50))
        server_s_min = int(
            server_metadata.get("training_paper_rtc", {}).get(
                "execution_horizon",
                server_metadata.get("training_paper_rtc", {}).get("execute_horizon", 10),
            )
        )
        execute_horizon = (
            int(args.client_chunk_execute_horizon) if args.client_chunk_execute_horizon > 0 else server_s_min
        )
        execute_horizon = max(1, min(execute_horizon, action_horizon - 1))
        max_delay_steps = (
            action_horizon - 1
            if args.client_chunk_max_delay_steps <= 0
            else min(args.client_chunk_max_delay_steps, action_horizon - 1)
        )
        warmup_observation = paper_rtc.make_paper_rtc_client_chunk_request(
            obs_buffer.get_snapshot(copy_images=True),
            active_prompt,
            client_step=0,
            local_chunk_id=-1,
            local_chunk_index=0,
            local_remaining=0,
            prev_leftover_robot=None,
            fixed_delay_steps=0,
            prefix_attention_horizon=0,
        )
        warmup_observation["_completion_observation_monotonic_s"] = time.monotonic()
        warmup_observation["_completion_prompt_generation"] = prompt_generation
        warmup_observation["_completion_task_index"] = task_index
        warmup_observation["_completion_request_sequence"] = -1
        warmup_observation["_completion_skip_history"] = True
        event_logger.log(
            "action_chunk_request",
            control_step=0,
            task_index=task_index,
            prompt_generation=prompt_generation,
            reason="warmup",
        )
        warmup_start = time.monotonic()
        warmup_output = broker.request_action(warmup_observation)
        warmup_actions = np.asarray(warmup_output.get("actions"), dtype=np.float32)
        event_logger.log(
            "action_chunk_response",
            control_step=0,
            task_index=task_index,
            prompt_generation=prompt_generation,
            reason="warmup",
            action_shape=list(warmup_actions.shape),
            total_ms=(time.monotonic() - warmup_start) * 1000.0,
        )

        chunk_planner = ActionChunkPlanner(
            paper_rtc=paper_rtc,
            broker=broker,
            obs_buffer=obs_buffer,
            logger=event_logger,
            fixed_delay_steps=args.client_chunk_fixed_delay_steps,
            delay_mode=args.client_chunk_delay_mode,
            control_hz=args.control_hz,
            max_delay_steps=max_delay_steps,
        )
        chunk_planner.start()
        queue_action_request("initial", 0)

        hold_steps = 0
        hold_started = time.monotonic()
        while True:
            loop_started = time.monotonic()
            # Inspect readiness without consuming the first packet; the regular
            # control loop installs it and applies the normal generation check.
            first_ready = chunk_planner.ready()
            if first_ready and time.monotonic() - hold_started >= max(0.0, args.startup_init_hold_min_s):
                break
            if not chunk_planner.pending():
                queue_action_request("initial_retry", 0)
            if args.max_steps > 0 and hold_steps >= args.max_steps:
                return
            service_obs, service_done, service_info = robot_service.step(last_service_action.copy())
            obs_buffer.set_snapshot(paper_rtc.robot_service_obs_to_snapshot(service_obs))
            hold_steps += 1
            event_logger.log(
                "startup_init_hold_step",
                control_step=hold_steps - 1,
                service_step=service_info.get("step"),
                done=service_done,
            )
            if service_done:
                return
            time.sleep(max(0.0, 1.0 / args.control_hz - (time.monotonic() - loop_started)))
        event_logger.log("startup_init_hold_complete", hold_steps=hold_steps, elapsed_s=time.monotonic() - hold_started)

        step = 0
        while True:
            loop_started = time.monotonic()
            if args.max_steps > 0 and step >= args.max_steps:
                break
            snapshot = obs_buffer.get_snapshot(copy_images=False)
            switched = False

            if keyboard.pop() == "n" and not episode_complete:
                switched = switch_prompt("manual", step)

            packet = chunk_planner.pop_ready()
            if packet is not None:
                request = dict(packet.get("request", {}))
                packet_generation = int(request.get("prompt_generation", -1))
                completion = dict(packet.get("completion", {}))
                server_generation = int(completion.get("prompt_generation", packet_generation))
                score = completion.get("score")
                event_logger.log(
                    "completion_result",
                    control_step=step,
                    task_index=request.get("task_index"),
                    prompt_generation=packet_generation,
                    server_task_index=completion.get("task_index"),
                    server_prompt_generation=server_generation,
                    request_sequence=completion.get("request_sequence"),
                    observation_monotonic_s=completion.get("observation_monotonic_s"),
                    score=score,
                    logit=completion.get("logit"),
                    history_ready=completion.get("history_ready", False),
                    history_size=completion.get("history_size", 0),
                    history_not_ready_reason=completion.get("history_not_ready_reason"),
                    selected_observation_times=completion.get("selected_observation_times", []),
                    relative_times=completion.get("relative_times", []),
                    target_time_errors=completion.get("target_time_errors", []),
                    head_score_ms=completion.get("head_score_ms", 0.0),
                )
                if episode_complete or packet_generation != prompt_generation or server_generation != prompt_generation:
                    event_logger.log(
                        "action_chunk_drop_stale",
                        control_step=step,
                        current_prompt_generation=prompt_generation,
                        packet_prompt_generation=packet_generation,
                        server_prompt_generation=server_generation,
                        packet_task_index=request.get("task_index"),
                        reason=request.get("reason"),
                    )
                else:
                    last_completion = {
                        "score": score,
                        "history_size": completion.get("history_size", 0),
                        "history_ready": completion.get("history_ready", False),
                    }
                    completion_triggered = (
                        not episode_complete
                        and bool(completion.get("history_ready"))
                        and score is not None
                        and float(score) >= args.completion_threshold
                    )
                    if completion_triggered:
                        event_logger.log(
                            "action_chunk_drop_completed",
                            control_step=step,
                            task_index=task_index,
                            prompt_generation=prompt_generation,
                            score=float(score),
                            threshold=args.completion_threshold,
                            request_step=request.get("client_step"),
                        )
                        switched = switch_prompt("auto", step, score=float(score)) or switched
                    else:
                        current_robot_chunk = np.asarray(packet["robot_actions"], dtype=np.float32).copy()
                        skip = 0 if request.get("reason") == "initial" else int(request.get("fixed_delay_steps", 5))
                        current_chunk_index = max(0, min(skip, len(current_robot_chunk) - 1))
                        installed_chunk_count += 1
                        current_chunk_id = installed_chunk_count
                        event_logger.log(
                            "action_chunk_install",
                            control_step=step,
                            task_index=task_index,
                            prompt_generation=prompt_generation,
                            chunk_id=current_chunk_id,
                            chunk_length=len(current_robot_chunk),
                            chunk_index=current_chunk_index,
                            request_step=request.get("client_step"),
                            reason=request.get("reason"),
                        )

            if episode_complete or current_robot_chunk is None or current_chunk_index >= len(current_robot_chunk):
                action_source = "hold_last_pose"
                raw_policy_action = np.concatenate(
                    [
                        last_service_action[:6],
                        np.array([last_service_action[6] / 0.105], dtype=np.float32),
                        last_service_action[7:13],
                        np.array([last_service_action[13] / 0.105], dtype=np.float32),
                    ],
                    axis=0,
                ).astype(np.float32)
                service_action = last_service_action.copy()
                chunk_step = -1
            else:
                action_source = "action_chunk"
                raw_policy_action = current_robot_chunk[current_chunk_index].copy()
                chunk_step = current_chunk_index
                act_jl = raw_policy_action[:6]
                act_gl = float(raw_policy_action[6]) * 0.105
                act_jr = raw_policy_action[7:13]
                act_gr = float(raw_policy_action[13]) * 0.105
                s_jl = paper_rtc.smooth_action(last_service_action[:6], act_jl, max_delta=args.max_joint_delta_rad)
                s_jr = paper_rtc.smooth_action(last_service_action[7:13], act_jr, max_delta=args.max_joint_delta_rad)
                s_gl = paper_rtc.smooth_action(
                    last_service_action[6:7], np.array([act_gl], dtype=np.float32), max_delta=0.05
                )
                s_gr = paper_rtc.smooth_action(
                    last_service_action[13:14], np.array([act_gr], dtype=np.float32), max_delta=0.05
                )
                service_action = np.concatenate([s_jl, s_gl, s_jr, s_gr], axis=0).astype(np.float32)

            if video is not None:
                video.push(
                    snapshot,
                    {
                        "task_index": task_index,
                        "prompt_generation": prompt_generation,
                        "prompt": active_prompt,
                        "score": last_completion.get("score"),
                        "threshold": args.completion_threshold,
                        "history_size": last_completion.get("history_size", 0),
                        "history_ready": last_completion.get("history_ready", False),
                        "chunk_id": current_chunk_id,
                        "chunk_index": chunk_step,
                        "switch_label": last_switch_label,
                        "control_step": step,
                        "capture_monotonic_s": time.monotonic(),
                    },
                )
            last_switch_label = ""

            service_start = time.monotonic()
            if action_source == "action_chunk" and current_chunk_id != first_executed_chunk_id:
                first_executed_chunk_id = current_chunk_id
                event_logger.log(
                    "action_chunk_first_execute",
                    control_step=step,
                    task_index=task_index,
                    prompt_generation=prompt_generation,
                    chunk_id=current_chunk_id,
                    chunk_index=chunk_step,
                    command_start_monotonic_s=service_start,
                )
            service_obs, service_done, service_info = robot_service.step(service_action)
            obs_buffer.set_snapshot(paper_rtc.robot_service_obs_to_snapshot(service_obs))
            service_ms = (time.monotonic() - service_start) * 1000.0
            last_service_action = service_action.copy()
            if (
                action_source == "action_chunk"
                and current_robot_chunk is not None
                and current_chunk_index < len(current_robot_chunk)
            ):
                event_logger.log(
                    "action_chunk_execute",
                    control_step=step,
                    task_index=task_index,
                    prompt_generation=prompt_generation,
                    chunk_id=current_chunk_id,
                    chunk_index=current_chunk_index,
                )
                current_chunk_index += 1
            event_logger.log(
                "control_step",
                control_step=step,
                task_index=task_index,
                prompt_generation=prompt_generation,
                prompt=active_prompt,
                action_source=action_source,
                chunk_id=current_chunk_id,
                chunk_step=chunk_step,
                service_ms=service_ms,
                service_step=service_info.get("step"),
                done=service_done,
                completion_score=last_completion.get("score"),
                completion_threshold=args.completion_threshold,
                completion_history_size=last_completion.get("history_size", 0),
                completion_history_ready=last_completion.get("history_ready", False),
            )
            if service_done:
                break

            if (
                not episode_complete
                and current_robot_chunk is not None
                and current_chunk_index >= execute_horizon
                and not chunk_planner.pending()
            ):
                leftover = current_robot_chunk[current_chunk_index:action_horizon].copy()
                if len(leftover) > 0:
                    queued = chunk_planner.request(
                        client_step=step,
                        local_chunk_id=current_chunk_id,
                        local_chunk_index=current_chunk_index,
                        prev_leftover_robot=leftover,
                        prompt=active_prompt,
                        task_index=task_index,
                        prompt_generation=prompt_generation,
                        reason="replan",
                    )
                    if queued:
                        event_logger.log(
                            "action_chunk_request_queued",
                            control_step=step,
                            task_index=task_index,
                            prompt_generation=prompt_generation,
                            reason="replan",
                            local_chunk_id=current_chunk_id,
                            local_chunk_index=current_chunk_index,
                        )
            if not episode_complete and current_robot_chunk is None and not chunk_planner.pending():
                queue_action_request("switch", step)

            step += 1
            time.sleep(max(0.0, 1.0 / args.control_hz - (time.monotonic() - loop_started)))
    except KeyboardInterrupt:
        stop_status = "ctrl_c"
    except Exception as exc:
        stop_status = "error"
        event_logger.log("session_error", error=str(exc), task_index=task_index, prompt_generation=prompt_generation)
        raise
    finally:
        keyboard.close()
        if chunk_planner is not None:
            chunk_planner.stop()
        broker.close()
        if robot_service is not None:
            robot_service.stop()
            robot_service.close()
        if video is not None:
            video.close()
        event_logger.log(
            "session_stop",
            status=stop_status,
            task_index=task_index,
            prompt_generation=prompt_generation,
            episode_complete=episode_complete,
        )
        event_logger.close()
        try:
            from render_integrated_completion_session import render_session_html  # noqa: PLC0415

            report_path = render_session_html(session_dir)
            print(f"  [session report] {report_path}")
        except Exception as exc:
            print(f"  [session report] generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    effective_argv = sys.argv[1:] if argv is None else argv
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=pathlib.Path, default=_DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args(effective_argv)
    parser = build_parser()
    if "--help" in effective_argv or "-h" in effective_argv:
        parser.parse_args(effective_argv)
        return
    if pre_args.config:
        parser.set_defaults(**_load_yaml_defaults(pre_args.config))
    args = parser.parse_args(effective_argv)
    run(args)


if __name__ == "__main__":
    main()
