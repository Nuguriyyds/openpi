"""Send synthetic RTC observations to a running integrated server."""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy


_PROMPT = "Pick up all bread pieces from the bread rack and insert them into the toaster."


def _jpeg() -> bytes:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("unable to encode smoke-test image")
    return encoded.tobytes()


def _request(timestamp: float, sequence: int, *, skip_history: bool = False) -> dict:
    image = _jpeg()
    return {
        "state": np.zeros((12,), dtype=np.float32),
        "gripper_position": np.full((2,), 0.5, dtype=np.float32),
        "images": {
            "cam_top": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": _PROMPT,
        "_ra_image_transport": 1,
        "_paper_rtc_client_chunk_request": True,
        "_paper_rtc_client_step": sequence,
        "_paper_rtc_local_chunk_id": sequence - 1,
        "_paper_rtc_local_chunk_index": 0,
        "_paper_rtc_local_remaining": 0,
        "_paper_rtc_inference_delay_steps": 0,
        "_paper_rtc_prefix_attention_horizon": 0,
        "_completion_observation_monotonic_s": timestamp,
        "_completion_prompt_generation": 0,
        "_completion_task_index": 0,
        "_completion_request_sequence": sequence,
        "_completion_skip_history": skip_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18123)
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    base_time = time.monotonic()
    outputs = [client.infer(_request(base_time - 2.0, -1, skip_history=True))]
    for sequence, offset in enumerate((0.0, 0.5, 1.0)):
        outputs.append(client.infer(_request(base_time + offset, sequence)))

    summary = []
    for output in outputs:
        actions = np.asarray(output["actions"])
        completion = dict(output["completion"])
        summary.append(
            {
                "actions_shape": list(actions.shape),
                "history_ready": completion.get("history_ready"),
                "history_size": completion.get("history_size"),
                "reason": completion.get("history_not_ready_reason"),
                "score": completion.get("score"),
                "relative_times": completion.get("relative_times"),
                "target_time_errors": completion.get("target_time_errors"),
            }
        )
    print(json.dumps(summary, indent=2))
    client._ws.close()  # noqa: SLF001


if __name__ == "__main__":
    main()
