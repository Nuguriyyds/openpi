"""Verify that the legacy and raw-prefix action APIs return identical actions."""

from __future__ import annotations

import pathlib
import sys

import numpy as np


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for path in (_ROOT, _SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import serve_training_paper_rtc_base as _base  # noqa: E402
import serve_training_paper_rtc_policy as _rtc  # noqa: E402


_CONFIG = "agilex_make_breakfast_generalize_720_subtasks_1500_relative_balanced_ttrtc"
_CHECKPOINT = (
    "/home/geekplus/develop/ra_ttrtc/openpi/checkpoints/"
    "agilex_make_breakfast_generalize_720_subtasks_1500_relative_balanced_ttrtc/39999"
)
_PROMPT = "Pick up all bread pieces from the bread rack and insert them into the toaster."


def main() -> None:
    _rtc._install_training_paper_rtc_overrides()
    args = _base.Args(policy=_base.Checkpoint(config=_CONFIG, dir=_CHECKPOINT))
    policy = _base.create_policy(args)
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    observation = {
        "state": np.zeros((12,), dtype=np.float32),
        "gripper_position": np.full((2,), 0.5, dtype=np.float32),
        "images": {
            "cam_top": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": _PROMPT,
    }
    noise = np.random.default_rng(42).standard_normal((50, 32)).astype(np.float32)
    action_prefix = np.zeros((50, 32), dtype=np.float32)
    legacy = policy.infer(
        observation,
        noise=noise,
        action_prefix=action_prefix,
        delay=0,
        num_steps=5,
    )
    integrated = policy.infer_with_raw_prefix(
        observation,
        noise=noise,
        action_prefix=action_prefix,
        delay=0,
        num_steps=5,
    )
    legacy_actions = np.asarray(legacy["actions"], dtype=np.float32)
    integrated_actions = np.asarray(integrated["actions"], dtype=np.float32)
    if not np.array_equal(legacy_actions, integrated_actions):
        raise AssertionError(f"action API mismatch: max abs diff={np.max(np.abs(legacy_actions-integrated_actions))}")
    print(
        {
            "actions_shape": legacy_actions.shape,
            "max_abs_action_diff": 0.0,
            "raw_prefix_shape": tuple(integrated["raw_prefix_out"].shape),
            "raw_prefix_mask_shape": tuple(integrated["raw_prefix_mask"].shape),
        }
    )


if __name__ == "__main__":
    main()
