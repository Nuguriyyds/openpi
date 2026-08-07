import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_agilex_example() -> dict:
    """Creates a random input example for the AgileX policy."""
    return {
        "state": np.ones((12,)),
        "gripper_position": np.random.rand(2),
        "images": {
            "cam_top": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AgileXInputs(transforms.DataTransformFn):
    """Inputs for the AgileX LeRobot policy format used by the local TTRTC setup."""

    adapt_to_pi: bool = True
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_top", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_agilex(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_top"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgileXOutputs(transforms.DataTransformFn):
    """Outputs for the AgileX policy."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}


def _decode_agilex(data: dict, *, adapt_to_pi: bool = False) -> dict:
    del adapt_to_pi
    state_arm = np.asarray(data["state"])
    gripper = np.asarray(data["gripper_position"])
    state = np.concatenate([state_arm[:6], gripper[:1], state_arm[6:], gripper[1:]], axis=0)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return einops.rearrange(img, "c h w -> h w c")

    data = dict(data)
    data["images"] = {name: convert_image(img) for name, img in data["images"].items()}
    data["state"] = state
    return data
