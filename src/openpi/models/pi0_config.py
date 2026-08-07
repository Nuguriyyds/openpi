from collections.abc import Collection
import dataclasses
from typing import TYPE_CHECKING, Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.completion import CompletionHeadConfig
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


ParameterGroup = Literal["vlm", "action", "completion"]

_ACTION_MODULES = frozenset(
    {
        "action_in_proj",
        "action_out_proj",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "time_mlp_in",
        "time_mlp_out",
    }
)


def classify_parameter_path(path: nnx.filterlib.PathParts) -> ParameterGroup:
    """Classifies a Pi0 parameter using the audited NNX module layout.

    PaliGemma's Linen bridge names the first (VLM) expert without a suffix and
    the second (action) expert with ``_1``. This function is intentionally
    strict: a new or renamed parameter fails the startup audit instead of being
    silently frozen or optimized in the wrong group.
    """

    parts = tuple(str(part) for part in path)
    if not parts:
        raise ValueError("encountered an empty Pi0 parameter path")
    if parts[0] == "completion_head":
        return "completion"
    if parts[0] in _ACTION_MODULES:
        return "action"
    if parts[:2] == ("PaliGemma", "img"):
        return "vlm"
    if parts[:2] == ("PaliGemma", "llm"):
        return "action" if any(part.endswith("_1") for part in parts[2:]) else "vlm"
    raise ValueError(f"unrecognized Pi0 NNX parameter path: {'/'.join(parts)}")


@dataclasses.dataclass(frozen=True)
class FreezeVLMFilter:
    """NNX filter that freezes exactly the audited vision and VLM parameters."""

    def __call__(self, path: nnx.filterlib.PathParts, value: Any) -> bool:
        del value
        return classify_parameter_path(path) == "vlm"


@dataclasses.dataclass(frozen=True)
class FreezeAllExceptCompletionFilter:
    """NNX filter used by S2 to freeze the VLM and the full action path."""

    def __call__(self, path: nnx.filterlib.PathParts, value: Any) -> bool:
        del value
        return classify_parameter_path(path) != "completion"


@dataclasses.dataclass(frozen=True)
class ParameterAudit:
    frozen_vlm: tuple[str, ...]
    frozen_action: tuple[str, ...]
    frozen_completion: tuple[str, ...]
    trainable_action: tuple[str, ...]
    trainable_completion: tuple[str, ...]


def audit_frozen_vlm_parameters(
    model: nnx.Module,
    freeze_filter: nnx.filterlib.Filter,
    *,
    trainable_groups: Collection[ParameterGroup] = ("action", "completion"),
) -> ParameterAudit:
    """Audits every real NNX path against the stage's expected trainable groups."""

    expected_trainable = frozenset(trainable_groups)
    invalid_groups = expected_trainable - {"action", "completion"}
    if invalid_groups:
        raise ValueError(f"invalid trainable parameter groups: {sorted(invalid_groups)}")

    all_params = nnx.state(model, nnx.Param).flat_state()
    frozen_paths = {tuple(path) for path in nnx.state(model, nnx.All(nnx.Param, freeze_filter)).flat_state()}
    frozen: dict[ParameterGroup, list[str]] = {"vlm": [], "action": [], "completion": []}
    trainable: dict[ParameterGroup, list[str]] = {"vlm": [], "action": [], "completion": []}
    for path, variable_state in all_params.items():
        group = classify_parameter_path(path)
        joined = "/".join(str(part) for part in path)
        is_frozen = tuple(path) in frozen_paths
        should_train = group in expected_trainable
        if should_train and is_frozen:
            raise ValueError(f"{group} parameter is unexpectedly frozen: {joined}")
        if not should_train and not is_frozen:
            raise ValueError(f"{group} parameter is unexpectedly trainable: {joined}")
        if group == "completion" and variable_state.value.dtype != jnp.float32:
            raise ValueError(f"completion head parameter must be float32, got {variable_state.value.dtype} at {joined}")
        (frozen if is_frozen else trainable)[group].append(joined)

    if not frozen["vlm"]:
        raise ValueError("parameter audit found no VLM parameters")
    if not frozen["action"] and not trainable["action"]:
        raise ValueError("parameter audit found no action parameters")
    for group in expected_trainable:
        if not trainable[group]:
            raise ValueError(f"parameter audit found no trainable {group} parameters")
    return ParameterAudit(
        frozen_vlm=tuple(sorted(frozen["vlm"])),
        frozen_action=tuple(sorted(frozen["action"])),
        frozen_completion=tuple(sorted(frozen["completion"])),
        trainable_action=tuple(sorted(trainable["action"])),
        trainable_completion=tuple(sorted(trainable["completion"])),
    )


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Disabled by default so existing models and checkpoints retain their exact
    # structure. When enabled, the head input width is read from PaliGemma.
    completion_head: CompletionHeadConfig = dataclasses.field(default_factory=CompletionHeadConfig)

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0  # noqa: PLC0415

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_vlm_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the visual encoder and PaliGemma backbone, but not the action expert/head."""

        return FreezeVLMFilter()

    def get_completion_head_only_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze every audited parameter except the FP32 completion head."""

        return FreezeAllExceptCompletionFilter()
