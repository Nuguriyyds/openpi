"""Configuration and numerically stable losses for completion training."""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp

CompletionTrainingStage = Literal["disabled", "action", "head"]


@dataclasses.dataclass(frozen=True)
class CompletionTrainingConfig:
    """Options for the staged breakfast action/completion training workflow.

    ``action`` trains only the TTRTC action path while using the persisted split
    manifest, but it does not require completion labels. ``head`` loads the S1
    checkpoint, audits completion labels, and trains only the completion head.
    ``disabled`` preserves the legacy training path.
    """

    stage: CompletionTrainingStage = "disabled"
    label_key: str = "completion"
    split_manifest_path: str | None = None
    split_seed: int = 42
    episodes_per_group: int = 4
    val_groups: int = 5
    test_groups: int = 5
    val_interval: int = 1_000

    warmup_steps: int = 500
    peak_lr: float = 1.0e-4
    decay_lr: float = 1.0e-5
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.stage not in ("disabled", "action", "head"):
            raise ValueError(f"unsupported completion training stage: {self.stage!r}")
        if not self.label_key:
            raise ValueError("completion.label_key must not be empty")
        if self.episodes_per_group <= 0:
            raise ValueError("completion.episodes_per_group must be positive")
        if self.val_groups <= 0:
            raise ValueError("completion.val_groups must be positive for the persisted staged split")
        if self.test_groups < 0:
            raise ValueError("completion.test_groups must be non-negative")
        if self.val_interval <= 0:
            raise ValueError("completion.val_interval must be positive")
        if self.warmup_steps < 0:
            raise ValueError("completion.warmup_steps must be non-negative")
        if self.peak_lr <= 0 or self.decay_lr < 0:
            raise ValueError("completion learning rates must be non-negative and peak_lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("completion.weight_decay must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("completion.gradient_clip_norm must be positive")

    @property
    def uses_completion_data(self) -> bool:
        """Whether the persisted breakfast split is required."""

        return self.stage != "disabled"

    @property
    def trains_completion_head(self) -> bool:
        return self.stage == "head"

    @property
    def requires_completion_labels(self) -> bool:
        """Whether the raw dataset must contain audited completion labels."""

        return self.stage == "head"


def positive_class_weight(negative_count: int, positive_count: int) -> float:
    """Computes the fixed train-only positive weight, capped at 50."""

    if positive_count <= 0:
        raise ValueError("train split has no positive completion labels")
    if negative_count < 0:
        raise ValueError("negative completion count must be non-negative")
    return min(negative_count / positive_count, 50.0)


def weighted_bce_with_logits(logits: jax.Array, targets: jax.Array, pos_weight: float | jax.Array) -> jax.Array:
    """Stable per-example BCE-with-logits with a positive-class weight."""

    logits = jnp.asarray(logits, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    if logits.shape != targets.shape:
        raise ValueError(f"completion logits shape {logits.shape} does not match targets shape {targets.shape}")
    pos_weight = jnp.asarray(pos_weight, dtype=jnp.float32)
    return targets * pos_weight * jax.nn.softplus(-logits) + (1.0 - targets) * jax.nn.softplus(logits)


def bce_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Stable unweighted per-example BCE used for validation reporting."""

    return weighted_bce_with_logits(logits, targets, 1.0)
