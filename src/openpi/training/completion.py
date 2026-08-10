"""Configuration and numerically stable losses for completion training."""

from __future__ import annotations

import dataclasses
from typing import Final, Literal

import jax
import jax.numpy as jnp

CompletionTrainingStage = Literal["disabled", "action", "head"]
CompletionObjective = Literal["binary", "progress"]
PROGRESS_BIN_COUNT: Final = 10


@dataclasses.dataclass(frozen=True)
class CompletionTrainingConfig:
    """Options for the staged breakfast action/completion training workflow.

    ``action`` trains only the TTRTC action path while using the persisted split
    manifest, but it does not require head labels. ``head`` loads the S1
    checkpoint, audits labels for the selected objective, and trains only the
    completion-head parameter path. ``disabled`` preserves the legacy path.
    """

    stage: CompletionTrainingStage = "disabled"
    # ``binary`` preserves the original last-two-frames completion task.
    # ``progress`` regresses a continuous, within-subtask [0, 1] target.
    objective: CompletionObjective = "binary"
    label_key: str = "completion"
    split_manifest_path: str | None = None
    # A labeled derivative can live under a different LeRobot repo_id while
    # retaining the same episodes as the source action dataset. When set, this
    # is the canonical dataset identity recorded in and validated against the
    # persisted split manifest; the loader still reads ``data.repo_id``.
    split_manifest_repo_id: str | None = None
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

    # The progress objective applies Huber to sigmoid(logits), not to logits.
    huber_delta: float = 0.1

    # Focal loss parameters.  When ``focal_gamma > 0`` the head training uses
    # focal loss instead of weighted BCE.  ``focal_alpha`` balances the positive
    # class (same role as ``pos_weight`` but applied inside the focal term).
    focal_gamma: float = 0.0
    focal_alpha: float = 0.25

    # Optional diagnostic sampler for the completion head. It constructs every
    # training batch from a fixed mixture of positive frames, negatives close
    # to the positive suffix, and ordinary early negatives. Validation always
    # keeps the original frame distribution.
    balanced_sampling: bool = False
    balanced_positive_fraction: float = 0.25
    balanced_hard_negative_fraction: float = 0.25
    hard_negative_window: int = 16
    train_episode_limit: int | None = None
    bce_pos_weight_override: float | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("disabled", "action", "head"):
            raise ValueError(f"unsupported completion training stage: {self.stage!r}")
        if self.objective not in ("binary", "progress"):
            raise ValueError(f"unsupported completion objective: {self.objective!r}")
        if not self.label_key:
            raise ValueError("completion.label_key must not be empty")
        if self.split_manifest_repo_id is not None and not self.split_manifest_repo_id:
            raise ValueError("completion.split_manifest_repo_id must not be empty when set")
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
        if self.huber_delta <= 0:
            raise ValueError("completion.huber_delta must be positive")
        if self.focal_gamma < 0:
            raise ValueError("completion.focal_gamma must be non-negative")
        if not 0.0 < self.focal_alpha < 1.0:
            raise ValueError("completion.focal_alpha must be in (0, 1)")
        if not 0.0 < self.balanced_positive_fraction < 1.0:
            raise ValueError("completion.balanced_positive_fraction must be in (0, 1)")
        if not 0.0 <= self.balanced_hard_negative_fraction < 1.0:
            raise ValueError("completion.balanced_hard_negative_fraction must be in [0, 1)")
        if self.balanced_positive_fraction + self.balanced_hard_negative_fraction >= 1.0:
            raise ValueError("completion balanced positive and hard-negative fractions must sum to less than 1")
        if self.hard_negative_window <= 0:
            raise ValueError("completion.hard_negative_window must be positive")
        if self.train_episode_limit is not None and self.train_episode_limit <= 0:
            raise ValueError("completion.train_episode_limit must be positive when set")
        if self.bce_pos_weight_override is not None and self.bce_pos_weight_override <= 0:
            raise ValueError("completion.bce_pos_weight_override must be positive when set")
        if self.balanced_sampling and self.stage != "head":
            raise ValueError("completion.balanced_sampling is only supported for stage 'head'")
        if self.train_episode_limit is not None and self.stage != "head":
            raise ValueError("completion.train_episode_limit is only supported for stage 'head'")
        if self.objective == "progress":
            if self.stage != "head":
                raise ValueError("completion.objective='progress' is only supported for stage 'head'")
            if self.focal_gamma != 0.0:
                raise ValueError("progress objective does not support focal loss")
            if self.balanced_sampling:
                raise ValueError("progress objective uses progress-stratified sampling, not balanced_sampling")
            if self.bce_pos_weight_override is not None:
                raise ValueError("progress objective does not support bce_pos_weight_override")

    @property
    def uses_completion_data(self) -> bool:
        """Whether the persisted breakfast split is required."""

        return self.stage != "disabled"

    @property
    def trains_completion_head(self) -> bool:
        return self.stage == "head"

    @property
    def requires_completion_labels(self) -> bool:
        """Whether the raw dataset must contain audited head labels."""

        return self.stage == "head"

    @property
    def uses_focal_loss(self) -> bool:
        """Whether head training uses focal loss instead of weighted BCE."""

        return self.objective == "binary" and self.focal_gamma > 0.0

    @property
    def uses_progress_objective(self) -> bool:
        """Whether head training regresses within-subtask progress."""

        return self.stage == "head" and self.objective == "progress"

    @property
    def uses_progress_stratified_sampling(self) -> bool:
        """Whether the training loader must use the progress sampler."""

        return self.uses_progress_objective


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


def focal_loss_with_logits(logits: jax.Array, targets: jax.Array, *, gamma: float, alpha: float) -> jax.Array:
    """Numerically stable focal loss with logits.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    where ``p_t`` is the predicted probability for the true class.  The
    ``(1 - p_t)^gamma`` term down-weights well-classified examples, letting the
    model focus on hard examples — effective for extreme class imbalance.

    Args:
        logits: Raw model logits, shape ``[batch]``.
        targets: Binary targets (0 or 1), same shape as ``logits``.
        gamma: Focusing parameter; ``0`` reduces to standard weighted BCE.
        alpha: Weight for the positive class (``1 - alpha`` for negative).
    """

    logits = jnp.asarray(logits, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    if logits.shape != targets.shape:
        raise ValueError(f"completion logits shape {logits.shape} does not match targets shape {targets.shape}")

    # Per-example cross-entropy: softplus(-|logit|) adjusted by sign.
    # For positive target: softplus(-logit); for negative: softplus(logit).
    # This is equivalent to -log(sigmoid(logit * (2*target - 1))).
    signed_logits = logits * (2.0 * targets - 1.0)
    cross_entropy = jax.nn.softplus(-signed_logits)

    # p_t = probability of the true class.
    p_t = jnp.exp(-cross_entropy)

    # Focal modulating factor: (1 - p_t)^gamma.
    focal_weight = (1.0 - p_t) ** gamma

    # alpha_t: alpha for positive, (1 - alpha) for negative.
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)

    return alpha_t * focal_weight * cross_entropy


def progress_predictions_from_logits(logits: jax.Array) -> jax.Array:
    """Maps FP32 completion logits to bounded continuous progress predictions."""

    return jax.nn.sigmoid(jnp.asarray(logits, dtype=jnp.float32))


def progress_huber_loss(
    logits: jax.Array,
    targets: jax.Array,
    *,
    delta: float = 0.1,
) -> jax.Array:
    """Per-example Huber loss on ``sigmoid(logits)`` for progress regression."""

    if delta <= 0:
        raise ValueError("progress Huber delta must be positive")
    predictions = progress_predictions_from_logits(logits)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    if predictions.shape != targets.shape:
        raise ValueError(f"progress logits shape {predictions.shape} does not match targets shape {targets.shape}")
    absolute_error = jnp.abs(predictions - targets)
    quadratic = jnp.minimum(absolute_error, jnp.asarray(delta, dtype=jnp.float32))
    linear = absolute_error - quadratic
    return 0.5 * jnp.square(quadratic) + jnp.asarray(delta, dtype=jnp.float32) * linear
