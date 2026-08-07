"""Training-time RTC configuration.

This module only defines the training switch and its extra parameters.  The
Pi0 model consumes the config when the training loop asks for the TTRTC loss.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class TrainingTimeRTCConfig:
    """Options for prefix-conditioned training-time RTC.

    The default leaves regular pi0/pi0.5 training unchanged.  When enabled, the
    Pi0 loss samples an inference-delay prefix, feeds that prefix as clean action
    tokens, sets those prefix timesteps to OpenPI's clean endpoint, and masks
    prefix tokens out of the loss.
    """

    enabled: bool = False
    # Matches the local TTRTC helper: a value of 5 samples delays from [0, 5).
    simulated_delay: int = 5
    delay_sampling: Literal["exponential", "uniform", "fixed"] = "exponential"
    fixed_delay: int | None = None
    clean_timestep: float = 0.0
    # "reference" preserves the local TTRTC helper's non-prefix normalization.
    loss_normalization: Literal["reference", "token_mean"] = "reference"
