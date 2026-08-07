import numpy as np
import pytest

from openpi.shared import array_typing as at
from openpi.training import weight_loaders


def _reference_params():
    return {
        "action_in_proj": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "completion_head": {"kernel": np.zeros((2, 1), dtype=np.float32)},
    }


def test_s2_loader_allows_only_random_completion_head_initialization():
    reference = _reference_params()
    loaded_s1 = {"action_in_proj": {"kernel": np.ones((2, 2), dtype=np.float32)}}

    merged = weight_loaders._merge_params(  # noqa: SLF001
        loaded_s1,
        reference,
        missing_regex=r"completion_head/.*",
    )

    at.check_pytree_equality(expected=reference, got=merged, check_shapes=True, check_dtypes=True)
    np.testing.assert_array_equal(merged["completion_head"]["kernel"], reference["completion_head"]["kernel"])


def test_s2_loader_does_not_hide_missing_action_weights():
    reference = _reference_params()
    merged = weight_loaders._merge_params(  # noqa: SLF001
        {},
        reference,
        missing_regex=r"completion_head/.*",
    )

    with pytest.raises(ValueError, match="different structure"):
        at.check_pytree_equality(expected=reference, got=merged, check_shapes=True, check_dtypes=True)
