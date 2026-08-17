import numpy as np
import pytest

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

    assert merged.keys() == reference.keys()
    assert merged["action_in_proj"]["kernel"].shape == reference["action_in_proj"]["kernel"].shape
    assert merged["action_in_proj"]["kernel"].dtype == reference["action_in_proj"]["kernel"].dtype
    np.testing.assert_array_equal(merged["completion_head"]["kernel"], reference["completion_head"]["kernel"])


def test_s2_loader_does_not_hide_missing_action_weights():
    reference = _reference_params()
    merged = weight_loaders._merge_params(  # noqa: SLF001
        {},
        reference,
        missing_regex=r"completion_head/.*",
    )

    assert set(merged) == {"completion_head"}
    assert "action_in_proj" not in merged


def test_s2_loader_rejects_unexpected_checkpoint_head_when_requested():
    reference = _reference_params()
    loaded = {
        "action_in_proj": {"kernel": np.ones((2, 2), dtype=np.float32)},
        "old_completion_head": {"kernel": np.ones((2, 1), dtype=np.float32)},
    }

    with pytest.raises(ValueError, match="unexpected parameter keys: old_completion_head/kernel"):
        weight_loaders._merge_params(  # noqa: SLF001
            loaded,
            reference,
            missing_regex=r"completion_head/.*",
            reject_unexpected=True,
        )
