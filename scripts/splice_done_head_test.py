import numpy as np
import pytest

from scripts import splice_done_head


def test_splice_replaces_only_completion_head():
    vla = {
        "PaliGemma": {"kernel": np.asarray([1.0])},
        "action_out_proj": {"kernel": np.asarray([2.0])},
    }
    trained = {
        "PaliGemma": {"kernel": np.asarray([999.0])},
        "action_out_proj": {"kernel": np.asarray([999.0])},
        "completion_head": {"output": {"kernel": np.asarray([3.0])}},
    }

    merged = splice_done_head.splice_done_head(vla, trained)

    np.testing.assert_array_equal(merged["PaliGemma"]["kernel"], [1.0])
    np.testing.assert_array_equal(merged["action_out_proj"]["kernel"], [2.0])
    np.testing.assert_array_equal(merged["completion_head"]["output"]["kernel"], [3.0])
    assert "completion_head" not in vla


def test_splice_requires_a_done_head():
    with pytest.raises(ValueError, match="does not contain completion_head"):
        splice_done_head.splice_done_head({"backbone": {}}, {"backbone": {}})
