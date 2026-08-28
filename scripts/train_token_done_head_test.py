import json

import numpy as np

from scripts import train_token_done_head


def _write_shard(root, *, index, start, stop, tokens, masks, row_arrays=None):
    path = root / f"shard_{index}"
    path.mkdir(parents=True)
    np.save(path / "tokens.npy", tokens)
    np.save(path / "masks.npy", masks)
    if row_arrays is not None:
        for name, values in row_arrays.items():
            np.save(path / f"{name}.npy", values)
    metadata = {
        "feature_plan_sha256": "same-plan",
        "feature_count": 4,
        "feature_dim": 2,
        "token_count": 2,
        "row_count": 2,
        "feature_shard": {"index": index, "count": 2, "start": start, "stop": stop},
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_token_cache_reads_histories_across_shards(tmp_path):
    row_arrays = {
        "history_indices": np.asarray([[0, 2, 3], [1, 2, 0]], dtype=np.int32),
        "labels": np.asarray([0, 1], dtype=np.uint8),
        "splits": np.asarray(["train", "val"], dtype="<U5"),
        "sample_ids": np.asarray(["a", "b"], dtype="<U96"),
    }
    first = np.arange(8, dtype=np.float16).reshape(2, 2, 2)
    second = np.arange(8, 16, dtype=np.float16).reshape(2, 2, 2)
    _write_shard(
        tmp_path,
        index=0,
        start=0,
        stop=2,
        tokens=first,
        masks=np.asarray([[1, 1], [1, 0]], dtype=np.bool_),
        row_arrays=row_arrays,
    )
    _write_shard(
        tmp_path,
        index=1,
        start=2,
        stop=4,
        tokens=second,
        masks=np.asarray([[1, 0], [1, 1]], dtype=np.bool_),
        row_arrays=row_arrays,
    )

    cache = train_token_done_head.load_token_cache(tmp_path)
    tokens, masks = cache.histories(np.asarray([0]))

    assert tokens.shape == (1, 3, 2, 2)
    assert masks.shape == (1, 3, 2)
    np.testing.assert_array_equal(tokens[0, 0], first[0])
    np.testing.assert_array_equal(tokens[0, 1], second[0])
    np.testing.assert_array_equal(tokens[0, 2], second[1])
