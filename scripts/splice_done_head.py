"""Attach only a trained ``completion_head`` subtree to a VLA checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as model_api


def splice_done_head(
    vla_params: Mapping[str, Any],
    done_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Returns VLA params with only ``completion_head`` taken from done params."""

    if "completion_head" not in done_params:
        raise ValueError("done-head checkpoint does not contain completion_head")
    merged = copy.copy(dict(vla_params))
    merged["completion_head"] = copy.deepcopy(done_params["completion_head"])
    return merged


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-params", type=Path, required=True)
    parser.add_argument("--done-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    vla_params = model_api.restore_params(args.vla_params, restore_type=np.ndarray)
    done_params = model_api.restore_params(args.done_params, restore_type=np.ndarray)
    merged = splice_done_head(vla_params, done_params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(args.output, {"params": merged})
    print(f"Wrote VLA + done head params: {args.output}")


if __name__ == "__main__":
    main()
