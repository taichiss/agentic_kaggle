"""Interpolate compatible 12k Mixup model weights for an adaptive LB test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.first_weight <= 1:
        parser.error("first-weight must be in [0, 1]")
    if not args.first.is_file() or not args.second.is_file():
        parser.error("both checkpoints must exist")
    first = torch.load(args.first, map_location="cpu", weights_only=False)
    second = torch.load(args.second, map_location="cpu", weights_only=False)
    for name in ("model_config", "max_peaks", "ppm", "candidate_formulas", "step", "seed"):
        if first[name] != second[name]:
            raise ValueError(f"checkpoint {name} differs")
    if first["step"] != 12000:
        raise ValueError("only completed 12k checkpoints can be blended")
    if not np.array_equal(first["formula_counts"], second["formula_counts"]):
        raise ValueError("checkpoint CHNO catalogs differ")
    if first["state_dict"].keys() != second["state_dict"].keys():
        raise ValueError("checkpoint parameter names differ")
    blended = dict(first)
    weights = {}
    for name, a in first["state_dict"].items():
        b = second["state_dict"][name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"checkpoint parameter {name} differs")
        if not torch.is_floating_point(a):
            if not torch.equal(a, b):
                raise ValueError(f"nonfloat parameter {name} differs")
            weights[name] = a
        else:
            weights[name] = (
                args.first_weight * a.float() + (1 - args.first_weight) * b.float()
            ).to(a.dtype)
    blended["state_dict"] = weights
    blended["blend"] = {
        "first_sha256": sha(args.first),
        "second_sha256": sha(args.second),
        "first_weight": args.first_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blended, args.output)
    receipt = dict(blended["blend"], output_sha256=sha(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
