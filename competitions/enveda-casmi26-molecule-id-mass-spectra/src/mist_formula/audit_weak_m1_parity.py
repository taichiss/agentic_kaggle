"""Check zero-mix CUDA step one against the frozen A and strict M1 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-history", type=Path, required=True)
    parser.add_argument("--m1-zero-mix", type=Path, required=True)
    parser.add_argument("--weak-zero-mix", type=Path, required=True)
    parser.add_argument("--weak-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a_event = json.loads(args.a_history.read_text())[0]
    weak_event = json.loads(args.weak_history.read_text())[0]
    losses = ("loss", "bce", "ranking")
    if any(a_event[name] != weak_event[name] for name in losses):
        raise AssertionError("Zero-mix loss differs from frozen A")
    if weak_event["mix_attempted"] != 0 or weak_event["mix_applied"] != 0:
        raise AssertionError("The parity step must use no mixup")
    m1 = torch.load(args.m1_zero_mix, map_location="cpu", weights_only=False)
    weak = torch.load(args.weak_zero_mix, map_location="cpu", weights_only=False)
    weights_m1, weights_weak = m1["state_dict"], weak["state_dict"]
    tensor_names_equal = list(weights_m1) == list(weights_weak)
    tensors_equal = tensor_names_equal and all(
        torch.equal(weights_m1[name], weights_weak[name]) for name in weights_m1
    )
    catalog_equal = np.array_equal(m1["formula_counts"], weak["formula_counts"])
    if not tensors_equal or not catalog_equal:
        raise AssertionError("WEAK zero-mix model differs from strict M1")
    report = {
        "losses_equal_to_actual_A": True,
        "losses": {name: weak_event[name] for name in losses},
        "state_dict_tensors": len(weights_m1),
        "state_dict_exact_equal_to_M1": True,
        "formula_catalog_exact_equal_to_M1": True,
        "input_sha256": {
            name: sha(getattr(args, name))
            for name in ("a_history", "m1_zero_mix", "weak_zero_mix", "weak_history")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
