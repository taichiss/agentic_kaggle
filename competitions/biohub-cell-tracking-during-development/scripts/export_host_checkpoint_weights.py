#!/usr/bin/env python
"""Export a periodic host checkpoint as a frozen Kaggle inference bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

COMPETITION_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_model_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("model config must contain a JSON object")
    return value


def _provenance_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(COMPETITION_ROOT).as_posix()
    except ValueError:
        return path.name


def export_checkpoint(
    checkpoint: Path,
    model_config: Path,
    output_dir: Path,
    *,
    expected_completed_epoch: int,
    weight_view: str,
) -> dict[str, Any]:
    """Extract ``model_state_dict`` while binding it to source provenance."""
    import torch

    if weight_view not in {"raw", "ema"}:
        raise ValueError("weight_view must be 'raw' or 'ema'")
    if expected_completed_epoch <= 0:
        raise ValueError("expected_completed_epoch must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not model_config.is_file():
        raise FileNotFoundError(model_config)

    wrapper = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(wrapper, Mapping):
        raise TypeError("checkpoint wrapper must contain a mapping")
    completed_epoch = int(wrapper.get("completed_epochs", -1))
    if completed_epoch != expected_completed_epoch:
        raise ValueError(
            f"checkpoint completed epoch {completed_epoch} does not match "
            f"expected {expected_completed_epoch}"
        )
    state_dict = wrapper.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("checkpoint model_state_dict must contain a non-empty mapping")
    if not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise TypeError("model_state_dict must map string keys to tensors")
    is_ema_wrapper = "ema_state_dict" in wrapper and "ema_validation" in wrapper
    if weight_view == "ema" and not is_ema_wrapper:
        raise ValueError("EMA export requires an EMA checkpoint wrapper")
    if weight_view == "raw" and is_ema_wrapper:
        raise ValueError("raw export cannot use an EMA checkpoint wrapper")

    config_payload = _json_model_config(model_config)
    embedded_config = wrapper.get("model_config")
    if embedded_config is not None and dict(embedded_config) != config_payload:
        raise ValueError("checkpoint and supplied model configs differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "edge_predictor_best.pth"
    temporary_weights = output_dir / ".edge_predictor_best.pth.tmp"
    torch.save(dict(state_dict), temporary_weights)
    os.replace(temporary_weights, weights_path)
    exported_config = output_dir / "config.json"
    shutil.copy2(model_config, exported_config)

    metadata = {
        "schema_version": 1,
        "completed_epochs": completed_epoch,
        "weight_view": weight_view,
        "source_checkpoint": _provenance_path(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "weights_sha256": _sha256(weights_path),
        "model_config_sha256": _sha256(exported_config),
        "state_dict_tensors": len(state_dict),
    }
    metadata_path = output_dir / "checkpoint-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--completed-epoch", type=int, required=True)
    parser.add_argument("--weight-view", choices=("raw", "ema"), required=True)
    args = parser.parse_args()
    export_checkpoint(
        args.checkpoint.resolve(),
        args.model_config.resolve(),
        args.output_dir.resolve(),
        expected_completed_epoch=args.completed_epoch,
        weight_view=args.weight_view,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
