#!/usr/bin/env python
"""Run the pinned organizer baseline on a tiny real-data slice."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0001-host-smoke.toml"


def _load_config(path: Path) -> dict:
    with path.open("rb") as file:
        config = tomllib.load(file)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported smoke config schema")
    return config


def _host_revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def run(config_path: Path) -> dict:
    """Train, infer, round-trip CSV/GEFF, and evaluate a tiny official-baseline run."""
    config = _load_config(config_path)
    source = config["source"]
    data = config["data"]
    train_cfg = config["train"]
    infer_cfg = config["inference"]
    output = config["output"]

    host_repo = COMPETITION_ROOT / source["repository_path"]
    if not host_repo.is_dir():
        raise FileNotFoundError(
            f"organizer repository not found: {host_repo}; run fetch_assets.py baseline"
        )
    revision = _host_revision(host_repo)
    if revision != source["revision"]:
        raise ValueError(
            f"organizer revision mismatch: expected {source['revision']}, got {revision}"
        )

    sys.path[:0] = [str(host_repo / "src"), str(host_repo / "scripts")]
    import numpy as np
    import torch
    from csv_to_geffs import csv_to_geffs
    from evaluate import evaluate_pairs
    from geffs_to_csv import geffs_to_csv
    from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video
    from tracking_cellmot.io import save_graph
    from tracking_cellmot.metrics import summarise
    from train_unet_transformer import train

    sys.path.insert(0, str(COMPETITION_ROOT / "src"))
    from submission import validate_submission

    if not torch.cuda.is_available():
        raise RuntimeError("the pinned organizer training loop requires a CUDA device")

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The organizer model uses max_pool3d backward, which has no deterministic CUDA kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_dir = COMPETITION_ROOT / data["train_dir"]
    dataset_path = train_dir / data["dataset"]
    artifact_dir = COMPETITION_ROOT / output["artifact_dir"]
    pred_dir = artifact_dir / "predicted"
    roundtrip_dir = artifact_dir / "roundtrip"
    submission_path = COMPETITION_ROOT / output["submission_file"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    train(
        data_dir=train_dir,
        fold=0,
        splits_file=artifact_dir / "unused-splits.json",
        method=train_cfg["method"],
        n_epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["learning_rate"]),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        unet_out_channels=int(train_cfg["unet_out_channels"]),
        unet_layers=[int(value) for value in train_cfg["unet_layers"]],
        downsample=tuple(int(value) for value in train_cfg["downsample"]),
        det_loss_weight=float(train_cfg["det_loss_weight"]),
        det_neg_weight=float(train_cfg["det_neg_weight"]),
        max_iters=int(train_cfg["max_iters"]),
        debug_video=dataset_path,
        seed=seed,
        max_frames=int(data["max_frames"]),
        window_size=int(data["window_size"]),
        augmentations=[],
        pool_kernel_um=float(train_cfg["pool_kernel_um"]),
        data_parallel=False,
    )
    train_seconds = time.monotonic() - started

    weights_path = (
        host_repo
        / "weights"
        / train_cfg["method"]
        / "split_0"
        / "edge_predictor_best.pth"
    )
    device = torch.device("cuda")
    model, window_size, downsample = load_model(weights_path, device)
    predict_config = PredictConfig(
        det_threshold=float(infer_cfg["det_threshold"]),
        det_tta=bool(infer_cfg["det_tta"]),
        pool_kernel_um=float(infer_cfg["pool_kernel_um"]),
        threshold=float(infer_cfg["edge_threshold"]),
    )

    started = time.monotonic()
    coords, edges = predict_video(
        model,
        dataset_path,
        device,
        predict_config,
        window_size=window_size,
        max_frames=int(data["max_frames"]),
        downsample=downsample,
    )
    graph = build_graph(coords, edges)
    geff_path = pred_dir / f"{data['dataset']}.geff"
    save_graph(graph, geff_path, overwrite=True)
    inference_seconds = time.monotonic() - started

    geffs_to_csv(pred_dir, submission_path)
    validation = validate_submission(submission_path)
    csv_to_geffs(submission_path, roundtrip_dir, overwrite=True)
    metric_rows, skipped = evaluate_pairs(roundtrip_dir, train_dir)
    metric = summarise(metric_rows)

    result = {
        "experiment_id": config["experiment_id"],
        "source_revision": revision,
        "dataset": data["dataset"],
        "frames": int(data["max_frames"]),
        "device": torch.cuda.get_device_name(0),
        "train_seconds": round(train_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "nodes": graph.num_nodes(),
        "edges": graph.num_edges(),
        "submission": validation,
        "score": metric["score"],
        "skipped": skipped,
        "weights_path": str(weights_path),
        "submission_path": str(submission_path),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
