#!/usr/bin/env python
"""Sweep detection thresholds on the held-out embryo for a saved A/B model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_ROOT = SCRIPT_DIR.parent
SOURCE_ROOT = COMPETITION_ROOT / "src"
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0005b-backbone-ab-nnunet.toml"

sys.path.insert(0, str(SOURCE_ROOT))
from backbone_ab.config import load_and_validate_config  # noqa: E402


def run(config_path: Path, checkpoint: Path, thresholds: list[float]) -> dict:
    config = load_and_validate_config(config_path)
    if config["backbone"].get("contract", "legacy") == "corrected_v2":
        raise ValueError(
            "corrected_v2 calibration must use "
            "evaluate_backbone_ab_competition_screen.py with the pinned "
            "calibration/report split; calibrate_backbone_ab_detection.py is legacy-only"
        )
    organizer = COMPETITION_ROOT / config["source"]["organizer_repository_path"]
    sys.path[:0] = [str(organizer / "src"), str(organizer / "scripts"), str(SCRIPT_DIR)]

    import torch
    import train_unet_transformer as host_training
    from backbone_ab.backbones import build_joint_model
    from run_host_baseline_training import _build_grouped_folds
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("threshold calibration requires CUDA")

    train_dir = COMPETITION_ROOT / config["data"]["train_dir"]
    folds, _ = _build_grouped_folds(train_dir, config["data"]["group_delimiter"])
    fold = folds[int(config["data"]["fold"])]
    window_size = int(config["data"]["window_size"])
    downsample = tuple(int(value) for value in config["train"]["downsample"])

    def load_video(name: str):
        return host_training.load_dataset_windows(
            train_dir / name,
            window_size=window_size,
            downsample=downsample,
        )

    validation_videos = [load_video(name) for name in fold["test"]]
    all_videos = [load_video(name) for name in fold["train"]] + validation_videos
    max_nodes = max(
        max(window.node_counts)
        for _, windows in all_videos
        for window in windows
    )
    dataset = host_training.FrameWindowDataset(validation_videos, max_nodes=max_nodes)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["train"]["num_workers"]),
        shuffle=False,
    )

    device = torch.device("cuda")
    model = build_joint_model(config["backbone"], host_training).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()

    totals = {
        threshold: {"predicted": 0, "matched": 0, "ground_truth": 0}
        for threshold in thresholds
    }
    pool_kernel_um = float(config["train"]["pool_kernel_um"])
    with torch.inference_mode():
        for batch in loader:
            images = batch["imgs"].to(device, dtype=torch.float32)
            coords = batch["coords"].to(device)
            masks = batch["masks"].to(device)
            image_shape = tuple(batch["image_shape"][0].tolist())
            voxel_size = tuple(batch["voxel_size"][0].tolist())
            _, detection_logits = model.encode(images)
            for threshold in thresholds:
                raw_threshold = torch.logit(torch.tensor(threshold)).item()
                for frame_index in range(images.shape[1]):
                    _, _, detected_mask, matches = host_training.detect_and_match(
                        detection_logits[frame_index],
                        coords[:, frame_index],
                        masks[:, frame_index],
                        image_shape,
                        det_threshold=raw_threshold,
                        voxel_size=voxel_size,
                        pool_kernel_um=pool_kernel_um,
                        frame_index=frame_index,
                        window_size=window_size,
                    )
                    totals[threshold]["predicted"] += int(detected_mask.sum().item())
                    totals[threshold]["matched"] += sum(
                        int((sample_matches >= 0).sum().item())
                        for sample_matches in matches
                    )
                    totals[threshold]["ground_truth"] += int(
                        masks[:, frame_index].sum().item()
                    )

    rows = []
    for threshold in thresholds:
        counts = totals[threshold]
        predicted = counts["predicted"]
        matched = counts["matched"]
        ground_truth = counts["ground_truth"]
        precision = matched / max(predicted, 1)
        recall = matched / max(ground_truth, 1)
        rows.append(
            {
                "threshold": threshold,
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            }
        )
    result = {"checkpoint": str(checkpoint), "thresholds": rows}
    output = checkpoint.parent / f"{checkpoint.stem}_detection_calibration.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    )
    args = parser.parse_args()
    run(args.config.resolve(), args.checkpoint.resolve(), args.thresholds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
