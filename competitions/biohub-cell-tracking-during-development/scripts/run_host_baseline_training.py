#!/usr/bin/env python
"""Train the pinned organizer baseline on a reproducible embryo-grouped fold."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

from run_host_baseline_smoke import (
    COMPETITION_ROOT,
    EPOCH_PATTERN,
    _epoch_history,
    _host_revision,
    _init_wandb,
    _load_config,
    _sha256,
)

DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0004-host-baseline-fold0-50e.toml"


class _LiveEpochTee(io.TextIOBase):
    """Mirror stdout, retain it, and stream completed epoch metrics to W&B."""

    def __init__(self, capture: io.StringIO, wandb_run, history_path: Path) -> None:
        self.capture = capture
        self.wandb_run = wandb_run
        self.history_path = history_path
        self.pending = ""

    def write(self, text: str) -> int:
        sys.__stdout__.write(text)
        self.capture.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = EPOCH_PATTERN.search(line)
            if match:
                metrics = _epoch_history(line)
                if metrics:
                    if self.wandb_run is not None:
                        self.wandb_run.log(metrics[0])
                    with self.history_path.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(metrics[0]) + "\n")
        return len(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.capture.flush()


def _build_grouped_folds(
    train_dir: Path,
    delimiter: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    stems = sorted(
        path.name.removesuffix(".zarr")
        for path in train_dir.glob("*.zarr")
        if (train_dir / f"{path.name.removesuffix('.zarr')}.geff").exists()
    )
    if not stems:
        raise FileNotFoundError(f"no paired Zarr/GEFF datasets found in {train_dir}")

    grouped: dict[str, list[str]] = {}
    for stem in stems:
        group = stem.split(delimiter, 1)[0]
        grouped.setdefault(group, []).append(stem)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError(f"embryo-grouped split needs at least two groups, got {groups}")

    folds: list[dict[str, object]] = []
    for fold, validation_group in enumerate(groups):
        validation = grouped[validation_group]
        training = [
            stem
            for group in groups
            if group != validation_group
            for stem in grouped[group]
        ]
        folds.append(
            {
                "split": fold,
                "group_key": f"prefix-before-{delimiter}",
                "train_groups": [group for group in groups if group != validation_group],
                "test_groups": [validation_group],
                "train": training,
                "test": validation,
            }
        )
    return folds, {group: len(stems) for group, stems in sorted(grouped.items())}


def _ensure_checkpoint_patch(host_repo: Path) -> str:
    patch_path = COMPETITION_ROOT / "patches/organizer-periodic-checkpoints.patch"
    trainer_path = host_repo / "scripts/train_unet_transformer.py"
    trainer_source = trainer_path.read_text(encoding="utf-8")
    if "checkpoint_every: int | None = None" not in trainer_source:
        subprocess.run(
            ["git", "-C", str(host_repo), "apply", str(patch_path)],
            check=True,
        )
    return _sha256(patch_path)


def _read_history(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)
    source = config["source"]
    data = config["data"]
    train_cfg = config["train"]
    runtime_cfg = config.get("runtime", {})
    checkpoint_cfg = config.get("checkpoint", {})
    output = config["output"]

    if data["split_strategy"] != "embryo-prefix":
        raise ValueError(f"unsupported split strategy: {data['split_strategy']}")

    host_repo = COMPETITION_ROOT / source["repository_path"]
    revision = _host_revision(host_repo)
    if revision != source["revision"]:
        raise ValueError(
            f"organizer revision mismatch: expected {source['revision']}, got {revision}"
        )
    patch_sha256 = _ensure_checkpoint_patch(host_repo)

    sys.path[:0] = [str(host_repo / "src"), str(host_repo / "scripts")]
    import numpy as np
    import torch
    from train_unet_transformer import DEFAULT_AUGMENTATIONS, train

    if not torch.cuda.is_available():
        raise RuntimeError("the pinned organizer training loop requires a CUDA device")

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cuda_deterministic = bool(runtime_cfg.get("cuda_deterministic", False))
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.use_deterministic_algorithms(cuda_deterministic, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    mixed_precision = runtime_cfg.get("mixed_precision")
    if mixed_precision == "bfloat16":
        if runtime_cfg.get("sdp_backend") == "math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif mixed_precision in (None, "none"):
        autocast_context = contextlib.nullcontext()
    else:
        raise ValueError(f"unsupported mixed precision mode: {mixed_precision}")

    train_dir = COMPETITION_ROOT / data["train_dir"]
    artifact_dir = COMPETITION_ROOT / output["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds, group_counts = _build_grouped_folds(train_dir, data["group_delimiter"])
    fold = int(data["fold"])
    if fold >= len(folds):
        raise ValueError(f"fold {fold} is outside available folds 0..{len(folds) - 1}")
    selected = folds[fold]
    expected_group = data.get("validation_group")
    actual_group = selected["test_groups"][0]
    if expected_group and actual_group != expected_group:
        raise ValueError(
            "fold contract mismatch: expected validation group "
            f"{expected_group}, got {actual_group}"
        )

    splits_path = artifact_dir / "dataset_splits.json"
    splits_path.write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    split_summary = {
        "strategy": data["split_strategy"],
        "delimiter": data["group_delimiter"],
        "fold": fold,
        "group_counts": group_counts,
        "train_groups": selected["train_groups"],
        "validation_groups": selected["test_groups"],
        "train_datasets": len(selected["train"]),
        "validation_datasets": len(selected["test"]),
    }
    (artifact_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_summary, indent=2), flush=True)

    model_output_dir = (
        host_repo / "weights" / train_cfg["method"] / f"split_{fold}"
    )
    periodic_checkpoints = sorted(model_output_dir.glob("checkpoint_epoch_*.pth"))
    resume_checkpoint = (
        periodic_checkpoints[-1]
        if checkpoint_cfg.get("auto_resume", False) and periodic_checkpoints
        else None
    )
    resume_epoch = 0
    if resume_checkpoint is not None:
        resume_payload = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        resume_epoch = int(resume_payload["completed_epochs"])
        print(f"Auto-resume checkpoint: {resume_checkpoint}", flush=True)

    history_path = artifact_dir / "epoch_history.jsonl"
    retained_history = [
        item for item in _read_history(history_path) if int(item["epoch"]) < resume_epoch
    ]
    history_path.write_text(
        "".join(json.dumps(item) + "\n" for item in retained_history),
        encoding="utf-8",
    )
    wandb_run = _init_wandb(config, artifact_dir)
    if wandb_run is not None and not getattr(wandb_run, "resumed", False):
        for item in retained_history:
            wandb_run.log(item)
    started = time.monotonic()
    captured_stdout = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(
                _LiveEpochTee(captured_stdout, wandb_run, history_path)
            ),
            autocast_context,
        ):
            train(
                data_dir=train_dir,
                fold=fold,
                splits_file=splits_path,
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
                max_iters=None,
                debug_video=None,
                seed=seed,
                max_frames=None,
                window_size=int(data["window_size"]),
                augmentations=DEFAULT_AUGMENTATIONS,
                pool_kernel_um=float(train_cfg["pool_kernel_um"]),
                data_parallel=False,
                checkpoint_every=int(checkpoint_cfg["every_epochs"]),
                resume_checkpoint=resume_checkpoint,
                gradient_checkpointing=bool(
                    runtime_cfg.get("gradient_checkpointing", True)
                ),
            )
    except BaseException:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise

    train_seconds = time.monotonic() - started
    history = _read_history(history_path)
    if len(history) != int(train_cfg["epochs"]):
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise RuntimeError(
            f"expected {train_cfg['epochs']} parsed epoch records, got {len(history)}"
        )

    weights_path = model_output_dir / "edge_predictor_best.pth"
    config_path_out = weights_path.with_name("config.json")
    periodic_checkpoints = sorted(model_output_dir.glob("checkpoint_epoch_*.pth"))
    result = {
        "experiment_id": config["experiment_id"],
        "source_revision": revision,
        "source_patch_sha256": patch_sha256,
        "device": torch.cuda.get_device_name(0),
        "cuda_devices": torch.cuda.device_count(),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "train_seconds": round(train_seconds, 3),
        "split": split_summary,
        "parameters": train_cfg,
        "runtime": runtime_cfg,
        "epoch_history": history,
        "best_epoch": max(history, key=lambda item: item["validation/best_acc_recall"])["epoch"],
        "weights_path": str(weights_path),
        "weights_sha256": _sha256(weights_path),
        "model_config_path": str(config_path_out),
        "model_config_sha256": _sha256(config_path_out),
        "periodic_checkpoints": [
            {
                "completed_epochs": int(path.stem.rsplit("_", 1)[1]),
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in periodic_checkpoints
        ],
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        "wandb_url": wandb_run.url if wandb_run is not None else None,
    }
    (artifact_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.summary["runtime/train_seconds"] = result["train_seconds"]
        wandb_run.summary["runtime/peak_cuda_memory_bytes"] = result["peak_cuda_memory_bytes"]
        wandb_run.summary["checkpoint/sha256"] = result["weights_sha256"]
        wandb_run.summary["source/revision"] = revision
        wandb_run.summary["split/train_datasets"] = split_summary["train_datasets"]
        wandb_run.summary["split/validation_datasets"] = split_summary["validation_datasets"]
        wandb_run.finish()
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
