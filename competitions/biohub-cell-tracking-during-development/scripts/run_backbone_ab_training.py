#!/usr/bin/env python
"""Train the controlled custom U-Net/nnU-Net backbone A/B joint model."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_ROOT = SCRIPT_DIR.parent
SOURCE_ROOT = COMPETITION_ROOT / "src"
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0005b-backbone-ab-nnunet.toml"

sys.path.insert(0, str(SOURCE_ROOT))
from backbone_ab.checkpointing import (  # noqa: E402
    canonical_json_sha256,
    load_checkpoint,
    sha256_file,
)
from backbone_ab.config import load_and_validate_config  # noqa: E402

_PERIODIC_CHECKPOINT_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pth")


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an atomic replacement."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        # Some filesystems/platforms do not support directory fsync.
        with contextlib.suppress(OSError):
            os.fsync(descriptor)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _atomic_torch_save(torch_module, payload: object, destination: Path) -> None:
    """Durably stage a torch payload beside its destination, then replace it."""
    destination = Path(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint parent directory is missing: {destination.parent}"
        )
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    open_descriptor: int | None = descriptor
    try:
        stream = os.fdopen(descriptor, "wb")
        open_descriptor = None
        with stream:
            torch_module.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if open_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(open_descriptor)
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise


def _resume_candidates(artifact_dir: Path) -> list[Path]:
    """Return last first, then periodic checkpoints newest to oldest."""
    last = artifact_dir / "last_checkpoint.pth"
    periodic: list[tuple[int, Path]] = []
    for path in artifact_dir.glob("checkpoint_epoch_*.pth"):
        match = _PERIODIC_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file() and not path.is_symlink():
            periodic.append((int(match.group(1)), path))
    candidates = [last] if last.is_file() and not last.is_symlink() else []
    candidates.extend(path for _, path in sorted(periodic, reverse=True))
    return candidates


def _select_resume_checkpoint(
    artifact_dir: Path,
    *,
    experiment_id: str,
    contract: str,
    resume_fingerprint: str,
    validation_manifest_hash: str | None,
    target_epochs: int,
    map_location: str | object = "cpu",
    checkpoint_loader=load_checkpoint,
    restore_checkpoint=None,
):
    """Select the newest loadable, contract-compatible resume checkpoint.

    A corrupt or incompatible ``last_checkpoint.pth`` is retained for diagnosis;
    the selector falls back to a matching periodic checkpoint without deleting or
    rewriting any candidate.  A compatible checkpoint beyond ``target_epochs`` is
    a configuration contraction, however, and is rejected instead of silently
    rolling the artifact directory back to an older periodic checkpoint.
    """
    candidates = _resume_candidates(artifact_dir)
    failures: list[str] = []
    best_valid_path: Path | None = None
    best_completed_epochs = -1

    def load_validate_restore(path: Path):
        loaded = checkpoint_loader(path, map_location=map_location)
        metadata = loaded.metadata
        if metadata.get("experiment_id") != experiment_id:
            raise ValueError("experiment_id mismatch")
        saved_resume_fingerprint = metadata.get("resume_fingerprint")
        if saved_resume_fingerprint is None and contract == "corrected_v2":
            raise ValueError("missing resume_fingerprint")
        if (
            saved_resume_fingerprint is not None
            and saved_resume_fingerprint != resume_fingerprint
        ):
            raise ValueError("resume fingerprint mismatch")
        if metadata.get("validation_subset_manifest_sha256") != validation_manifest_hash:
            raise ValueError("validation subset manifest hash mismatch")
        if "optimizer_state_dict" not in metadata:
            raise ValueError("missing optimizer_state_dict")
        if contract == "corrected_v2" and metadata.get("rng_state") is None:
            raise ValueError("missing rng_state")
        float(metadata["best_score"])
        completed_epochs = int(metadata.get("completed_epochs", 0))
        if completed_epochs <= 0:
            raise ValueError("completed_epochs must be positive")
        periodic_match = _PERIODIC_CHECKPOINT_PATTERN.fullmatch(path.name)
        if (
            periodic_match is not None
            and completed_epochs != int(periodic_match.group(1))
        ):
            raise ValueError("filename/completed_epochs mismatch")
        if restore_checkpoint is not None:
            restore_checkpoint(loaded)
        return loaded, completed_epochs, periodic_match

    for path in candidates:
        try:
            _, completed_epochs, periodic_match = load_validate_restore(path)
        except Exception as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        if completed_epochs > target_epochs:
            raise RuntimeError(
                "refusing to contract an existing training run: "
                f"compatible checkpoint {path.name} completed {completed_epochs} "
                f"epochs, which exceeds target {target_epochs}"
            )
        if completed_epochs > best_completed_epochs:
            best_valid_path = path
            best_completed_epochs = completed_epochs
        # Periodic candidates are ordered newest first. Once one is valid, no
        # later periodic file can beat it; compare it with the retained last.
        if periodic_match is not None:
            break
        remaining_periodic_epochs = [
            int(match.group(1))
            for candidate in candidates[1:]
            if (match := _PERIODIC_CHECKPOINT_PATTERN.fullmatch(candidate.name))
            is not None
        ]
        if not remaining_periodic_epochs or completed_epochs >= max(
            remaining_periodic_epochs
        ):
            break
    if best_valid_path is not None:
        # A rejected newer candidate may have partially mutated model or optimizer
        # state before its restore callback failed.  Re-load and re-apply the final
        # selection so the caller never resumes from a mixed runtime state.
        try:
            selected, selected_epochs, _ = load_validate_restore(best_valid_path)
        except Exception as error:
            raise RuntimeError(
                "selected resume checkpoint could not be restored consistently: "
                f"{best_valid_path.name}: {type(error).__name__}: {error}"
            ) from error
        if selected_epochs > target_epochs:
            raise RuntimeError(
                "refusing to contract an existing training run: "
                f"compatible checkpoint {best_valid_path.name} completed "
                f"{selected_epochs} epochs, which exceeds target {target_epochs}"
            )
        if selected_epochs != best_completed_epochs:
            raise RuntimeError(
                "selected resume checkpoint changed during validation: "
                f"expected {best_completed_epochs} completed epochs, got "
                f"{selected_epochs}"
            )
        for failure in failures:
            print(f"resume candidate rejected: {failure}", file=sys.stderr)
        return best_valid_path, selected
    if failures:
        details = "; ".join(failures)
        raise RuntimeError(f"no valid resume checkpoint found: {details}")
    return None


def _select_resume_then_persist_provenance(
    artifact_dir: Path,
    config_path: Path,
    folds: list[dict],
    *,
    resume_selector,
):
    """Validate resume state before creating or overwriting run provenance."""
    selected_resume = resume_selector()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    splits_path = artifact_dir / "dataset_splits.json"
    splits_path.write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path, artifact_dir / "experiment.toml")
    return selected_resume


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def validate_setup(config_path: Path) -> dict:
    """Validate paths and the controlled experiment contract without importing CUDA."""
    config = load_and_validate_config(config_path)
    organizer = COMPETITION_ROOT / config["source"]["organizer_repository_path"]
    if not organizer.is_dir():
        raise FileNotFoundError(f"organizer source is missing: {organizer}")
    actual_revision = _git_revision(organizer)
    expected_revision = config["source"]["organizer_revision"]
    if actual_revision != expected_revision:
        raise ValueError(
            f"organizer revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    train_dir = COMPETITION_ROOT / config["data"]["train_dir"]
    if not train_dir.is_dir():
        raise FileNotFoundError(f"training data is missing: {train_dir}")
    validation_manifest = config["data"].get("validation_subset_manifest")
    validation_manifest_path = (
        COMPETITION_ROOT / validation_manifest if validation_manifest else None
    )
    if validation_manifest_path is not None and not validation_manifest_path.is_file():
        raise FileNotFoundError(
            f"validation subset manifest is missing: {validation_manifest_path}"
        )
    return {
        "experiment_id": config["experiment_id"],
        "backbone": config["backbone"]["name"],
        "model_contract": config["backbone"].get("contract", "legacy"),
        "feature_dim": config["backbone"]["feature_dim"],
        "organizer_revision": actual_revision,
        "nnunet_revision": config["source"]["nnunet_revision"],
        "train_dir": str(train_dir),
        "validation_subset_manifest_sha256": (
            sha256_file(validation_manifest_path)
            if validation_manifest_path is not None
            else None
        ),
    }


def _resume_fingerprint(config: dict) -> str:
    """Hash training semantics while allowing an epochs-only continuation."""
    resume_config = json.loads(json.dumps(config))
    resume_config["train"].pop("epochs", None)
    return canonical_json_sha256(resume_config)


def _capture_rng_state(torch, np, loader_generator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, torch, np, loader_generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    loader_generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_validation_subset(
    config: dict,
    selected_fold: dict,
    *,
    competition_root: Path = COMPETITION_ROOT,
) -> tuple[list[str], dict | None]:
    """Resolve and validate a fixed calibration/report dataset manifest."""
    relative_path = config["data"].get("validation_subset_manifest")
    if relative_path is None:
        return list(selected_fold["test"]), None
    subset = str(config["data"]["validation_subset"])
    path = competition_root / relative_path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported validation subset manifest schema")
    selected = manifest.get(subset)
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"validation subset {subset!r} must be a non-empty list")
    if not all(isinstance(name, str) and name for name in selected):
        raise ValueError("validation subset dataset names must be non-empty strings")
    if len(selected) != len(set(selected)):
        raise ValueError(f"validation subset {subset!r} contains duplicates")
    fold_names = set(selected_fold["test"])
    unknown = sorted(set(selected) - fold_names)
    if unknown:
        raise ValueError(
            f"validation subset contains datasets outside the selected fold: {unknown}"
        )
    return list(selected), {
        "manifest": str(relative_path),
        "manifest_sha256": sha256_file(path),
        "subset": subset,
        "datasets": list(selected),
    }


def _load_runtime(config: dict):
    organizer = COMPETITION_ROOT / config["source"]["organizer_repository_path"]
    sys.path[:0] = [str(organizer / "src"), str(organizer / "scripts"), str(SCRIPT_DIR)]
    import torch
    import train_unet_transformer as host_training
    from backbone_ab.backbones import build_joint_model, count_trainable_parameters
    from backbone_ab.dataset import DeterministicAugmentationDataset
    from backbone_ab.training import evaluate_predicted_nodes, train_epoch
    from run_host_baseline_training import _build_grouped_folds

    return (
        torch,
        host_training,
        _build_grouped_folds,
        build_joint_model,
        count_trainable_parameters,
        DeterministicAugmentationDataset,
        train_epoch,
        evaluate_predicted_nodes,
    )


def run(config_path: Path) -> dict:
    setup = validate_setup(config_path)
    config = load_and_validate_config(config_path)
    contract = config["backbone"].get("contract", "legacy")
    (
        torch,
        host_training,
        build_grouped_folds,
        build_joint_model,
        count_parameters,
        deterministic_augmentation_dataset,
        train_one_epoch,
        evaluate_model,
    ) = _load_runtime(config)

    if not torch.cuda.is_available():
        raise RuntimeError("backbone A/B training requires CUDA")

    import numpy as np
    from torch.utils.data import DataLoader

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runtime = config["runtime"]
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(runtime["cuda_deterministic"])
    torch.use_deterministic_algorithms(bool(runtime["cuda_deterministic"]), warn_only=True)

    train_dir = COMPETITION_ROOT / config["data"]["train_dir"]
    folds, group_counts = build_grouped_folds(train_dir, config["data"]["group_delimiter"])
    fold_index = int(config["data"]["fold"])
    selected_fold = folds[fold_index]
    expected_group = config["data"].get("validation_group")
    if expected_group and selected_fold["test_groups"] != [expected_group]:
        raise ValueError(
            f"validation group mismatch: expected {expected_group}, "
            f"got {selected_fold['test_groups']}"
        )

    artifact_dir = COMPETITION_ROOT / config["output"]["artifact_dir"]

    downsample = tuple(int(value) for value in config["train"]["downsample"])
    window_size = int(config["data"]["window_size"])

    def load_videos(names: list[str]):
        loaded = []
        for name in names:
            loaded.append(
                host_training.load_dataset_windows(
                    train_dir / name,
                    window_size=window_size,
                    downsample=downsample,
                )
            )
        return loaded

    training_videos = load_videos(selected_fold["train"])
    validation_names, validation_subset_info = _resolve_validation_subset(
        config, selected_fold
    )
    validation_videos = load_videos(validation_names)
    all_windows = [
        window
        for _, windows in training_videos + validation_videos
        for window in windows
    ]
    max_nodes = max(max(window.node_counts) for window in all_windows)
    if contract == "legacy":
        train_dataset = host_training.FrameWindowDataset(
            training_videos,
            max_nodes=max_nodes,
            augmentations=host_training.DEFAULT_AUGMENTATIONS,
        )
    else:
        unaugmented_dataset = host_training.FrameWindowDataset(
            training_videos,
            max_nodes=max_nodes,
        )
        train_dataset = deterministic_augmentation_dataset(
            unaugmented_dataset,
            host_training.DEFAULT_AUGMENTATIONS,
            seed=seed,
        )
    validation_dataset = host_training.FrameWindowDataset(
        validation_videos,
        max_nodes=max_nodes,
    )
    generator = torch.Generator().manual_seed(seed)
    workers = int(config["train"]["num_workers"])
    loader_kwargs = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": workers,
        "persistent_workers": workers > 0 and contract == "legacy",
        "prefetch_factor": 2 if workers > 0 else None,
        "pin_memory": False,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda")
    model = build_joint_model(config["backbone"], host_training).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
    )
    parameter_count = count_parameters(model)
    print(f"backbone={config['backbone']['name']} parameters={parameter_count:,}", flush=True)

    checkpoint_path = artifact_dir / "best_model.pth"
    periodic_path = artifact_dir / "last_checkpoint.pth"
    checkpoint_every = int(config["checkpoint"]["every_epochs"])
    if checkpoint_every <= 0:
        raise ValueError("checkpoint.every_epochs must be positive")
    start_epoch = 0
    best_score = 0.0
    config_fingerprint = canonical_json_sha256(config)
    resume_fingerprint = _resume_fingerprint(config)
    resume_rng_state = None
    resume_history = None
    def select_resume():
        if not bool(config["checkpoint"]["auto_resume"]):
            return None
        current_manifest_hash = (
            validation_subset_info["manifest_sha256"]
            if validation_subset_info is not None
            else None
        )

        def restore_checkpoint(loaded_checkpoint) -> None:
            model.load_state_dict(loaded_checkpoint.state_dict)
            optimizer.load_state_dict(
                loaded_checkpoint.metadata["optimizer_state_dict"]
            )

        return _select_resume_checkpoint(
            artifact_dir,
            experiment_id=str(config["experiment_id"]),
            contract=contract,
            resume_fingerprint=resume_fingerprint,
            validation_manifest_hash=current_manifest_hash,
            target_epochs=int(config["train"]["epochs"]),
            map_location="cpu",
            restore_checkpoint=restore_checkpoint,
        )

    selected_resume = _select_resume_then_persist_provenance(
        artifact_dir,
        config_path,
        folds,
        resume_selector=select_resume,
    )
    if selected_resume is not None:
        resume_path, loaded = selected_resume
        metadata = loaded.metadata
        start_epoch = int(metadata["completed_epochs"])
        best_score = float(metadata["best_score"])
        resume_rng_state = metadata.get("rng_state")
        resume_history = metadata.get("history")
        if contract == "corrected_v2" and resume_rng_state is None:
            raise ValueError("corrected_v2 checkpoint is missing rng_state")
        print(
            f"resuming after {start_epoch} completed epochs from {resume_path.name}",
            flush=True,
        )

    history_path = artifact_dir / "epoch_history.jsonl"
    if start_epoch == 0:
        history_path.write_text("", encoding="utf-8")
    history = _load_history(history_path)
    if start_epoch and len(history) < start_epoch:
        if not isinstance(resume_history, list) or len(resume_history) != start_epoch:
            raise ValueError(
                f"history/checkpoint mismatch: {len(history)} records for epoch {start_epoch}"
            )
        history = resume_history
        history_path.write_text(
            "".join(json.dumps(record) + "\n" for record in history),
            encoding="utf-8",
        )
    if start_epoch and len(history) > start_epoch:
        history = history[:start_epoch]
        history_path.write_text(
            "".join(json.dumps(record) + "\n" for record in history),
            encoding="utf-8",
        )
    if resume_rng_state is not None:
        _restore_rng_state(resume_rng_state, torch, np, generator)
    else:
        # Backbone construction consumes a different number of random values
        # in the identity and temporal arms. Reset runtime RNGs so dropout and
        # other stochastic training operations remain paired across A/B.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    started = time.monotonic()
    epochs = int(config["train"]["epochs"])
    validation_every = int(config["train"].get("validation_every_epochs", 1))
    for epoch in range(start_epoch, epochs):
        if contract == "corrected_v2":
            train_dataset.set_epoch(epoch)
        edge_loss, detection_loss, division_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            host_training,
            config["sparse_heatmap"],
            float(config["train"]["det_loss_weight"]),
            float(config["train"]["pool_kernel_um"]),
            runtime["mixed_precision"],
            float(config["train"].get("division_loss_weight", 0.0)),
            float(config["train"].get("division_negative_weight", 0.1)),
            contract=contract,
            node_proposal_strategy=config["train"].get(
                "node_proposal_strategy", "ground_truth"
            ),
            proposal_curriculum=config["train"].get("proposal_curriculum", {}),
            epoch_index=epoch,
            proposal_seed=seed,
        )
        should_validate = (epoch + 1) % validation_every == 0 or epoch + 1 == epochs
        if should_validate:
            validation_metrics: dict[str, float | None] = {}
            validation_loss, validation_accuracy, node_recall = evaluate_model(
                model,
                validation_loader,
                device,
                host_training,
                float(config["train"]["pool_kernel_um"]),
                float(config["train"]["validation_det_threshold"]),
                contract=contract,
                max_proposals_per_frame=int(
                    config["train"].get("proposal_curriculum", {}).get(
                        "max_proposals_per_frame", 96
                    )
                ),
                metrics_out=validation_metrics,
            )
            candidate_recall = validation_metrics.get("candidate_recall")
            score = validation_accuracy * node_recall
            best_improved = score >= best_score
            if best_improved:
                best_score = score
        else:
            validation_loss = None
            validation_accuracy = None
            node_recall = None
            candidate_recall = None
            best_improved = False
        proposal_ratio = (
            0.0
            if config["train"].get("node_proposal_strategy", "ground_truth")
            == "ground_truth"
            else float(
                config["train"].get("proposal_curriculum", {})[
                    "predicted_ratios"
                ][
                    min(
                        epoch,
                        len(
                            config["train"].get("proposal_curriculum", {})[
                                "predicted_ratios"
                            ]
                        )
                        - 1,
                    )
                ]
            )
        )
        record = {
            "epoch": epoch,
            "train/edge_loss": edge_loss,
            "train/detection_loss": detection_loss,
            "train/division_loss": division_loss,
            "train/predicted_node_ratio": proposal_ratio,
            "validation/loss": validation_loss,
            "validation/accuracy": validation_accuracy,
            "validation/node_recall": node_recall,
            "validation/candidate_recall": candidate_recall,
            "validation/best_acc_recall": best_score,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        checkpoint_payload = {
            "checkpoint_schema_version": 2,
            "experiment_id": config["experiment_id"],
            "model_contract": contract,
            "completed_epochs": epoch + 1,
            "best_score": best_score,
            "config_fingerprint": config_fingerprint,
            "resume_fingerprint": resume_fingerprint,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "rng_state": _capture_rng_state(torch, np, generator),
            "validation_subset_manifest_sha256": (
                validation_subset_info["manifest_sha256"]
                if validation_subset_info is not None
                else None
            ),
        }
        if best_improved:
            if contract == "corrected_v2":
                _atomic_torch_save(torch, checkpoint_payload, checkpoint_path)
            else:
                # Preserve the raw-state-dict contract of EXP-0005/0006.
                _atomic_torch_save(torch, model.state_dict(), checkpoint_path)
        _atomic_torch_save(torch, checkpoint_payload, periodic_path)
        if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
            _atomic_torch_save(
                torch,
                checkpoint_payload,
                artifact_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth",
            )
        print(json.dumps(record), flush=True)

    result = {
        **setup,
        "device": torch.cuda.get_device_name(0),
        "parameters": parameter_count,
        "split": {
            "fold": fold_index,
            "group_counts": group_counts,
            "train_groups": selected_fold["train_groups"],
            "validation_groups": selected_fold["test_groups"],
            "validation_subset": validation_subset_info,
        },
        "epochs": epochs,
        "model_contract": contract,
        "best_score": best_score,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "best_checkpoint": str(checkpoint_path),
        "history": history,
    }
    (artifact_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate config, source revision, and data paths without importing CUDA",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.validate_only:
        print(json.dumps(validate_setup(config_path), indent=2))
    else:
        run(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
