#!/usr/bin/env python
"""Run shared inference and ILP for a trained custom U-Net/nnU-Net A/B model."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_ROOT = SCRIPT_DIR.parent
SOURCE_ROOT = COMPETITION_ROOT / "src"
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0005b-backbone-ab-nnunet.toml"

sys.path.insert(0, str(SOURCE_ROOT))
from backbone_ab.config import load_and_validate_config  # noqa: E402


def _create_prediction_run_dir(
    artifact_dir: Path,
    *,
    allowed_root: Path = COMPETITION_ROOT,
) -> Path:
    """Create an isolated prediction directory without following an escape symlink."""
    artifact_dir = Path(artifact_dir)
    allowed_resolved = Path(allowed_root).resolve(strict=True)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact directory is missing: {artifact_dir}")
    if artifact_dir.is_symlink():
        raise ValueError(f"artifact directory must not be a symlink: {artifact_dir}")
    artifact_resolved = artifact_dir.resolve(strict=True)
    if not artifact_resolved.is_relative_to(allowed_resolved):
        raise ValueError(f"artifact directory escapes competition root: {artifact_dir}")

    runs_root = artifact_dir / "predicted-runs"
    if runs_root.is_symlink():
        raise ValueError(f"prediction run root must not be a symlink: {runs_root}")
    runs_root.mkdir(exist_ok=True)
    runs_resolved = runs_root.resolve(strict=True)
    if runs_resolved.parent != artifact_resolved:
        raise ValueError(f"prediction run root escapes artifact directory: {runs_root}")

    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=runs_resolved))
    run_resolved = run_dir.resolve(strict=True)
    if run_dir.is_symlink() or run_resolved.parent != runs_resolved:
        raise RuntimeError(f"prediction run directory escaped its root: {run_dir}")
    return run_dir


def _validate_prediction_outputs(
    prediction_dir: Path,
    expected_datasets: list[str],
) -> list[Path]:
    """Require one local, non-symlink GEFF for every dataset in this run."""
    prediction_dir = Path(prediction_dir)
    if not prediction_dir.is_dir() or prediction_dir.is_symlink():
        raise ValueError(f"unsafe prediction run directory: {prediction_dir}")
    resolved_dir = prediction_dir.resolve(strict=True)
    expected = sorted(expected_datasets)
    if len(expected) != len(set(expected)):
        raise ValueError("expected dataset names contain duplicates")

    geffs: list[Path] = []
    for path in prediction_dir.iterdir():
        if path.is_symlink():
            raise ValueError(f"prediction run contains a symlink: {path}")
        if path.suffix != ".geff":
            continue
        resolved = path.resolve(strict=True)
        if resolved.parent != resolved_dir:
            raise ValueError(f"prediction GEFF escapes run directory: {path}")
        geffs.append(path)
    actual = sorted(path.stem for path in geffs)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"prediction run dataset mismatch: missing={missing}, extra={extra}"
        )
    return sorted(geffs)


def _convert_current_prediction_run(
    geffs_to_csv,
    prediction_dir: Path,
    submission_path: Path,
    expected_datasets: list[str],
) -> None:
    """Convert only the GEFFs validated as outputs of this isolated run."""
    _validate_prediction_outputs(prediction_dir, expected_datasets)
    geffs_to_csv(prediction_dir, submission_path)


@contextlib.contextmanager
def _suppress_solver_output():
    with (
        Path(os.devnull).open("w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        yield


def run(
    config_path: Path,
    checkpoint_override: Path | None = None,
    inference_profile_path: Path | None = None,
) -> dict:
    config = load_and_validate_config(config_path)
    organizer = COMPETITION_ROOT / config["source"]["organizer_repository_path"]
    sys.path[:0] = [str(organizer / "src"), str(organizer / "scripts")]

    import torch
    import tracksdata as td
    import train_unet_transformer as host_training
    from backbone_ab.backbones import build_joint_model
    from backbone_ab.checkpointing import (
        InferenceProfile,
        load_checkpoint,
        load_inference_profile,
        sha256_file,
    )
    from backbone_ab.inference import predict_dataset_corrected
    from geffs_to_csv import geffs_to_csv
    from predict_unet_transformer import PredictConfig, build_graph, predict_video
    from submission import validate_submission
    from tracking_cellmot.io import save_graph

    if not torch.cuda.is_available():
        raise RuntimeError("backbone A/B inference requires CUDA")

    artifact_dir = COMPETITION_ROOT / config["output"]["artifact_dir"]
    checkpoint = checkpoint_override or artifact_dir / "best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    prediction_dir = _create_prediction_run_dir(artifact_dir)

    device = torch.device("cuda")
    model = build_joint_model(config["backbone"], host_training).to(device)
    loaded_checkpoint = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(loaded_checkpoint.state_dict)
    model.eval()

    inference = config["inference"]
    model_api = str(inference.get("model_api", "legacy"))
    profile = None
    if model_api == "corrected_v2":
        profile = (
            load_inference_profile(inference_profile_path)
            if inference_profile_path is not None
            else InferenceProfile.from_experiment_config(
                config,
                checkpoint_sha256=loaded_checkpoint.sha256,
                experiment_config_sha256=sha256_file(config_path),
            )
        )
        if profile.checkpoint_sha256 != loaded_checkpoint.sha256:
            raise ValueError("inference profile checkpoint hash does not match loaded weights")
        if profile.experiment_config_sha256 != sha256_file(config_path):
            raise ValueError("inference profile experiment config hash does not match")
        if profile.postprocess_profile != "none":
            raise ValueError(
                "corrected_v2 local inference currently requires postprocess_profile=none"
            )
    predict_config = PredictConfig(
        det_threshold=float(inference["det_threshold"]),
        det_tta=bool(inference["det_tta"]),
        pool_kernel_um=float(inference["pool_kernel_um"]),
        edge_activation=inference["edge_activation"],
        threshold=float(inference["edge_threshold"]),
        use_ilp=False,
    )
    test_dir = COMPETITION_ROOT / config["data"]["test_dir"]
    dataset_paths = sorted(test_dir.glob("*.zarr"))
    expected_datasets = [path.stem for path in dataset_paths]
    downsample = tuple(int(value) for value in config["train"]["downsample"])
    results = []
    for dataset_path in dataset_paths:
        if profile is not None:
            coords, edges = predict_dataset_corrected(
                model,
                dataset_path,
                device,
                profile,
            )
        else:
            coords, edges = predict_video(
                model,
                dataset_path,
                device,
                cfg=predict_config,
                window_size=int(config["data"]["window_size"]),
                downsample=downsample,
            )
        graph = build_graph(coords, edges)
        if profile is None and bool(inference["use_ilp"]) and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=float(inference["ilp_edge_weight"])
                * td.EdgeAttr("edge_prob"),
                appearance_weight=float(inference["ilp_appearance_weight"]),
                disappearance_weight=float(inference["ilp_disappearance_weight"]),
                division_weight=float(inference["ilp_division_weight"]),
            )
            with _suppress_solver_output():
                graph = solver.solve(graph)
        output_path = prediction_dir / f"{dataset_path.stem}.geff"
        save_graph(graph, output_path, overwrite=True)
        results.append(
            {
                "dataset": dataset_path.stem,
                "nodes": graph.num_nodes(),
                "edges": graph.num_edges(),
            }
        )

    submission_path = COMPETITION_ROOT / config["output"]["submission_file"]
    _convert_current_prediction_run(
        geffs_to_csv,
        prediction_dir,
        submission_path,
        expected_datasets,
    )
    validation = validate_submission(
        submission_path,
        expected_datasets=expected_datasets,
    )
    result = {
        "experiment_id": config["experiment_id"],
        "backbone": config["backbone"]["name"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": loaded_checkpoint.sha256,
        "checkpoint_format": loaded_checkpoint.source_format,
        "model_api": model_api,
        "inference_profile_sha256": profile.sha256 if profile is not None else None,
        "datasets": results,
        "prediction_run_dir": str(prediction_dir),
        "submission": validation,
        "submission_path": str(submission_path),
    }
    (artifact_dir / "inference_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inference-profile", type=Path)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.checkpoint.resolve() if args.checkpoint else None,
        args.inference_profile.resolve() if args.inference_profile else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
