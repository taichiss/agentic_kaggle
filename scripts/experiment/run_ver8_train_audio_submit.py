# ruff: noqa: E402
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.extract_train_audio_embeddings import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    extract_train_audio_embeddings,
)
from scripts.experiment.prepare_kaggle_ver8_kernel import (
    DEFAULT_OUTPUT_DIR as DEFAULT_KERNEL_OUTPUT_DIR,
)
from scripts.experiment.prepare_kaggle_ver8_kernel import (
    DEFAULT_SLUG,
    DEFAULT_TITLE,
    load_kaggle_username,
    prepare_bundle,
)
from scripts.experiment.train_audio_mlp_head import (
    DEFAULT_CACHE_PATH,
    DEFAULT_SOUNDSCAPE_CACHE_PATH,
    train_train_audio_head,
)
from scripts.experiment.train_audio_mlp_head import (
    DEFAULT_OUTPUT_DIR as DEFAULT_MODELS_OUTPUT_DIR,
)


def run_command(command: list[str], cwd: Path = REPO_ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def poll_kernel_status(owner: str, slug: str, interval_seconds: int, max_polls: int) -> None:
    target = f"{owner}/{slug}"
    for poll_index in range(max_polls):
        run_command(["kaggle", "kernels", "status", target])
        if poll_index + 1 < max_polls:
            time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and submit ver8 train_audio MLP head flow")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--soundscape-cache-path",
        type=Path,
        default=DEFAULT_SOUNDSCAPE_CACHE_PATH,
    )
    parser.add_argument("--models-output-dir", type=Path, default=DEFAULT_MODELS_OUTPUT_DIR)
    parser.add_argument("--kernel-output-dir", type=Path, default=DEFAULT_KERNEL_OUTPUT_DIR)
    parser.add_argument("--kaggle-config", type=Path, default=REPO_ROOT / "kaggle.json")
    parser.add_argument("--owner", type=str, default=None)
    parser.add_argument("--slug", type=str, default=DEFAULT_SLUG)
    parser.add_argument("--title", type=str, default=DEFAULT_TITLE)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-prepare-kernel", action="store_true")
    parser.add_argument("--push-dataset", action="store_true")
    parser.add_argument("--push-kernel", action="store_true")
    parser.add_argument("--check-status", action="store_true")
    parser.add_argument("--status-polls", type=int, default=1)
    parser.add_argument("--status-interval-seconds", type=int, default=60)
    parser.add_argument(
        "--dataset-version-message", type=str, default="Add ver8 train_audio MLP head bundle"
    )
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-chunks-per-file", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage1-val-fraction", type=float, default=0.1)
    parser.add_argument("--stage2-val-fraction", type=float, default=0.0)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--secondary-weight", type=float, default=0.7)
    parser.add_argument("--soundscape-dataset", type=str, default="all66")
    parser.add_argument("--force-rebuild-soundscape-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    owner = args.owner or load_kaggle_username(args.kaggle_config)

    if not args.skip_extract:
        extract_train_audio_embeddings(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            cache_path=args.cache_path,
            force_rebuild=args.force_rebuild_cache,
            batch_size=args.extract_batch_size,
            max_files=args.max_files,
            max_chunks_per_file=args.max_chunks_per_file,
        )

    if not args.skip_train:
        cache = dict(np.load(args.cache_path, allow_pickle=False))
        train_train_audio_head(
            cache=cache,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            output_dir=args.models_output_dir,
            train_audio_cache_path=args.cache_path,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            batch_size=args.train_batch_size,
            stage1_learning_rate=args.stage1_learning_rate,
            stage2_learning_rate=args.stage2_learning_rate,
            stage1_val_fraction=args.stage1_val_fraction,
            stage2_val_fraction=args.stage2_val_fraction,
            primary_weight=args.primary_weight,
            secondary_weight=args.secondary_weight,
            soundscape_dataset=args.soundscape_dataset,
            soundscape_cache_path=args.soundscape_cache_path,
            force_rebuild_soundscape_cache=args.force_rebuild_soundscape_cache,
            seed=args.seed,
        )

    if not args.skip_prepare_kernel:
        metadata_path = prepare_bundle(
            notebook_path=REPO_ROOT / "notebook.ipynb",
            output_dir=args.kernel_output_dir,
            owner=owner,
            slug=args.slug,
            title=args.title,
        )
        print(f"Kernel bundle prepared: {metadata_path}")

    if args.push_dataset:
        run_command(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(args.models_output_dir),
                "-m",
                args.dataset_version_message,
            ]
        )

    if args.push_kernel:
        run_command(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(args.kernel_output_dir),
            ]
        )

    if args.check_status:
        poll_kernel_status(
            owner=owner,
            slug=args.slug,
            interval_seconds=args.status_interval_seconds,
            max_polls=args.status_polls,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
