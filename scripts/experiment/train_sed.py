# ruff: noqa: E402
"""Train and evaluate a TensorFlow 20-second SED model with optional Perch teacher warmstart."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import tensorflow as tf

from core.config import AudioConfig, DistillConfig, ModelConfig, TrainConfig
from core.metrics import macro_roc_auc
from data.cv_splits import (
    ClipFoldRecord,
    file_group_kfold,
    site_balanced_file_folds,
    site_holdout_folds,
)
from data.dataset import (
    SoundscapeClipRecord,
    build_soundscape_clip_index,
    build_train_audio_perch_clip_index,
)
from models.sed import build_sed_model
from scripts.experiment.extract_train_audio_embeddings import extract_train_audio_embeddings
from scripts.experiment.perch_probe_cv import build_perch_mapping
from training.trainer import (
    SoundscapeSequence,
    StageResult,
    TrainAudioSequence,
    clone_weights,
    predict_soundscape_windows,
    restore_weights,
    run_soundscape_stage,
    run_train_audio_stage,
)

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "input" / "BirdCLEF+ 2026"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models"
DEFAULT_TRAIN_AUDIO_TEACHER_CACHE = (
    REPO_ROOT / "data" / "models" / "train_audio_perch_chunks_v1.npz"
)
DEFAULT_SOUNDSCAPE_TEACHER_CACHE = REPO_ROOT / "data" / "models" / "perch_labeled_cache_v1.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "models" / "sed_teacher"
SITE_PATTERN = re.compile(r"_(S\d{2})_")


@dataclass(frozen=True)
class FoldSummary:
    fold_id: int
    windows: int
    files: int
    auc: float
    train_stage: StageResult


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    dataset_name: str
    splitter: str
    n_splits: int
    stage1_enabled: bool
    include_teacher_distill: bool
    stage1_result: StageResult | None
    oof_macro_auc: float
    folds: tuple[FoldSummary, ...]


def load_primary_labels(data_dir: Path) -> tuple[str, ...]:
    with (data_dir / "sample_submission.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return tuple(header[1:])


def site_from_filename(filename: str) -> str:
    match = SITE_PATTERN.search(filename)
    return match.group(1) if match is not None else "unknown"


def build_fold_records(records: list[SoundscapeClipRecord]) -> list[ClipFoldRecord]:
    return [
        ClipFoldRecord(filename=record.filename, site=site_from_filename(record.filename))
        for record in records
    ]


def folds_for_records(
    records: list[SoundscapeClipRecord],
    splitter: str,
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    fold_records = build_fold_records(records)
    if splitter == "site_balanced":
        return site_balanced_file_folds(fold_records, n_splits=n_splits)
    if splitter == "file_gkf":
        return file_group_kfold(fold_records, n_splits=n_splits)
    if splitter == "site_holdout":
        return site_holdout_folds(fold_records, n_splits=n_splits)
    raise ValueError(f"unsupported splitter: {splitter}")


def truth_by_window(records: list[SoundscapeClipRecord]) -> dict[tuple[str, int], np.ndarray]:
    truth: dict[tuple[str, int], np.ndarray] = {}
    for record in records:
        for window_offset, end_sec in enumerate(record.window_end_secs):
            if record.window_mask[window_offset] <= 0.0:
                continue
            truth.setdefault(
                (record.filename, end_sec),
                record.window_targets[window_offset].astype(np.float32, copy=True),
            )
    return truth


def evaluate_prediction_map(
    truth_map: dict[tuple[str, int], np.ndarray],
    prediction_map: dict[tuple[str, int], np.ndarray],
    class_labels: tuple[str, ...],
) -> float:
    keys = sorted(truth_map)
    y_true = np.stack([truth_map[key] for key in keys]).astype(np.float64)
    y_pred = np.stack(
        [prediction_map.get(key, np.zeros(len(class_labels), dtype=np.float32)) for key in keys]
    ).astype(np.float64)
    macro_auc, _ = macro_roc_auc(y_true, y_pred, class_names=class_labels)
    return macro_auc


def ensure_train_audio_teacher_cache(
    data_dir: Path,
    model_dir: Path,
    cache_path: Path,
    *,
    extract_max_files: int,
    extract_max_chunks_per_file: int | None,
    extract_batch_size: int,
) -> None:
    if cache_path.exists():
        return
    extract_train_audio_embeddings(
        data_dir=data_dir,
        model_dir=model_dir,
        cache_path=cache_path,
        force_rebuild=False,
        batch_size=extract_batch_size,
        max_files=extract_max_files,
        max_chunks_per_file=extract_max_chunks_per_file,
    )


def load_mapped_class_mask(data_dir: Path, model_dir: Path) -> np.ndarray:
    _, mapped_mask = build_perch_mapping(data_dir, model_dir)
    return mapped_mask.astype(np.float32)


def train_stage1_if_enabled(
    *,
    enabled: bool,
    data_dir: Path,
    model_dir: Path,
    cache_path: Path,
    audio_cfg: AudioConfig,
    distill_cfg: DistillConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    batch_size: int,
    hidden_dim: int,
    dropout_rate: float,
    backbone_trainable_layers: int,
    stage1_clip_limit: int | None,
    extract_max_files: int,
    extract_max_chunks_per_file: int | None,
    extract_batch_size: int,
    mapped_class_mask: np.ndarray,
) -> tuple[list[np.ndarray] | None, StageResult | None]:
    if not enabled:
        return None, None

    ensure_train_audio_teacher_cache(
        data_dir=data_dir,
        model_dir=model_dir,
        cache_path=cache_path,
        extract_max_files=extract_max_files,
        extract_max_chunks_per_file=extract_max_chunks_per_file,
        extract_batch_size=extract_batch_size,
    )

    train_audio_records, _ = build_train_audio_perch_clip_index(
        data_dir=data_dir,
        teacher_cache_path=cache_path,
        audio_cfg=audio_cfg,
        distill_cfg=distill_cfg,
    )
    if stage1_clip_limit is not None:
        train_audio_records = train_audio_records[:stage1_clip_limit]

    stage1_model = build_sed_model(
        audio_cfg=audio_cfg,
        model_cfg=model_cfg,
        num_windows=distill_cfg.teacher_windows_per_clip,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        backbone_trainable_layers=backbone_trainable_layers,
    )
    stage1_sequence = TrainAudioSequence(
        train_audio_records,
        audio_cfg=audio_cfg,
        batch_size=batch_size,
        shuffle=True,
    )
    stage1_result = run_train_audio_stage(
        stage1_model,
        stage1_sequence,
        train_cfg=train_cfg,
        distill_cfg=distill_cfg,
        mapped_class_mask=mapped_class_mask,
    )
    return clone_weights(stage1_model), stage1_result


def run_cv_experiment(
    *,
    name: str,
    dataset_name: str,
    soundscape_records: list[SoundscapeClipRecord],
    class_labels: tuple[str, ...],
    audio_cfg: AudioConfig,
    distill_cfg: DistillConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    batch_size: int,
    splitter: str,
    n_splits: int,
    include_teacher_distill: bool,
    pretrained_weights: list[np.ndarray] | None,
    hidden_dim: int,
    dropout_rate: float,
    backbone_trainable_layers: int,
    mapped_class_mask: np.ndarray,
    stage1_result: StageResult | None,
) -> ExperimentSummary:
    folds = folds_for_records(soundscape_records, splitter=splitter, n_splits=n_splits)
    all_truth = truth_by_window(soundscape_records)
    oof_predictions: dict[tuple[str, int], np.ndarray] = {}
    fold_summaries: list[FoldSummary] = []

    for fold_id, (train_indices, val_indices) in enumerate(folds, start=1):
        train_records = [soundscape_records[index] for index in train_indices]
        val_records = [soundscape_records[index] for index in val_indices]

        model = build_sed_model(
            audio_cfg=audio_cfg,
            model_cfg=model_cfg,
            num_windows=distill_cfg.teacher_windows_per_clip,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            backbone_trainable_layers=backbone_trainable_layers,
        )
        if pretrained_weights is not None:
            restore_weights(model, pretrained_weights)

        train_sequence = SoundscapeSequence(
            train_records,
            audio_cfg=audio_cfg,
            batch_size=batch_size,
            shuffle=True,
        )
        val_sequence = SoundscapeSequence(
            val_records,
            audio_cfg=audio_cfg,
            batch_size=batch_size,
            shuffle=False,
        )
        stage_result = run_soundscape_stage(
            model,
            train_sequence,
            val_sequence,
            train_cfg=train_cfg,
            distill_cfg=distill_cfg,
            mapped_class_mask=mapped_class_mask,
            include_teacher_distill=include_teacher_distill,
        )
        prediction_map = predict_soundscape_windows(
            model,
            val_records,
            audio_cfg=audio_cfg,
            batch_size=batch_size,
        )
        fold_truth = truth_by_window(val_records)
        fold_auc = evaluate_prediction_map(
            truth_map=fold_truth,
            prediction_map=prediction_map,
            class_labels=class_labels,
        )
        for key in fold_truth:
            oof_predictions[key] = prediction_map.get(
                key, np.zeros(len(class_labels), dtype=np.float32)
            )
        fold_summaries.append(
            FoldSummary(
                fold_id=fold_id,
                windows=len(fold_truth),
                files=len({record.filename for record in val_records}),
                auc=fold_auc,
                train_stage=stage_result,
            )
        )

    oof_macro_auc = evaluate_prediction_map(
        truth_map=all_truth,
        prediction_map=oof_predictions,
        class_labels=class_labels,
    )
    return ExperimentSummary(
        name=name,
        dataset_name=dataset_name,
        splitter=splitter,
        n_splits=n_splits,
        stage1_enabled=pretrained_weights is not None,
        include_teacher_distill=include_teacher_distill,
        stage1_result=stage1_result,
        oof_macro_auc=oof_macro_auc,
        folds=tuple(fold_summaries),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train 20-second SED with optional Perch teacher warmstart"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--train-audio-teacher-cache", type=Path, default=DEFAULT_TRAIN_AUDIO_TEACHER_CACHE
    )
    parser.add_argument(
        "--soundscape-teacher-cache", type=Path, default=DEFAULT_SOUNDSCAPE_TEACHER_CACHE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--soundscape-dataset", choices=["all66", "full59"], default="all66")
    parser.add_argument(
        "--splitter",
        choices=["site_balanced", "file_gkf", "site_holdout"],
        default="site_balanced",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--backbone-trainable-layers", type=int, default=20)
    parser.add_argument("--stage1-enable", action="store_true")
    parser.add_argument("--stage1-max-files", type=int, default=512)
    parser.add_argument("--stage1-max-chunks-per-file", type=int, default=8)
    parser.add_argument("--stage1-clip-limit", type=int, default=1024)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-epochs", type=int, default=2)
    parser.add_argument("--stage2-lr", type=float, default=2e-4)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.3)
    parser.add_argument("--compare-label-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = load_primary_labels(args.data_dir)
    mapped_class_mask = load_mapped_class_mask(args.data_dir, args.model_dir)

    audio_cfg = AudioConfig()
    distill_cfg = DistillConfig(
        enabled=True,
        teacher_loss_weight=args.teacher_loss_weight,
        teacher_cache_path=str(args.train_audio_teacher_cache),
    )
    model_cfg = ModelConfig(num_classes=len(class_labels))

    soundscape_records = build_soundscape_clip_index(
        args.data_dir,
        class_labels=class_labels,
        teacher_cache_path=args.soundscape_teacher_cache,
        dataset_name=args.soundscape_dataset,
        audio_cfg=audio_cfg,
        distill_cfg=distill_cfg,
    )

    stage1_weights, stage1_result = train_stage1_if_enabled(
        enabled=args.stage1_enable,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_path=args.train_audio_teacher_cache,
        audio_cfg=audio_cfg,
        distill_cfg=distill_cfg,
        model_cfg=model_cfg,
        train_cfg=TrainConfig(
            epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            lr=args.stage1_lr,
        ),
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        backbone_trainable_layers=args.backbone_trainable_layers,
        stage1_clip_limit=args.stage1_clip_limit,
        extract_max_files=args.stage1_max_files,
        extract_max_chunks_per_file=args.stage1_max_chunks_per_file,
        extract_batch_size=args.extract_batch_size,
        mapped_class_mask=mapped_class_mask,
    )

    stage2_cfg = TrainConfig(
        epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        lr=args.stage2_lr,
    )

    experiment_summaries: list[ExperimentSummary] = []
    if args.compare_label_only:
        experiment_summaries.append(
            run_cv_experiment(
                name="label_only",
                dataset_name=args.soundscape_dataset,
                soundscape_records=soundscape_records,
                class_labels=class_labels,
                audio_cfg=audio_cfg,
                distill_cfg=distill_cfg,
                model_cfg=model_cfg,
                train_cfg=stage2_cfg,
                batch_size=args.batch_size,
                splitter=args.splitter,
                n_splits=args.n_splits,
                include_teacher_distill=False,
                pretrained_weights=None,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout_rate,
                backbone_trainable_layers=args.backbone_trainable_layers,
                mapped_class_mask=mapped_class_mask,
                stage1_result=None,
            )
        )

    experiment_summaries.append(
        run_cv_experiment(
            name="teacher_warmstart" if args.stage1_enable else "teacher_soundscape_only",
            dataset_name=args.soundscape_dataset,
            soundscape_records=soundscape_records,
            class_labels=class_labels,
            audio_cfg=audio_cfg,
            distill_cfg=distill_cfg,
            model_cfg=model_cfg,
            train_cfg=stage2_cfg,
            batch_size=args.batch_size,
            splitter=args.splitter,
            n_splits=args.n_splits,
            include_teacher_distill=True,
            pretrained_weights=stage1_weights,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            backbone_trainable_layers=args.backbone_trainable_layers,
            mapped_class_mask=mapped_class_mask,
            stage1_result=stage1_result,
        )
    )

    payload = {
        "config": {
            "soundscape_dataset": args.soundscape_dataset,
            "splitter": args.splitter,
            "n_splits": args.n_splits,
            "batch_size": args.batch_size,
            "stage1_enable": args.stage1_enable,
        },
        "experiments": [asdict(summary) for summary in experiment_summaries],
    }
    output_path = args.output_dir / "sed_teacher_cv_summary.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
