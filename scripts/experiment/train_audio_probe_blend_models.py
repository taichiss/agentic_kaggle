# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np

from scripts.experiment.cv_soundscape_validation import (
    DEFAULT_DATA_DIR,
    SoundscapeWindow,
    auc_binary,
    folds_for_experiment,
    load_primary_labels,
    load_soundscape_rows,
)
from scripts.experiment.perch_probe_cv import (
    DEFAULT_CACHE_PATH as DEFAULT_SOUNDSCAPE_CACHE_PATH,
)
from scripts.experiment.perch_probe_cv import (
    DEFAULT_MODEL_DIR,
    align_rows_and_features,
    build_perch_mapping,
    extract_perch_features,
    macro_auc,
)
from scripts.experiment.train_audio_mlp_head import (
    DEFAULT_BUNDLE_NAME,
    DEFAULT_CACHE_PATH,
    DEFAULT_META_NAME,
    DEFAULT_OUTPUT_DIR,
    build_class_weights,
    build_model,
    build_sample_weights_from_targets,
    build_targets,
    fit_stage,
    make_initial_bias,
    predict_train_audio_mlp_head,
    resolve_train_audio_cache_path,
    set_seed,
)
from scripts.experiment.train_perch_probe_models import default_patterns, predict_probe_bundle


@dataclass(frozen=True)
class TrainAudioPayload:
    embeddings: np.ndarray
    groups: np.ndarray
    targets: np.ndarray
    sample_weights: np.ndarray
    primary_labels: list[str]
    class_weights: np.ndarray


@dataclass(frozen=True)
class BlendEvaluation:
    pattern: str
    splitter: str
    n_splits: int
    probe_oof_macro_auc: float
    head_oof_macro_auc: float
    blend_oof_macro_auc: float
    active_classes: int
    mean_alpha: float
    nonzero_alpha_classes: int


def build_train_audio_payload(
    cache: dict[str, np.ndarray],
    data_dir: Path,
    primary_weight: float,
    secondary_weight: float,
) -> TrainAudioPayload:
    primary_labels = load_primary_labels(data_dir)
    class_weights = build_class_weights(primary_labels, data_dir)
    targets = build_targets(cache, primary_labels, primary_weight, secondary_weight)
    sample_weights = build_sample_weights_from_targets(targets, class_weights)
    return TrainAudioPayload(
        embeddings=cache["embeddings"].astype(np.float32),
        groups=cache["filenames"].astype(str),
        targets=targets,
        sample_weights=sample_weights,
        primary_labels=primary_labels,
        class_weights=class_weights,
    )


def build_soundscape_payload_from_rows(
    rows: list[SoundscapeWindow],
    features: list[Any],
    label_to_idx: dict[str, int],
    class_weights: np.ndarray,
) -> dict[str, Any]:
    aligned_rows, logits, embeddings, labels = align_rows_and_features(rows, features, label_to_idx)
    targets = labels.astype(np.float32)
    groups = np.asarray([row.filename for row in aligned_rows], dtype=str)
    sample_weights = build_sample_weights_from_targets(targets, class_weights)
    return {
        "rows": aligned_rows,
        "raw_logits": logits.astype(np.float32),
        "embeddings": embeddings.astype(np.float32),
        "targets": targets,
        "sample_weights": sample_weights,
        "groups": groups,
    }


def extract_head_bundle(
    model: Any,
    primary_labels: list[str],
    train_audio_cache_path: Path,
    soundscape_cache_path: Path,
    soundscape_dataset: str,
    primary_weight: float,
    secondary_weight: float,
    seed: int,
    unmapped_indices: np.ndarray,
    stage_metrics: list[dict[str, Any]],
    hidden_dim: int,
    dropout_rate: float,
) -> dict[str, Any]:
    layer_norm = model.get_layer("layer_norm")
    hidden = model.get_layer("hidden")
    output = model.get_layer("logits")
    gamma, beta = [np.asarray(weight, dtype=np.float32) for weight in layer_norm.get_weights()]
    hidden_kernel, hidden_bias = [
        np.asarray(weight, dtype=np.float32) for weight in hidden.get_weights()
    ]
    output_kernel, output_bias = [
        np.asarray(weight, dtype=np.float32) for weight in output.get_weights()
    ]
    return {
        "type": "train_audio_pretrained_soundscape_finetuned_mlp_head",
        "architecture": "LayerNorm->Dense(512, gelu)->Dropout->Dense(234)",
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "primary_labels": primary_labels,
        "layer_norm_epsilon": 1e-6,
        "layer_norm_gamma": gamma,
        "layer_norm_beta": beta,
        "hidden_kernel": hidden_kernel,
        "hidden_bias": hidden_bias,
        "output_kernel": output_kernel,
        "output_bias": output_bias,
        "unmapped_class_indices": unmapped_indices.astype(np.int32),
        "stage1_cache_path": str(train_audio_cache_path),
        "stage2_cache_path": str(soundscape_cache_path),
        "soundscape_dataset": soundscape_dataset,
        "primary_weight": primary_weight,
        "secondary_weight": secondary_weight,
        "seed": seed,
        "stage_metrics": stage_metrics,
    }


def train_head_bundle(
    train_audio_payload: TrainAudioPayload,
    soundscape_payload: dict[str, Any],
    train_audio_cache_path: Path,
    soundscape_cache_path: Path,
    soundscape_dataset: str,
    hidden_dim: int,
    dropout_rate: float,
    stage1_epochs: int,
    stage2_epochs: int,
    batch_size: int,
    stage1_learning_rate: float,
    stage2_learning_rate: float,
    stage1_val_fraction: float,
    stage2_val_fraction: float,
    primary_weight: float,
    secondary_weight: float,
    seed: int,
    unmapped_indices: np.ndarray,
) -> dict[str, Any]:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    initial_targets = (
        train_audio_payload.targets if stage1_epochs > 0 else soundscape_payload["targets"]
    )
    model = build_model(
        input_dim=train_audio_payload.embeddings.shape[1],
        output_dim=train_audio_payload.targets.shape[1],
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        learning_rate=stage1_learning_rate,
        initial_bias=make_initial_bias(initial_targets),
    )

    stage_metrics: list[dict[str, Any]] = []
    stage1_metrics = fit_stage(
        model=model,
        stage_name="stage1_train_audio",
        embeddings=train_audio_payload.embeddings,
        targets=train_audio_payload.targets,
        sample_weights=train_audio_payload.sample_weights,
        groups=train_audio_payload.groups,
        learning_rate=stage1_learning_rate,
        epochs=stage1_epochs,
        batch_size=batch_size,
        val_fraction=stage1_val_fraction,
        seed=seed,
    )
    if stage1_metrics is not None:
        stage_metrics.append(asdict(stage1_metrics))

    stage2_metrics = fit_stage(
        model=model,
        stage_name=f"stage2_soundscape_{soundscape_dataset}",
        embeddings=soundscape_payload["embeddings"],
        targets=soundscape_payload["targets"],
        sample_weights=soundscape_payload["sample_weights"],
        groups=soundscape_payload["groups"],
        learning_rate=stage2_learning_rate,
        epochs=stage2_epochs,
        batch_size=batch_size,
        val_fraction=stage2_val_fraction,
        seed=seed,
    )
    if stage2_metrics is not None:
        stage_metrics.append(asdict(stage2_metrics))

    return extract_head_bundle(
        model=model,
        primary_labels=train_audio_payload.primary_labels,
        train_audio_cache_path=train_audio_cache_path,
        soundscape_cache_path=soundscape_cache_path,
        soundscape_dataset=soundscape_dataset,
        primary_weight=primary_weight,
        secondary_weight=secondary_weight,
        seed=seed,
        unmapped_indices=unmapped_indices,
        stage_metrics=stage_metrics,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )


def fit_alpha_vector(
    targets: np.ndarray,
    probe_scores: np.ndarray,
    head_scores: np.ndarray,
    alpha_grid: np.ndarray,
) -> np.ndarray:
    alpha_vector = np.zeros(targets.shape[1], dtype=np.float32)
    for class_idx in range(targets.shape[1]):
        y_true = targets[:, class_idx].astype(int).tolist()
        baseline_auc = auc_binary(y_true, probe_scores[:, class_idx].astype(float).tolist())
        best_auc = -1.0 if baseline_auc is None else float(baseline_auc)
        best_alpha = 0.0
        for alpha in alpha_grid:
            blended = (1.0 - float(alpha)) * probe_scores[:, class_idx] + float(
                alpha
            ) * head_scores[:, class_idx]
            candidate_auc = auc_binary(y_true, blended.astype(float).tolist())
            if candidate_auc is None:
                continue
            if candidate_auc > best_auc + 1e-9:
                best_auc = float(candidate_auc)
                best_alpha = float(alpha)
        alpha_vector[class_idx] = best_alpha
    return alpha_vector


def blend_scores(
    probe_scores: np.ndarray,
    head_scores: np.ndarray,
    alpha_vector: np.ndarray,
) -> np.ndarray:
    return (
        (1.0 - alpha_vector[None, :]) * probe_scores + alpha_vector[None, :] * head_scores
    ).astype(np.float32)


def find_pattern(pattern_name: str) -> Any:
    for pattern in default_patterns():
        if pattern.name == pattern_name:
            return pattern
    raise ValueError(f"unknown pattern: {pattern_name}")


def evaluate_pattern(
    pattern_name: str,
    probe_pattern_dir: Path,
    rows: list[SoundscapeWindow],
    features: list[Any],
    train_audio_payload: TrainAudioPayload,
    label_to_idx: dict[str, int],
    hidden_dim: int,
    dropout_rate: float,
    stage1_epochs: int,
    stage2_epochs: int,
    batch_size: int,
    stage1_learning_rate: float,
    stage2_learning_rate: float,
    stage1_val_fraction: float,
    stage2_val_fraction: float,
    primary_weight: float,
    secondary_weight: float,
    seed: int,
    train_audio_cache_path: Path,
    soundscape_cache_path: Path,
    soundscape_dataset: str,
    unmapped_indices: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[BlendEvaluation, np.ndarray]:
    pattern = find_pattern(pattern_name)
    soundscape_payload = build_soundscape_payload_from_rows(
        rows=rows,
        features=features,
        label_to_idx=label_to_idx,
        class_weights=train_audio_payload.class_weights,
    )
    folds = folds_for_experiment(soundscape_payload["rows"], pattern.splitter, pattern.n_splits)

    probe_oof = np.zeros_like(soundscape_payload["raw_logits"], dtype=np.float32)
    head_oof = np.zeros_like(soundscape_payload["raw_logits"], dtype=np.float32)
    blend_oof = np.zeros_like(soundscape_payload["raw_logits"], dtype=np.float32)
    alpha_vectors: list[np.ndarray] = []

    for fold_id, (train_indices, val_indices) in enumerate(folds, start=1):
        fold_rows = [soundscape_payload["rows"][index] for index in train_indices]
        fold_payload = build_soundscape_payload_from_rows(
            rows=fold_rows,
            features=features,
            label_to_idx=label_to_idx,
            class_weights=train_audio_payload.class_weights,
        )
        head_bundle = train_head_bundle(
            train_audio_payload=train_audio_payload,
            soundscape_payload=fold_payload,
            train_audio_cache_path=train_audio_cache_path,
            soundscape_cache_path=soundscape_cache_path,
            soundscape_dataset=soundscape_dataset,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            batch_size=batch_size,
            stage1_learning_rate=stage1_learning_rate,
            stage2_learning_rate=stage2_learning_rate,
            stage1_val_fraction=stage1_val_fraction,
            stage2_val_fraction=stage2_val_fraction,
            primary_weight=primary_weight,
            secondary_weight=secondary_weight,
            seed=seed + fold_id,
            unmapped_indices=unmapped_indices,
        )

        probe_bundle = joblib.load(
            probe_pattern_dir / f"fold_{fold_id:02d}" / "probe_bundle.joblib"
        )
        train_probe_scores = predict_probe_bundle(
            bundle=probe_bundle,
            raw_logits=soundscape_payload["raw_logits"][train_indices],
            embeddings=soundscape_payload["embeddings"][train_indices],
        )
        val_probe_scores = predict_probe_bundle(
            bundle=probe_bundle,
            raw_logits=soundscape_payload["raw_logits"][val_indices],
            embeddings=soundscape_payload["embeddings"][val_indices],
        )
        train_head_scores = predict_train_audio_mlp_head(
            head_bundle,
            soundscape_payload["embeddings"][train_indices],
        )
        val_head_scores = predict_train_audio_mlp_head(
            head_bundle,
            soundscape_payload["embeddings"][val_indices],
        )
        alpha_vector = fit_alpha_vector(
            targets=soundscape_payload["targets"][train_indices],
            probe_scores=train_probe_scores,
            head_scores=train_head_scores,
            alpha_grid=alpha_grid,
        )
        alpha_vectors.append(alpha_vector)

        probe_oof[val_indices] = val_probe_scores
        head_oof[val_indices] = val_head_scores
        blend_oof[val_indices] = blend_scores(val_probe_scores, val_head_scores, alpha_vector)

    probe_auc, active_classes = macro_auc(soundscape_payload["targets"], probe_oof)
    head_auc, _ = macro_auc(soundscape_payload["targets"], head_oof)
    blend_auc, _ = macro_auc(soundscape_payload["targets"], blend_oof)
    alpha_matrix = np.stack(alpha_vectors).astype(np.float32)
    evaluation = BlendEvaluation(
        pattern=pattern.name,
        splitter=pattern.splitter,
        n_splits=pattern.n_splits,
        probe_oof_macro_auc=probe_auc,
        head_oof_macro_auc=head_auc,
        blend_oof_macro_auc=blend_auc,
        active_classes=active_classes,
        mean_alpha=float(alpha_matrix.mean()),
        nonzero_alpha_classes=int((alpha_matrix.mean(axis=0) > 0.0).sum()),
    )
    return evaluation, alpha_matrix.mean(axis=0).astype(np.float32)


def save_final_bundle(
    output_dir: Path,
    head_bundle: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(head_bundle, output_dir / DEFAULT_BUNDLE_NAME, compress=3)
    (output_dir / DEFAULT_META_NAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-audio-cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--soundscape-cache-path", type=Path, default=DEFAULT_SOUNDSCAPE_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-pattern", type=str, default="main_all66_sitebalanced3")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--stage1-epochs", type=int, default=4)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--stage1-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=2e-5)
    parser.add_argument("--stage1-val-fraction", type=float, default=0.1)
    parser.add_argument("--stage2-val-fraction", type=float, default=0.1)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--secondary-weight", type=float, default=0.7)
    parser.add_argument("--soundscape-dataset", type=str, default="all66")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-grid-step", type=float, default=0.1)
    args = parser.parse_args()

    train_audio_cache_path = resolve_train_audio_cache_path(args.train_audio_cache_path)
    train_audio_cache = dict(np.load(train_audio_cache_path, allow_pickle=False))
    train_audio_payload = build_train_audio_payload(
        cache=train_audio_cache,
        data_dir=args.data_dir,
        primary_weight=args.primary_weight,
        secondary_weight=args.secondary_weight,
    )

    primary_labels = train_audio_payload.primary_labels
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}
    all_rows = load_soundscape_rows(args.data_dir)
    if args.soundscape_dataset != "all66":
        raise ValueError(f"unsupported soundscape dataset: {args.soundscape_dataset}")

    extracted_features = extract_perch_features(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_path=args.soundscape_cache_path,
        force_rebuild=False,
    )
    _, mapped_mask = build_perch_mapping(args.data_dir, args.model_dir)
    unmapped_indices = np.flatnonzero(~mapped_mask).astype(np.int32)
    alpha_grid = np.arange(0.0, 1.0 + args.alpha_grid_step * 0.5, args.alpha_grid_step).astype(
        np.float32
    )

    probe_pattern_dir = args.output_dir / "cv_models" / args.probe_pattern
    evaluation, mean_alpha = evaluate_pattern(
        pattern_name=args.probe_pattern,
        probe_pattern_dir=probe_pattern_dir,
        rows=all_rows,
        features=extracted_features,
        train_audio_payload=train_audio_payload,
        label_to_idx=label_to_idx,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        stage1_learning_rate=args.stage1_learning_rate,
        stage2_learning_rate=args.stage2_learning_rate,
        stage1_val_fraction=args.stage1_val_fraction,
        stage2_val_fraction=args.stage2_val_fraction,
        primary_weight=args.primary_weight,
        secondary_weight=args.secondary_weight,
        seed=args.seed,
        train_audio_cache_path=train_audio_cache_path,
        soundscape_cache_path=args.soundscape_cache_path,
        soundscape_dataset=args.soundscape_dataset,
        unmapped_indices=unmapped_indices,
        alpha_grid=alpha_grid,
    )

    full_soundscape_payload = build_soundscape_payload_from_rows(
        rows=all_rows,
        features=extracted_features,
        label_to_idx=label_to_idx,
        class_weights=train_audio_payload.class_weights,
    )
    full_head_bundle = train_head_bundle(
        train_audio_payload=train_audio_payload,
        soundscape_payload=full_soundscape_payload,
        train_audio_cache_path=train_audio_cache_path,
        soundscape_cache_path=args.soundscape_cache_path,
        soundscape_dataset=args.soundscape_dataset,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        stage1_learning_rate=args.stage1_learning_rate,
        stage2_learning_rate=args.stage2_learning_rate,
        stage1_val_fraction=args.stage1_val_fraction,
        stage2_val_fraction=args.stage2_val_fraction,
        primary_weight=args.primary_weight,
        secondary_weight=args.secondary_weight,
        seed=args.seed,
        unmapped_indices=unmapped_indices,
    )

    full_probe_bundle = joblib.load(probe_pattern_dir / "full_fit" / "probe_bundle.joblib")
    full_probe_scores = predict_probe_bundle(
        bundle=full_probe_bundle,
        raw_logits=full_soundscape_payload["raw_logits"],
        embeddings=full_soundscape_payload["embeddings"],
    )
    full_head_scores = predict_train_audio_mlp_head(
        full_head_bundle,
        full_soundscape_payload["embeddings"],
    )
    full_alpha = fit_alpha_vector(
        targets=full_soundscape_payload["targets"],
        probe_scores=full_probe_scores,
        head_scores=full_head_scores,
        alpha_grid=alpha_grid,
    )

    full_blend_scores = blend_scores(full_probe_scores, full_head_scores, full_alpha)
    full_blend_auc, _ = macro_auc(full_soundscape_payload["targets"], full_blend_scores)
    full_head_bundle["probe_blend_alpha_by_class"] = full_alpha.astype(np.float32)
    full_head_bundle["probe_blend_pattern"] = args.probe_pattern
    full_head_bundle["probe_blend_alpha_grid"] = alpha_grid.astype(np.float32)
    full_head_bundle["cv_probe_oof_macro_auc"] = evaluation.probe_oof_macro_auc
    full_head_bundle["cv_head_oof_macro_auc"] = evaluation.head_oof_macro_auc
    full_head_bundle["cv_blend_oof_macro_auc"] = evaluation.blend_oof_macro_auc
    full_head_bundle["full_fit_blend_train_auc"] = full_blend_auc

    report = {
        "probe_pattern": args.probe_pattern,
        "soundscape_dataset": args.soundscape_dataset,
        "train_audio_cache_path": str(train_audio_cache_path),
        "soundscape_cache_path": str(args.soundscape_cache_path),
        "config": {
            "hidden_dim": args.hidden_dim,
            "dropout_rate": args.dropout_rate,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "batch_size": args.batch_size,
            "stage1_learning_rate": args.stage1_learning_rate,
            "stage2_learning_rate": args.stage2_learning_rate,
            "stage1_val_fraction": args.stage1_val_fraction,
            "stage2_val_fraction": args.stage2_val_fraction,
            "primary_weight": args.primary_weight,
            "secondary_weight": args.secondary_weight,
            "seed": args.seed,
            "alpha_grid_step": args.alpha_grid_step,
        },
        "evaluation": asdict(evaluation),
        "full_fit_blend_train_auc": full_blend_auc,
        "full_fit_mean_alpha": float(full_alpha.mean()),
        "full_fit_nonzero_alpha_classes": int((full_alpha > 0.0).sum()),
        "mean_fold_alpha": mean_alpha.tolist(),
    }
    save_final_bundle(args.output_dir, full_head_bundle, report)
    (args.output_dir / f"train_audio_probe_blend_report_{args.probe_pattern}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report["evaluation"], indent=2, ensure_ascii=False))
    print(f"full_fit_blend_train_auc={full_blend_auc:.6f}")
    print(
        "full_fit_alpha_summary="
        f"mean={float(full_alpha.mean()):.4f} nonzero={int((full_alpha > 0.0).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
