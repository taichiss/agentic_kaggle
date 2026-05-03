# ruff: noqa: E402
"""Train a single Perch MLP head with train_audio pretraining and soundscape finetuning."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scripts.experiment.cv_soundscape_validation import (
    filter_fully_labeled_rows,
    load_soundscape_rows,
)
from scripts.experiment.perch_probe_cv import (
    align_rows_and_features,
    build_perch_mapping,
    extract_perch_features,
    load_primary_labels,
)

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "input" / "BirdCLEF+ 2026"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models"
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "models" / "train_audio_perch_cache.npz"
DEFAULT_FALLBACK_CACHE_PATH = REPO_ROOT / "data" / "models" / "train_audio_perch_chunks_v1.npz"
DEFAULT_SOUNDSCAPE_CACHE_PATH = REPO_ROOT / "data" / "models" / "perch_labeled_cache_v1.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "models"
DEFAULT_BUNDLE_NAME = "train_audio_mlp_head_bundle.joblib"
DEFAULT_META_NAME = "train_audio_mlp_head_meta.json"
LAYER_NORM_EPSILON = 1e-6


@dataclass(frozen=True)
class StageMetrics:
    name: str
    train_rows: int
    val_rows: int
    train_groups: int
    val_groups: int
    learning_rate: float
    epochs_requested: int
    epochs_completed: int
    train_loss: float
    val_loss: float | None
    history: dict[str, list[float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_train_audio_cache_path(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    if cache_path == DEFAULT_CACHE_PATH and DEFAULT_FALLBACK_CACHE_PATH.exists():
        print(
            f"Primary train_audio cache not found at {cache_path}; "
            f"using fallback {DEFAULT_FALLBACK_CACHE_PATH}"
        )
        return DEFAULT_FALLBACK_CACHE_PATH
    return cache_path


def build_targets(
    cache: dict[str, np.ndarray],
    primary_labels: list[str],
    primary_weight: float,
    secondary_weight: float,
) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}
    n_chunks = len(cache["primary_labels_per_chunk"])
    n_classes = len(primary_labels)
    targets = np.zeros((n_chunks, n_classes), dtype=np.float32)

    for index in range(n_chunks):
        primary_label = str(cache["primary_labels_per_chunk"][index])
        if primary_label in label_to_idx:
            targets[index, label_to_idx[primary_label]] = primary_weight

        secondary_labels_text = str(cache["secondary_labels_per_chunk"][index])
        try:
            secondary_labels = ast.literal_eval(secondary_labels_text)
        except (ValueError, SyntaxError):
            secondary_labels = []
        if not isinstance(secondary_labels, list):
            continue

        for secondary_label in secondary_labels:
            label = str(secondary_label).strip()
            if label and label in label_to_idx:
                class_index = label_to_idx[label]
                targets[index, class_index] = max(targets[index, class_index], secondary_weight)
    return targets


def build_class_weights(primary_labels: list[str], data_dir: Path) -> np.ndarray:
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    label_to_class = dict(
        zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str), strict=True)
    )
    multipliers = {
        "Aves": 1.0,
        "Amphibia": 3.0,
        "Insecta": 3.0,
        "Mammalia": 3.0,
        "Reptilia": 5.0,
    }
    return np.asarray(
        [multipliers.get(label_to_class.get(label, "Aves"), 1.0) for label in primary_labels],
        dtype=np.float32,
    )


def build_sample_weights_from_targets(
    targets: np.ndarray,
    class_weights: np.ndarray,
) -> np.ndarray:
    weighted = targets * class_weights[None, :]
    weights = weighted.max(axis=1)
    return np.where(weights > 0.0, weights, 1.0).astype(np.float32)


def grouped_train_validation_split(
    groups: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0.0:
        all_indices = np.arange(len(groups), dtype=np.int64)
        return all_indices, np.zeros(0, dtype=np.int64)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        all_indices = np.arange(len(groups), dtype=np.int64)
        return all_indices, np.zeros(0, dtype=np.int64)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_indices, val_indices = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_indices.astype(np.int64), val_indices.astype(np.int64)


def make_initial_bias(targets: np.ndarray) -> np.ndarray:
    class_priors = np.clip(targets.mean(axis=0), 1e-4, 1.0 - 1e-4)
    return (np.log(class_priors) - np.log1p(-class_priors)).astype(np.float32)


def build_model(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout_rate: float,
    learning_rate: float,
    initial_bias: np.ndarray,
) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_dim,), name="perch_embedding")
    x = tf.keras.layers.LayerNormalization(
        epsilon=LAYER_NORM_EPSILON,
        name="layer_norm",
    )(inputs)
    x = tf.keras.layers.Dense(
        hidden_dim,
        activation=tf.nn.gelu,
        name="hidden",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        output_dim,
        activation=None,
        name="logits",
        bias_initializer=tf.keras.initializers.Constant(initial_bias),
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="perch_domain_adapted_mlp_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    return model


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))


def apply_layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    return normalized * gamma + beta


def predict_train_audio_mlp_head(
    bundle: dict[str, Any],
    embeddings: np.ndarray,
) -> np.ndarray:
    x = embeddings.astype(np.float32, copy=False)
    x = apply_layer_norm(
        x=x,
        gamma=np.asarray(bundle["layer_norm_gamma"], dtype=np.float32),
        beta=np.asarray(bundle["layer_norm_beta"], dtype=np.float32),
        epsilon=float(bundle["layer_norm_epsilon"]),
    )
    x = x @ np.asarray(bundle["hidden_kernel"], dtype=np.float32) + np.asarray(
        bundle["hidden_bias"], dtype=np.float32
    )
    x = gelu(x).astype(np.float32)
    logits = x @ np.asarray(bundle["output_kernel"], dtype=np.float32) + np.asarray(
        bundle["output_bias"], dtype=np.float32
    )
    return logits.astype(np.float32)


def weighted_bce_from_logits(
    targets: np.ndarray,
    logits: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    clipped_logits = np.clip(logits, -30.0, 30.0)
    losses = (
        np.maximum(clipped_logits, 0.0)
        - clipped_logits * targets
        + np.log1p(np.exp(-np.abs(clipped_logits)))
    )
    weighted = losses * sample_weights[:, None]
    return float(weighted.mean())


def build_soundscape_training_payload(
    data_dir: Path,
    model_dir: Path,
    soundscape_cache_path: Path,
    soundscape_dataset: str,
    label_to_idx: dict[str, int],
    force_rebuild_cache: bool,
    class_weights: np.ndarray,
) -> dict[str, Any]:
    rows = load_soundscape_rows(data_dir)
    if soundscape_dataset == "full59":
        rows = filter_fully_labeled_rows(rows)
    elif soundscape_dataset != "all66":
        raise ValueError(f"unsupported soundscape dataset: {soundscape_dataset}")

    features = extract_perch_features(
        data_dir=data_dir,
        model_dir=model_dir,
        cache_path=soundscape_cache_path,
        force_rebuild=force_rebuild_cache,
    )
    aligned_rows, _, embeddings, labels = align_rows_and_features(rows, features, label_to_idx)
    targets = labels.astype(np.float32)
    sample_weights = build_sample_weights_from_targets(targets, class_weights)
    groups = np.asarray([row.filename for row in aligned_rows], dtype=str)
    return {
        "rows": aligned_rows,
        "embeddings": embeddings.astype(np.float32),
        "targets": targets,
        "sample_weights": sample_weights,
        "groups": groups,
        "files": len(np.unique(groups)),
        "sites": len({row.site for row in aligned_rows}),
    }


def set_learning_rate(model: Any, learning_rate: float) -> None:
    optimizer_lr = model.optimizer.learning_rate
    if hasattr(optimizer_lr, "assign"):
        optimizer_lr.assign(learning_rate)
        return
    model.optimizer.learning_rate = learning_rate


def fit_stage(
    model: Any,
    stage_name: str,
    embeddings: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    groups: np.ndarray,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
) -> StageMetrics | None:
    if epochs <= 0:
        return None

    import tensorflow as tf

    train_indices, val_indices = grouped_train_validation_split(groups, val_fraction, seed)
    train_embeddings = embeddings[train_indices]
    train_targets = targets[train_indices]
    train_weights = sample_weights[train_indices]
    val_embeddings = embeddings[val_indices]
    val_targets = targets[val_indices]
    val_weights = sample_weights[val_indices]

    print(
        f"{stage_name}: train_rows={len(train_indices)} val_rows={len(val_indices)} "
        f"train_groups={len(np.unique(groups[train_indices]))} "
        f"val_groups={len(np.unique(groups[val_indices])) if len(val_indices) else 0}"
    )

    set_learning_rate(model, learning_rate)

    callbacks: list[Any] = []
    if len(val_indices):
        callbacks.extend(
            [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                ),
            ]
        )

    fit_kwargs: dict[str, Any] = {
        "x": train_embeddings,
        "y": train_targets,
        "sample_weight": train_weights,
        "epochs": epochs,
        "batch_size": batch_size,
        "verbose": 2,
        "callbacks": callbacks,
        "shuffle": True,
    }
    if len(val_indices):
        fit_kwargs["validation_data"] = (val_embeddings, val_targets, val_weights)

    history = model.fit(**fit_kwargs)
    train_logits = model.predict(train_embeddings, batch_size=batch_size, verbose=0).astype(
        np.float32
    )
    val_logits = (
        model.predict(val_embeddings, batch_size=batch_size, verbose=0).astype(np.float32)
        if len(val_indices)
        else np.zeros((0, targets.shape[1]), dtype=np.float32)
    )

    return StageMetrics(
        name=stage_name,
        train_rows=len(train_indices),
        val_rows=len(val_indices),
        train_groups=len(np.unique(groups[train_indices])),
        val_groups=len(np.unique(groups[val_indices])) if len(val_indices) else 0,
        learning_rate=learning_rate,
        epochs_requested=epochs,
        epochs_completed=len(history.history.get("loss", [])),
        train_loss=weighted_bce_from_logits(train_targets, train_logits, train_weights),
        val_loss=(
            weighted_bce_from_logits(val_targets, val_logits, val_weights)
            if len(val_indices)
            else None
        ),
        history={
            key: [float(value) for value in values] for key, values in history.history.items()
        },
    )


def train_train_audio_head(
    cache: dict[str, np.ndarray],
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    train_audio_cache_path: Path,
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
    soundscape_dataset: str,
    soundscape_cache_path: Path,
    force_rebuild_soundscape_cache: bool,
    seed: int,
) -> dict[str, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    primary_labels = load_primary_labels(data_dir)
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}
    class_weights = build_class_weights(primary_labels, data_dir)

    train_audio_embeddings = cache["embeddings"].astype(np.float32)
    train_audio_groups = cache["filenames"].astype(str)
    train_audio_targets = build_targets(cache, primary_labels, primary_weight, secondary_weight)
    train_audio_sample_weights = build_sample_weights_from_targets(
        train_audio_targets, class_weights
    )

    soundscape_payload = build_soundscape_training_payload(
        data_dir=data_dir,
        model_dir=model_dir,
        soundscape_cache_path=soundscape_cache_path,
        soundscape_dataset=soundscape_dataset,
        label_to_idx=label_to_idx,
        force_rebuild_cache=force_rebuild_soundscape_cache,
        class_weights=class_weights,
    )

    initial_targets = train_audio_targets if stage1_epochs > 0 else soundscape_payload["targets"]
    model = build_model(
        input_dim=train_audio_embeddings.shape[1],
        output_dim=train_audio_targets.shape[1],
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        learning_rate=stage1_learning_rate,
        initial_bias=make_initial_bias(initial_targets),
    )

    stage_metrics: list[StageMetrics] = []
    stage1_metrics = fit_stage(
        model=model,
        stage_name="stage1_train_audio",
        embeddings=train_audio_embeddings,
        targets=train_audio_targets,
        sample_weights=train_audio_sample_weights,
        groups=train_audio_groups,
        learning_rate=stage1_learning_rate,
        epochs=stage1_epochs,
        batch_size=batch_size,
        val_fraction=stage1_val_fraction,
        seed=seed,
    )
    if stage1_metrics is not None:
        stage_metrics.append(stage1_metrics)

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
        stage_metrics.append(stage2_metrics)

    _, mapped_mask = build_perch_mapping(data_dir, model_dir)
    unmapped_indices = np.flatnonzero(~mapped_mask).astype(np.int32)

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

    bundle: dict[str, Any] = {
        "type": "train_audio_pretrained_soundscape_finetuned_mlp_head",
        "architecture": "LayerNorm->Dense(512, gelu)->Dropout->Dense(234)",
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "primary_labels": primary_labels,
        "layer_norm_epsilon": LAYER_NORM_EPSILON,
        "layer_norm_gamma": gamma,
        "layer_norm_beta": beta,
        "hidden_kernel": hidden_kernel,
        "hidden_bias": hidden_bias,
        "output_kernel": output_kernel,
        "output_bias": output_bias,
        "unmapped_class_indices": unmapped_indices,
        "stage1_cache_path": str(train_audio_cache_path),
        "stage2_cache_path": str(soundscape_cache_path),
        "soundscape_dataset": soundscape_dataset,
        "primary_weight": primary_weight,
        "secondary_weight": secondary_weight,
        "seed": seed,
        "stage_metrics": [asdict(item) for item in stage_metrics],
        "soundscape_rows": len(soundscape_payload["embeddings"]),
        "soundscape_files": int(soundscape_payload["files"]),
        "soundscape_sites": int(soundscape_payload["sites"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / DEFAULT_BUNDLE_NAME
    joblib.dump(bundle, bundle_path, compress=3)

    final_soundscape_logits = model.predict(
        soundscape_payload["embeddings"],
        batch_size=batch_size,
        verbose=0,
    ).astype(np.float32)
    final_soundscape_loss = weighted_bce_from_logits(
        soundscape_payload["targets"],
        final_soundscape_logits,
        soundscape_payload["sample_weights"],
    )

    meta = {
        "type": bundle["type"],
        "bundle_path": str(bundle_path),
        "train_audio_rows": int(len(train_audio_embeddings)),
        "train_audio_files": int(len(np.unique(train_audio_groups))),
        "soundscape_rows": int(len(soundscape_payload["embeddings"])),
        "soundscape_files": int(soundscape_payload["files"]),
        "soundscape_sites": int(soundscape_payload["sites"]),
        "n_classes": int(train_audio_targets.shape[1]),
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "stage1_learning_rate": stage1_learning_rate,
        "stage2_learning_rate": stage2_learning_rate,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "stage_metrics": [asdict(item) for item in stage_metrics],
        "final_soundscape_train_loss": final_soundscape_loss,
        "unmapped_classes": int(len(unmapped_indices)),
        "positive_targets_min": float(train_audio_targets.sum(axis=0).min()),
        "positive_targets_max": float(train_audio_targets.sum(axis=0).max()),
        "positive_targets_mean": float(train_audio_targets.sum(axis=0).mean()),
        "soundscape_dataset": soundscape_dataset,
    }
    meta_path = output_dir / DEFAULT_META_NAME
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved bundle to {bundle_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Final soundscape train loss={final_soundscape_loss:.5f}")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a single MLP head via train_audio pretraining and soundscape finetuning"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--soundscape-cache-path", type=Path, default=DEFAULT_SOUNDSCAPE_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
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

    cache_path = resolve_train_audio_cache_path(args.cache_path)
    cache = dict(np.load(cache_path, allow_pickle=False))
    train_train_audio_head(
        cache=cache,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        train_audio_cache_path=cache_path,
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
        soundscape_dataset=args.soundscape_dataset,
        soundscape_cache_path=args.soundscape_cache_path,
        force_rebuild_soundscape_cache=args.force_rebuild_soundscape_cache,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
