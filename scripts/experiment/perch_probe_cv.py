from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from scripts.experiment.cv_soundscape_validation import SoundscapeWindow
else:
    SoundscapeWindow = Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cv_soundscape_validation = importlib.import_module("scripts.experiment.cv_soundscape_validation")
DEFAULT_DATA_DIR = cast(Path, cv_soundscape_validation.DEFAULT_DATA_DIR)
auc_binary = cast(
    Callable[[list[int], list[float]], float | None], cv_soundscape_validation.auc_binary
)
filter_fully_labeled_rows = cast(
    Callable[[list[SoundscapeWindow]], list[SoundscapeWindow]],
    cv_soundscape_validation.filter_fully_labeled_rows,
)
folds_for_experiment = cast(
    Callable[[list[SoundscapeWindow], str, int], list[tuple[list[int], list[int]]]],
    cv_soundscape_validation.folds_for_experiment,
)
load_primary_labels = cast(
    Callable[[Path], list[str]], cv_soundscape_validation.load_primary_labels
)
load_soundscape_rows = cast(
    Callable[[Path], list[SoundscapeWindow]],
    cv_soundscape_validation.load_soundscape_rows,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "perch_labeled_cache_v1.npz"
)
WINDOWS_PER_FILE = 12
WINDOW_SAMPLES = 32_000 * 5
EMBED_PCA_DIM = 64
PROBE_BLEND_ALPHA = 0.4
PROBE_MIN_POS = 3


@dataclass(frozen=True)
class WindowFeatures:
    row_id: str
    filename: str
    site: str
    hour_utc: int
    logits: np.ndarray
    embedding: np.ndarray


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: str
    splitter: str
    n_splits: int
    feature_mode: str


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    auc: float
    active_classes: int
    windows: int
    files: int
    sites: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    dataset: str
    splitter: str
    n_splits: int
    feature_mode: str
    windows: int
    files: int
    sites: int
    active_classes: int
    oof_macro_auc: float
    mean_fold_auc: float
    min_fold_auc: float
    max_fold_auc: float
    min_fold_active_classes: int
    max_fold_active_classes: int
    folds: tuple[FoldResult, ...]


def row_id_from_window(row: SoundscapeWindow) -> str:
    stem = row.filename.removesuffix(".ogg")
    return f"{stem}_{row.end_sec}"


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, int]:
    aucs: list[float] = []
    for class_idx in range(y_true.shape[1]):
        value = auc_binary(
            y_true[:, class_idx].astype(int).tolist(),
            y_score[:, class_idx].astype(float).tolist(),
        )
        if value is not None:
            aucs.append(value)
    return float(sum(aucs) / len(aucs)), len(aucs)


def logit_from_prob(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-5, 1.0 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def load_taxonomy(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "taxonomy.csv")


def build_perch_mapping(data_dir: Path, model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    taxonomy = load_taxonomy(data_dir)
    bc_labels = (
        pd.read_csv(model_dir / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    mapping = taxonomy.merge(bc_labels, on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(-1).astype(int)

    primary_labels = load_primary_labels(data_dir)
    label_to_bc = mapping.set_index("primary_label")["bc_index"]
    bc_indices = np.array([int(label_to_bc.loc[label]) for label in primary_labels], dtype=np.int32)
    mapped_mask = bc_indices >= 0
    return bc_indices, mapped_mask


def read_60s_audio(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) < WINDOWS_PER_FILE * WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOWS_PER_FILE * WINDOW_SAMPLES - len(audio)))
    else:
        audio = audio[: WINDOWS_PER_FILE * WINDOW_SAMPLES]
    return audio.reshape(WINDOWS_PER_FILE, WINDOW_SAMPLES)


def extract_perch_features(
    data_dir: Path,
    model_dir: Path,
    cache_path: Path,
    force_rebuild: bool,
) -> list[WindowFeatures]:
    if cache_path.exists() and not force_rebuild:
        cache = np.load(cache_path, allow_pickle=False)
        return [
            WindowFeatures(
                row_id=row_id,
                filename=filename,
                site=site,
                hour_utc=int(hour),
                logits=logits,
                embedding=embedding,
            )
            for row_id, filename, site, hour, logits, embedding in zip(
                cache["row_ids"].tolist(),
                cache["filenames"].tolist(),
                cache["sites"].tolist(),
                cache["hours"].tolist(),
                cache["logits"],
                cache["embeddings"],
                strict=True,
            )
        ]

    primary_labels = load_primary_labels(data_dir)
    bc_indices, mapped_mask = build_perch_mapping(data_dir, model_dir)
    model = tf.saved_model.load(str(model_dir))
    infer_fn = model.signatures["serving_default"]

    row_lookup = {row_id_from_window(row): row for row in load_soundscape_rows(data_dir)}
    train_soundscape_dir = data_dir / "train_soundscapes"
    file_paths = sorted(train_soundscape_dir / row.filename for row in row_lookup.values())
    file_paths = sorted({path for path in file_paths})

    features: list[WindowFeatures] = []
    mapped_positions = np.where(mapped_mask)[0]
    mapped_bc_indices = bc_indices[mapped_mask]

    for file_path in file_paths:
        windows = read_60s_audio(file_path)
        outputs = infer_fn(inputs=tf.convert_to_tensor(windows))
        logits_raw = outputs["label"].numpy().astype(np.float32)
        embeddings = outputs["embedding"].numpy().astype(np.float32)

        mapped_logits = np.zeros((WINDOWS_PER_FILE, len(primary_labels)), dtype=np.float32)
        mapped_logits[:, mapped_positions] = logits_raw[:, mapped_bc_indices]

        for window_idx in range(WINDOWS_PER_FILE):
            row_id = f"{file_path.stem}_{(window_idx + 1) * 5}"
            if row_id not in row_lookup:
                continue
            row = row_lookup[row_id]
            features.append(
                WindowFeatures(
                    row_id=row_id,
                    filename=row.filename,
                    site=row.site,
                    hour_utc=row.hour_utc,
                    logits=mapped_logits[window_idx],
                    embedding=embeddings[window_idx],
                )
            )

    features.sort(key=lambda item: (item.filename, int(item.row_id.rsplit("_", 1)[1])))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        row_ids=np.array([item.row_id for item in features], dtype=str),
        filenames=np.array([item.filename for item in features], dtype=str),
        sites=np.array([item.site for item in features], dtype=str),
        hours=np.array([item.hour_utc for item in features], dtype=np.int16),
        logits=np.stack([item.logits for item in features]).astype(np.float32),
        embeddings=np.stack([item.embedding for item in features]).astype(np.float32),
    )
    return features


def align_rows_and_features(
    rows: list[SoundscapeWindow],
    features: list[WindowFeatures],
    label_to_idx: dict[str, int],
) -> tuple[list[SoundscapeWindow], np.ndarray, np.ndarray, np.ndarray]:
    feature_by_row_id = {feature.row_id: feature for feature in features}
    aligned_rows: list[SoundscapeWindow] = []
    logits = []
    embeddings = []
    labels = []
    n_classes = len(label_to_idx)

    for row in rows:
        row_id = row_id_from_window(row)
        feature = feature_by_row_id.get(row_id)
        if feature is None:
            continue
        target = np.zeros(n_classes, dtype=np.uint8)
        for label in row.labels:
            idx = label_to_idx.get(label)
            if idx is not None:
                target[idx] = 1
        aligned_rows.append(row)
        logits.append(feature.logits)
        embeddings.append(feature.embedding)
        labels.append(target)

    return (
        aligned_rows,
        np.stack(logits).astype(np.float32),
        np.stack(embeddings).astype(np.float32),
        np.stack(labels).astype(np.uint8),
    )


def build_features_for_fold(
    feature_mode: str,
    train_logits: np.ndarray,
    valid_logits: np.ndarray,
    train_embeddings: np.ndarray,
    valid_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_mode == "raw_perch":
        return train_logits, valid_logits

    if feature_mode != "probe_pca_blend":
        raise ValueError(f"unknown feature_mode: {feature_mode}")

    scaler = StandardScaler()
    train_emb_scaled = scaler.fit_transform(train_embeddings)
    valid_emb_scaled = scaler.transform(valid_embeddings)

    pca_dim = min(EMBED_PCA_DIM, train_emb_scaled.shape[0] - 1, train_emb_scaled.shape[1])
    if pca_dim <= 0:
        train_emb_pca = train_emb_scaled
        valid_emb_pca = valid_emb_scaled
    else:
        pca = PCA(n_components=pca_dim, random_state=42)
        train_emb_pca = pca.fit_transform(train_emb_scaled)
        valid_emb_pca = pca.transform(valid_emb_scaled)

    train_x = np.hstack([train_logits, train_emb_pca]).astype(np.float32)
    valid_x = np.hstack([valid_logits, valid_emb_pca]).astype(np.float32)

    feature_scaler = StandardScaler()
    return (
        feature_scaler.fit_transform(train_x).astype(np.float32),
        feature_scaler.transform(valid_x).astype(np.float32),
    )


def default_experiments() -> tuple[ExperimentSpec, ...]:
    return (
        ExperimentSpec(
            name="full59_raw_perch_filegkf5",
            dataset="full59",
            splitter="file_group_kfold",
            n_splits=5,
            feature_mode="raw_perch",
        ),
        ExperimentSpec(
            name="full59_probe_pca_blend_filegkf5",
            dataset="full59",
            splitter="file_group_kfold",
            n_splits=5,
            feature_mode="probe_pca_blend",
        ),
        ExperimentSpec(
            name="all66_raw_perch_filegkf5",
            dataset="all66",
            splitter="file_group_kfold",
            n_splits=5,
            feature_mode="raw_perch",
        ),
        ExperimentSpec(
            name="all66_probe_pca_blend_filegkf5",
            dataset="all66",
            splitter="file_group_kfold",
            n_splits=5,
            feature_mode="probe_pca_blend",
        ),
        ExperimentSpec(
            name="all66_probe_pca_blend_sitebalanced3",
            dataset="all66",
            splitter="site_balanced_file",
            n_splits=3,
            feature_mode="probe_pca_blend",
        ),
    )


def evaluate_experiment(
    spec: ExperimentSpec,
    rows: list[SoundscapeWindow],
    logits: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> ExperimentResult:
    folds = folds_for_experiment(rows, spec.splitter, spec.n_splits)

    if spec.feature_mode == "raw_perch":
        oof_scores = logits.copy()
        fold_results = []
        for fold_id, (_, val_indices) in enumerate(folds, start=1):
            fold_auc, active_classes = macro_auc(labels[val_indices], oof_scores[val_indices])
            fold_results.append(
                FoldResult(
                    fold_id=fold_id,
                    auc=fold_auc,
                    active_classes=active_classes,
                    windows=len(val_indices),
                    files=len({rows[index].filename for index in val_indices}),
                    sites=tuple(sorted({rows[index].site for index in val_indices})),
                )
            )
    else:
        oof_scores = np.zeros_like(logits, dtype=np.float32)
        fold_results = []
        for fold_id, (train_indices, val_indices) in enumerate(folds, start=1):
            train_x, valid_x = build_features_for_fold(
                feature_mode=spec.feature_mode,
                train_logits=logits[train_indices],
                valid_logits=logits[val_indices],
                train_embeddings=embeddings[train_indices],
                valid_embeddings=embeddings[val_indices],
            )
            raw_valid_scores = logits[val_indices]
            valid_scores = raw_valid_scores.copy()
            for class_idx in range(labels.shape[1]):
                y_train = labels[train_indices, class_idx]
                positives = int(y_train.sum())
                if positives < PROBE_MIN_POS or positives == len(y_train):
                    continue

                clf = LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=300,
                    C=2.0,
                    random_state=42,
                )
                clf.fit(train_x, y_train)
                prob = clf.predict_proba(valid_x)[:, 1].astype(np.float32)
                valid_scores[:, class_idx] = (1.0 - PROBE_BLEND_ALPHA) * raw_valid_scores[
                    :, class_idx
                ] + PROBE_BLEND_ALPHA * logit_from_prob(prob)
            oof_scores[val_indices] = valid_scores
            fold_auc, active_classes = macro_auc(labels[val_indices], valid_scores)
            fold_results.append(
                FoldResult(
                    fold_id=fold_id,
                    auc=fold_auc,
                    active_classes=active_classes,
                    windows=len(val_indices),
                    files=len({rows[index].filename for index in val_indices}),
                    sites=tuple(sorted({rows[index].site for index in val_indices})),
                )
            )

    overall_auc, active_classes = macro_auc(labels, oof_scores)
    fold_aucs = [fold.auc for fold in fold_results]
    fold_class_counts = [fold.active_classes for fold in fold_results]
    return ExperimentResult(
        name=spec.name,
        dataset=spec.dataset,
        splitter=spec.splitter,
        n_splits=spec.n_splits,
        feature_mode=spec.feature_mode,
        windows=len(rows),
        files=len({row.filename for row in rows}),
        sites=len({row.site for row in rows}),
        active_classes=active_classes,
        oof_macro_auc=overall_auc,
        mean_fold_auc=float(np.mean(fold_aucs)),
        min_fold_auc=float(np.min(fold_aucs)),
        max_fold_auc=float(np.max(fold_aucs)),
        min_fold_active_classes=min(fold_class_counts),
        max_fold_active_classes=max(fold_class_counts),
        folds=tuple(fold_results),
    )


def format_summary(results: tuple[ExperimentResult, ...]) -> str:
    lines = ["# Perch probe CV", ""]
    for result in results:
        lines.append(
            "- "
            f"{result.name}: oof_macro_auc={result.oof_macro_auc:.6f} "
            f"mean_fold_auc={result.mean_fold_auc:.6f} "
            f"fold_auc_range=[{result.min_fold_auc:.6f}, {result.max_fold_auc:.6f}] "
            "fold_active_classes="
            f"[{result.min_fold_active_classes}, {result.max_fold_active_classes}]"
        )
        for fold in result.folds:
            lines.append(
                "  "
                f"fold={fold.fold_id} auc={fold.auc:.6f} "
                f"classes={fold.active_classes} windows={fold.windows} "
                f"files={fold.files} sites={list(fold.sites)}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    primary_labels = load_primary_labels(args.data_dir)
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}

    all_rows = load_soundscape_rows(args.data_dir)
    full_rows = filter_fully_labeled_rows(all_rows)
    extracted = extract_perch_features(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_path=args.cache_path,
        force_rebuild=args.force_rebuild_cache,
    )

    results = []
    for spec in default_experiments():
        selected_rows = full_rows if spec.dataset == "full59" else all_rows
        aligned_rows, logits, embeddings, labels = align_rows_and_features(
            rows=selected_rows,
            features=extracted,
            label_to_idx=label_to_idx,
        )
        results.append(
            evaluate_experiment(
                spec=spec,
                rows=aligned_rows,
                logits=logits,
                embeddings=embeddings,
                labels=labels,
            )
        )

    results_tuple = tuple(results)
    print(format_summary(results_tuple))

    if args.json_out is not None:
        payload = {"results": [asdict(result) for result in results_tuple]}
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON saved to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
