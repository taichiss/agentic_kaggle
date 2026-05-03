# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.experiment.cv_soundscape_validation import (
    DEFAULT_DATA_DIR,
    SoundscapeWindow,
    folds_for_experiment,
    load_primary_labels,
    load_soundscape_rows,
)
from scripts.experiment.perch_probe_cv import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MODEL_DIR,
    EMBED_PCA_DIM,
    PROBE_BLEND_ALPHA,
    PROBE_MIN_POS,
    ExperimentResult,
    FoldResult,
    WindowFeatures,
    align_rows_and_features,
    extract_perch_features,
    logit_from_prob,
    macro_auc,
)

OUTPUT_ROOT = REPO_ROOT / "output" / "models"
KAGGLE_DATASET_SLUG = "birdclef-2026-perch-probe-cv-models"
ProbeBundle: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class PatternSpec:
    name: str
    dataset: str
    splitter: str
    n_splits: int
    feature_mode: str = "probe_pca_blend"


def default_patterns() -> tuple[PatternSpec, ...]:
    return (
        PatternSpec(
            name="main_all66_sitebalanced3",
            dataset="all66",
            splitter="site_balanced_file",
            n_splits=3,
        ),
        PatternSpec(
            name="secondary_all66_filegkf5",
            dataset="all66",
            splitter="file_group_kfold",
            n_splits=5,
        ),
        PatternSpec(
            name="stress_all66_siteholdout3",
            dataset="all66",
            splitter="site_holdout",
            n_splits=3,
        ),
    )


def rows_for_dataset(
    dataset: str,
    all_rows: list[SoundscapeWindow],
) -> list[SoundscapeWindow]:
    if dataset != "all66":
        raise ValueError(f"unsupported dataset: {dataset}")
    return all_rows


def build_full59_cache(
    output_dir: Path,
    all_rows: list[SoundscapeWindow],
    features: list[WindowFeatures],
    label_to_idx: dict[str, int],
    primary_labels: list[str],
) -> None:
    feature_by_row_id = {feature.row_id: feature for feature in features}
    file_counts: dict[str, int] = {}
    for row in all_rows:
        file_counts[row.filename] = file_counts.get(row.filename, 0) + 1
    full_rows = [row for row in all_rows if file_counts[row.filename] == 12]

    meta_records = []
    logits = []
    embeddings = []
    for row in full_rows:
        row_id = f"{row.filename.removesuffix('.ogg')}_{row.end_sec}"
        feature = feature_by_row_id[row_id]
        meta_records.append(
            {
                "row_id": row_id,
                "filename": row.filename,
                "site": row.site,
                "hour_utc": row.hour_utc,
                "end_sec": row.end_sec,
            }
        )
        logits.append(feature.logits.astype(np.float32))
        embeddings.append(feature.embedding.astype(np.float32))

    pd.DataFrame(meta_records).to_parquet(output_dir / "perch_meta.parquet", index=False)
    np.savez_compressed(
        output_dir / "perch_arrays.npz",
        scores=np.stack(logits).astype(np.float32),
        embs=np.stack(embeddings).astype(np.float32),
        primary_labels=np.array(primary_labels, dtype=str),
    )


def fit_probe_bundle(
    train_logits: np.ndarray,
    train_embeddings: np.ndarray,
    train_y: np.ndarray,
) -> ProbeBundle:
    emb_scaler = StandardScaler()
    train_emb_scaled = emb_scaler.fit_transform(train_embeddings)

    pca_dim = min(EMBED_PCA_DIM, train_emb_scaled.shape[0] - 1, train_emb_scaled.shape[1])
    if pca_dim <= 0:
        pca = None
        train_emb_pca = train_emb_scaled
    else:
        pca = PCA(n_components=pca_dim, random_state=42)
        train_emb_pca = pca.fit_transform(train_emb_scaled)

    train_x_raw = np.hstack([train_logits, train_emb_pca]).astype(np.float32)
    feature_scaler = StandardScaler()
    train_x = feature_scaler.fit_transform(train_x_raw).astype(np.float32)

    classifiers: dict[int, LogisticRegression] = {}
    trained_classes: list[int] = []
    for class_idx in range(train_y.shape[1]):
        y = train_y[:, class_idx]
        positives = int(y.sum())
        if positives < PROBE_MIN_POS or positives == len(y):
            continue
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=300,
            C=2.0,
            random_state=42,
        )
        clf.fit(train_x, y)
        classifiers[class_idx] = clf
        trained_classes.append(class_idx)

    return {
        "feature_mode": "probe_pca_blend",
        "blend_alpha": PROBE_BLEND_ALPHA,
        "min_pos": PROBE_MIN_POS,
        "emb_scaler": emb_scaler,
        "pca": pca,
        "feature_scaler": feature_scaler,
        "classifiers": classifiers,
        "trained_classes": trained_classes,
    }


def transform_probe_features(
    bundle: ProbeBundle,
    logits: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    emb_scaler = bundle["emb_scaler"]
    pca = bundle["pca"]
    feature_scaler = bundle["feature_scaler"]

    emb_scaled = emb_scaler.transform(embeddings)
    if pca is None:
        emb_pca = emb_scaled
    else:
        emb_pca = pca.transform(emb_scaled)
    x_raw = np.hstack([logits, emb_pca]).astype(np.float32)
    return feature_scaler.transform(x_raw).astype(np.float32)


def predict_probe_bundle(
    bundle: ProbeBundle,
    raw_logits: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    x = transform_probe_features(bundle, raw_logits, embeddings)
    scores = raw_logits.copy()
    classifiers: dict[int, LogisticRegression] = bundle["classifiers"]

    for class_idx, clf in classifiers.items():
        prob = clf.predict_proba(x)[:, 1].astype(np.float32)
        scores[:, class_idx] = (1.0 - PROBE_BLEND_ALPHA) * raw_logits[
            :, class_idx
        ] + PROBE_BLEND_ALPHA * logit_from_prob(prob)
    return scores.astype(np.float32)


def fold_summary(
    fold_id: int,
    rows: list[SoundscapeWindow],
    val_indices: list[int],
    labels: np.ndarray,
    scores: np.ndarray,
) -> FoldResult:
    auc, active_classes = macro_auc(labels[val_indices], scores)
    return FoldResult(
        fold_id=fold_id,
        auc=auc,
        active_classes=active_classes,
        windows=len(val_indices),
        files=len({rows[index].filename for index in val_indices}),
        sites=tuple(sorted({rows[index].site for index in val_indices})),
    )


def save_fold_bundle(
    fold_dir: Path,
    bundle: ProbeBundle,
    train_indices: list[int],
    val_indices: list[int],
    rows: list[SoundscapeWindow],
    fold_result: FoldResult,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, fold_dir / "probe_bundle.joblib", compress=3)
    payload = {
        "fold_result": asdict(fold_result),
        "train_files": sorted({rows[index].filename for index in train_indices}),
        "val_files": sorted({rows[index].filename for index in val_indices}),
        "train_sites": sorted({rows[index].site for index in train_indices}),
        "val_sites": sorted({rows[index].site for index in val_indices}),
        "train_row_ids": [
            f"{rows[index].filename.removesuffix('.ogg')}_{rows[index].end_sec}"
            for index in train_indices
        ],
        "val_row_ids": [
            f"{rows[index].filename.removesuffix('.ogg')}_{rows[index].end_sec}"
            for index in val_indices
        ],
    }
    (fold_dir / "fold_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_pattern(
    output_dir: Path,
    pattern: PatternSpec,
    rows: list[SoundscapeWindow],
    logits: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    primary_labels: list[str],
) -> ExperimentResult:
    folds = folds_for_experiment(rows, splitter=pattern.splitter, n_splits=pattern.n_splits)
    pattern_dir = output_dir / "cv_models" / pattern.name
    pattern_dir.mkdir(parents=True, exist_ok=True)

    oof_scores = np.zeros_like(logits, dtype=np.float32)
    fold_results: list[FoldResult] = []

    for fold_id, (train_indices, val_indices) in enumerate(folds, start=1):
        bundle = fit_probe_bundle(
            train_logits=logits[train_indices],
            train_embeddings=embeddings[train_indices],
            train_y=labels[train_indices],
        )
        val_scores = predict_probe_bundle(
            bundle=bundle,
            raw_logits=logits[val_indices],
            embeddings=embeddings[val_indices],
        )
        oof_scores[val_indices] = val_scores
        result = fold_summary(fold_id, rows, val_indices, labels, val_scores)
        fold_results.append(result)
        save_fold_bundle(
            fold_dir=pattern_dir / f"fold_{fold_id:02d}",
            bundle=bundle,
            train_indices=train_indices,
            val_indices=val_indices,
            rows=rows,
            fold_result=result,
        )

    overall_auc, active_classes = macro_auc(labels, oof_scores)
    fold_aucs = [fold.auc for fold in fold_results]
    fold_class_counts = [fold.active_classes for fold in fold_results]
    experiment_result = ExperimentResult(
        name=pattern.name,
        dataset=pattern.dataset,
        splitter=pattern.splitter,
        n_splits=pattern.n_splits,
        feature_mode=pattern.feature_mode,
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

    full_bundle = fit_probe_bundle(
        train_logits=logits,
        train_embeddings=embeddings,
        train_y=labels,
    )
    full_fit_dir = pattern_dir / "full_fit"
    full_fit_dir.mkdir(exist_ok=True)
    joblib.dump(full_bundle, full_fit_dir / "probe_bundle.joblib", compress=3)
    (full_fit_dir / "full_fit_info.json").write_text(
        json.dumps(
            {
                "pattern": pattern.name,
                "dataset": pattern.dataset,
                "rows": len(rows),
                "files": len({row.filename for row in rows}),
                "sites": sorted({row.site for row in rows}),
                "primary_labels": primary_labels,
                "trained_classes": full_bundle["trained_classes"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        pattern_dir / "oof_outputs.npz",
        row_ids=np.array(
            [f"{row.filename.removesuffix('.ogg')}_{row.end_sec}" for row in rows],
            dtype=str,
        ),
        scores=oof_scores.astype(np.float32),
        labels=labels.astype(np.uint8),
        primary_labels=np.array(primary_labels, dtype=str),
    )
    (pattern_dir / "manifest.json").write_text(
        json.dumps(asdict(experiment_result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return experiment_result


def write_dataset_metadata(output_dir: Path) -> Path:
    metadata_path = output_dir / "dataset-metadata.json"
    metadata = {
        "title": "BirdCLEF 2026 Perch Probe CV Models",
        "id": f"suzukitaichi/{KAGGLE_DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Perch probe CV bundles and notebook-compatible cache",
        "description": (
            "Notebook-compatible Perch cache plus trained probe bundles for "
            "site-balanced, file-GroupKFold, and site-holdout validation."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def write_readme(output_dir: Path, results: list[ExperimentResult]) -> None:
    lines = [
        "# BirdCLEF 2026 Perch Probe CV Models",
        "",
        "## Notebook compatibility",
        "- `perch_meta.parquet` and `perch_arrays.npz` are saved at the dataset root.",
        "- These root files contain the notebook-compatible `full59` cache.",
        "- CV model bundles live under `cv_models/<pattern>/`.",
        "",
        "## Patterns",
    ]
    for result in results:
        lines.append(
            f"- `{result.name}`: oof_macro_auc={result.oof_macro_auc:.6f}, "
            f"splitter={result.splitter}, n_splits={result.n_splits}"
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    clean_output_dir(args.output_dir)

    primary_labels = load_primary_labels(args.data_dir)
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}
    all_rows = load_soundscape_rows(args.data_dir)
    features = extract_perch_features(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_path=args.cache_path,
        force_rebuild=args.force_rebuild_cache,
    )

    write_dataset_metadata(args.output_dir)
    build_full59_cache(args.output_dir, all_rows, features, label_to_idx, primary_labels)

    results: list[ExperimentResult] = []
    for pattern in default_patterns():
        selected_rows = rows_for_dataset(pattern.dataset, all_rows)
        aligned_rows, logits, embeddings, labels = align_rows_and_features(
            rows=selected_rows,
            features=features,
            label_to_idx=label_to_idx,
        )
        results.append(
            train_pattern(
                output_dir=args.output_dir,
                pattern=pattern,
                rows=aligned_rows,
                logits=logits,
                embeddings=embeddings,
                labels=labels,
                primary_labels=primary_labels,
            )
        )

    summary = {
        "dataset_slug": KAGGLE_DATASET_SLUG,
        "results": [asdict(result) for result in results],
    }
    (args.output_dir / "model_index.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(args.output_dir, results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
