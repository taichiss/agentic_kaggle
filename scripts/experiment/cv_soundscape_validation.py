from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "input" / "BirdCLEF+ 2026"
WINDOWS_PER_FILE = 12
SITE_PRIOR_SHRINK = 8.0
FILENAME_PATTERN = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


@dataclass(frozen=True)
class SoundscapeWindow:
    filename: str
    start: str
    end: str
    end_sec: int
    site: str
    hour_utc: int
    labels: tuple[str, ...]


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
    use_site_prior: bool
    use_hour_prior: bool
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


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: str
    splitter: str
    n_splits: int
    use_site_prior: bool
    use_hour_prior: bool


def parse_filename(filename: str) -> tuple[str, int]:
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return "unknown", -1
    _, site, _, hms = match.groups()
    return site, int(hms[:2])


def parse_hms_to_seconds(value: str) -> int:
    hours_str, minutes_str, seconds_str = value.split(":")
    return int(hours_str) * 3600 + int(minutes_str) * 60 + int(seconds_str)


def load_primary_labels(data_dir: Path) -> list[str]:
    with (data_dir / "sample_submission.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return header[1:]


def load_soundscape_rows(data_dir: Path) -> list[SoundscapeWindow]:
    grouped_labels: dict[tuple[str, str, str], set[str]] = {}
    labels_path = data_dir / "train_soundscapes_labels.csv"
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["filename"], row["start"], row["end"])
            labels = grouped_labels.setdefault(key, set())
            for token in row["primary_label"].split(";"):
                cleaned = token.strip()
                if cleaned:
                    labels.add(cleaned)

    rows: list[SoundscapeWindow] = []
    for (filename, start, end), label_set in grouped_labels.items():
        site, hour_utc = parse_filename(filename)
        rows.append(
            SoundscapeWindow(
                filename=filename,
                start=start,
                end=end,
                end_sec=parse_hms_to_seconds(end),
                site=site,
                hour_utc=hour_utc,
                labels=tuple(sorted(label_set)),
            )
        )
    return sorted(rows, key=lambda row: (row.filename, row.end_sec))


def filter_fully_labeled_rows(rows: list[SoundscapeWindow]) -> list[SoundscapeWindow]:
    counts = Counter(row.filename for row in rows)
    return [row for row in rows if counts[row.filename] == WINDOWS_PER_FILE]


def active_class_count(rows: list[SoundscapeWindow]) -> int:
    labels = {label for row in rows for label in row.labels}
    return len(labels)


def build_label_matrix(
    rows: list[SoundscapeWindow],
    label_to_idx: dict[str, int],
    n_classes: int,
) -> list[list[int]]:
    matrix: list[list[int]] = []
    for row in rows:
        target = [0] * n_classes
        for label in row.labels:
            idx = label_to_idx.get(label)
            if idx is not None:
                target[idx] = 1
        matrix.append(target)
    return matrix


def auc_binary(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(y_score, y_true, strict=True), key=lambda item: item[0])
    rank = 1
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        block_end = index + 1
        while block_end < len(ranked) and ranked[block_end][0] == ranked[index][0]:
            block_end += 1
        average_rank = (rank + rank + (block_end - index) - 1) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ranked[index:block_end])
        rank += block_end - index
        index = block_end

    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def macro_auc(y_true: list[list[int]], y_score: list[list[float]]) -> tuple[float, int]:
    aucs: list[float] = []
    n_classes = len(y_true[0])
    for class_idx in range(n_classes):
        class_auc = auc_binary(
            [row[class_idx] for row in y_true],
            [row[class_idx] for row in y_score],
        )
        if class_auc is not None:
            aucs.append(class_auc)
    return sum(aucs) / len(aucs), len(aucs)


def group_kfold(
    rows: list[SoundscapeWindow],
    key_fn: Callable[[SoundscapeWindow], str],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group_to_indices[key_fn(row)].append(index)

    items = sorted(group_to_indices.items(), key=lambda item: (-len(item[1]), item[0]))
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for _, indices in items:
        best_fold = min(range(n_splits), key=lambda fold_id: (fold_sizes[fold_id], fold_id))
        fold_indices[best_fold].extend(indices)
        fold_sizes[best_fold] += len(indices)

    universe = set(range(len(rows)))
    folds: list[tuple[list[int], list[int]]] = []
    for val_indices in fold_indices:
        val_indices_sorted = sorted(val_indices)
        train_indices_sorted = sorted(universe - set(val_indices_sorted))
        folds.append((train_indices_sorted, val_indices_sorted))
    return folds


def site_balanced_file_folds(
    rows: list[SoundscapeWindow],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    file_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        file_to_indices[row.filename].append(index)

    file_groups = []
    for filename, indices in file_to_indices.items():
        site = rows[indices[0]].site
        file_groups.append((filename, site, len(indices)))

    site_totals = Counter(site for _, site, _ in file_groups)
    target_windows_per_fold = sum(size for _, _, size in file_groups) / n_splits
    target_site_windows = {site: count / n_splits for site, count in site_totals.items()}

    fold_files: list[set[str]] = [set() for _ in range(n_splits)]
    fold_windows = [0] * n_splits
    fold_site_windows: list[Counter[str]] = [Counter() for _ in range(n_splits)]
    file_groups.sort(key=lambda item: (-item[2], item[1], item[0]))

    for filename, site, size in file_groups:
        best_fold = min(
            range(n_splits),
            key=lambda fold_id: (
                abs((fold_site_windows[fold_id][site] + size) - target_site_windows[site]),
                abs((fold_windows[fold_id] + size) - target_windows_per_fold),
                fold_windows[fold_id],
                fold_id,
            ),
        )
        fold_files[best_fold].add(filename)
        fold_windows[best_fold] += size
        fold_site_windows[best_fold][site] += size

    universe = set(range(len(rows)))
    folds: list[tuple[list[int], list[int]]] = []
    for filenames in fold_files:
        val_indices = sorted(index for index, row in enumerate(rows) if row.filename in filenames)
        train_indices = sorted(universe - set(val_indices))
        folds.append((train_indices, val_indices))
    return folds


def build_fold_safe_prior_predictions(
    train_rows: list[SoundscapeWindow],
    val_rows: list[SoundscapeWindow],
    label_to_idx: dict[str, int],
    n_classes: int,
    use_site_prior: bool,
    use_hour_prior: bool,
) -> list[list[float]]:
    global_counts = [0.0] * n_classes
    site_counts: dict[str, list[float]] = defaultdict(lambda: [0.0] * n_classes)
    hour_counts: dict[int, list[float]] = defaultdict(lambda: [0.0] * n_classes)
    site_sizes: Counter[str] = Counter()
    hour_sizes: Counter[int] = Counter()

    for row in train_rows:
        site_sizes[row.site] += 1
        hour_sizes[row.hour_utc] += 1
        for label in row.labels:
            idx = label_to_idx[label]
            global_counts[idx] += 1.0
            site_counts[row.site][idx] += 1.0
            hour_counts[row.hour_utc][idx] += 1.0

    global_prior = [count / len(train_rows) for count in global_counts]
    predictions: list[list[float]] = []
    for row in val_rows:
        probs = list(global_prior)
        if use_hour_prior and row.hour_utc in hour_sizes:
            hour_total = hour_sizes[row.hour_utc]
            hour_weight = hour_total / (hour_total + SITE_PRIOR_SHRINK)
            hour_prior = [count / hour_total for count in hour_counts[row.hour_utc]]
            probs = [
                hour_weight * hour_prob + (1.0 - hour_weight) * global_prob
                for hour_prob, global_prob in zip(hour_prior, global_prior, strict=True)
            ]
        if use_site_prior and row.site in site_sizes:
            site_total = site_sizes[row.site]
            site_weight = site_total / (site_total + SITE_PRIOR_SHRINK)
            site_prior = [count / site_total for count in site_counts[row.site]]
            probs = [
                site_weight * site_prob + (1.0 - site_weight) * current_prob
                for site_prob, current_prob in zip(site_prior, probs, strict=True)
            ]
        predictions.append(probs)
    return predictions


def default_experiments() -> tuple[ExperimentSpec, ...]:
    return (
        ExperimentSpec(
            name="notebook_full59_filegkf5_sitehour",
            dataset="full59",
            splitter="file_group_kfold",
            n_splits=5,
            use_site_prior=True,
            use_hour_prior=True,
        ),
        ExperimentSpec(
            name="all66_filegkf5_sitehour",
            dataset="all66",
            splitter="file_group_kfold",
            n_splits=5,
            use_site_prior=True,
            use_hour_prior=True,
        ),
        ExperimentSpec(
            name="all66_filegkf5_houronly",
            dataset="all66",
            splitter="file_group_kfold",
            n_splits=5,
            use_site_prior=False,
            use_hour_prior=True,
        ),
        ExperimentSpec(
            name="all66_siteholdout3_sitehour",
            dataset="all66",
            splitter="site_holdout",
            n_splits=3,
            use_site_prior=True,
            use_hour_prior=True,
        ),
        ExperimentSpec(
            name="all66_sitebalanced_file3_sitehour",
            dataset="all66",
            splitter="site_balanced_file",
            n_splits=3,
            use_site_prior=True,
            use_hour_prior=True,
        ),
    )


def rows_for_dataset(
    all_rows: list[SoundscapeWindow],
    full_rows: list[SoundscapeWindow],
    dataset: str,
) -> list[SoundscapeWindow]:
    if dataset == "all66":
        return all_rows
    if dataset == "full59":
        return full_rows
    raise ValueError(f"unknown dataset: {dataset}")


def folds_for_experiment(
    rows: list[SoundscapeWindow],
    splitter: str,
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    if splitter == "file_group_kfold":
        return group_kfold(rows, key_fn=lambda row: row.filename, n_splits=n_splits)
    if splitter == "site_holdout":
        return group_kfold(rows, key_fn=lambda row: row.site, n_splits=n_splits)
    if splitter == "site_balanced_file":
        return site_balanced_file_folds(rows, n_splits=n_splits)
    raise ValueError(f"unknown splitter: {splitter}")


def evaluate_experiment(
    spec: ExperimentSpec,
    rows: list[SoundscapeWindow],
    label_to_idx: dict[str, int],
    n_classes: int,
) -> ExperimentResult:
    folds = folds_for_experiment(rows, spec.splitter, spec.n_splits)
    y_true = build_label_matrix(rows, label_to_idx, n_classes)
    oof_scores = [[0.0] * n_classes for _ in rows]
    fold_results: list[FoldResult] = []

    for fold_id, (train_indices, val_indices) in enumerate(folds, start=1):
        train_rows = [rows[index] for index in train_indices]
        val_rows = [rows[index] for index in val_indices]
        predictions = build_fold_safe_prior_predictions(
            train_rows=train_rows,
            val_rows=val_rows,
            label_to_idx=label_to_idx,
            n_classes=n_classes,
            use_site_prior=spec.use_site_prior,
            use_hour_prior=spec.use_hour_prior,
        )
        for row_index, score in zip(val_indices, predictions, strict=True):
            oof_scores[row_index] = score

        fold_auc, fold_active_classes = macro_auc(
            [y_true[index] for index in val_indices],
            predictions,
        )
        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                auc=fold_auc,
                active_classes=fold_active_classes,
                windows=len(val_indices),
                files=len({rows[index].filename for index in val_indices}),
                sites=tuple(sorted({rows[index].site for index in val_indices})),
            )
        )

    oof_macro_auc, active_classes = macro_auc(y_true, oof_scores)
    fold_aucs = [fold.auc for fold in fold_results]
    fold_active_class_counts = [fold.active_classes for fold in fold_results]
    return ExperimentResult(
        name=spec.name,
        dataset=spec.dataset,
        splitter=spec.splitter,
        n_splits=spec.n_splits,
        use_site_prior=spec.use_site_prior,
        use_hour_prior=spec.use_hour_prior,
        windows=len(rows),
        files=len({row.filename for row in rows}),
        sites=len({row.site for row in rows}),
        active_classes=active_classes,
        oof_macro_auc=oof_macro_auc,
        mean_fold_auc=sum(fold_aucs) / len(fold_aucs),
        min_fold_auc=min(fold_aucs),
        max_fold_auc=max(fold_aucs),
        min_fold_active_classes=min(fold_active_class_counts),
        max_fold_active_classes=max(fold_active_class_counts),
        folds=tuple(fold_results),
    )


def run_all_experiments(
    data_dir: Path,
) -> tuple[dict[str, dict[str, int]], tuple[ExperimentResult, ...]]:
    primary_labels = load_primary_labels(data_dir)
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}
    all_rows = load_soundscape_rows(data_dir)
    full_rows = filter_fully_labeled_rows(all_rows)

    dataset_summary: dict[str, dict[str, int]] = {
        "all66": {
            "windows": len(all_rows),
            "files": len({row.filename for row in all_rows}),
            "sites": len({row.site for row in all_rows}),
            "active_classes": active_class_count(all_rows),
        },
        "full59": {
            "windows": len(full_rows),
            "files": len({row.filename for row in full_rows}),
            "sites": len({row.site for row in full_rows}),
            "active_classes": active_class_count(full_rows),
        },
    }

    results = []
    for spec in default_experiments():
        selected_rows = rows_for_dataset(all_rows, full_rows, spec.dataset)
        results.append(
            evaluate_experiment(
                spec=spec,
                rows=selected_rows,
                label_to_idx=label_to_idx,
                n_classes=len(primary_labels),
            )
        )
    return dataset_summary, tuple(results)


def format_summary(
    dataset_summary: dict[str, dict[str, int]],
    results: tuple[ExperimentResult, ...],
) -> str:
    lines = ["# CV soundscape validation", ""]
    lines.append("## datasets")
    for dataset_name, summary in dataset_summary.items():
        lines.append(
            "- "
            f"{dataset_name}: windows={summary['windows']} "
            f"files={summary['files']} sites={summary['sites']} "
            f"active_classes={summary['active_classes']}"
        )
    lines.append("")
    lines.append("## experiments")
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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="path to BirdCLEF+ 2026 data directory",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional path to write full results as JSON",
    )
    args = parser.parse_args()

    dataset_summary, results = run_all_experiments(args.data_dir)
    print(format_summary(dataset_summary, results))

    if args.json_out is not None:
        payload = {
            "dataset_summary": dataset_summary,
            "results": [asdict(result) for result in results],
        }
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON saved to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
