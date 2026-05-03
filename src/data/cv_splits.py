"""Fold helpers for soundscape file-grouped validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ClipFoldRecord:
    filename: str
    site: str


def _group_kfold(
    records: list[ClipFoldRecord],
    *,
    key_fn: Callable[[ClipFoldRecord], str],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        group_to_indices[key_fn(record)].append(index)

    items = sorted(group_to_indices.items(), key=lambda item: (-len(item[1]), item[0]))
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for _, indices in items:
        best_fold = min(range(n_splits), key=lambda fold_id: (fold_sizes[fold_id], fold_id))
        fold_indices[best_fold].extend(indices)
        fold_sizes[best_fold] += len(indices)

    universe = set(range(len(records)))
    return [
        (sorted(universe - set(val_indices)), sorted(val_indices)) for val_indices in fold_indices
    ]


def file_group_kfold(
    records: list[ClipFoldRecord],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    return _group_kfold(records, key_fn=lambda record: record.filename, n_splits=n_splits)


def site_holdout_folds(
    records: list[ClipFoldRecord],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    return _group_kfold(records, key_fn=lambda record: record.site, n_splits=n_splits)


def site_balanced_file_folds(
    records: list[ClipFoldRecord],
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    file_to_indices: dict[str, list[int]] = defaultdict(list)
    file_to_site: dict[str, str] = {}
    for index, record in enumerate(records):
        file_to_indices[record.filename].append(index)
        file_to_site[record.filename] = record.site

    file_groups = [
        (filename, file_to_site[filename], len(indices))
        for filename, indices in file_to_indices.items()
    ]
    site_totals = Counter(site for _, site, _ in file_groups)
    target_items_per_fold = sum(size for _, _, size in file_groups) / n_splits
    target_site_items = {site: count / n_splits for site, count in site_totals.items()}

    fold_files: list[set[str]] = [set() for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    fold_site_sizes: list[Counter[str]] = [Counter() for _ in range(n_splits)]
    file_groups.sort(key=lambda item: (-item[2], item[1], item[0]))

    for filename, site, size in file_groups:
        best_fold = min(
            range(n_splits),
            key=lambda fold_id: (
                abs((fold_site_sizes[fold_id][site] + size) - target_site_items[site]),
                abs((fold_sizes[fold_id] + size) - target_items_per_fold),
                fold_sizes[fold_id],
                fold_id,
            ),
        )
        fold_files[best_fold].add(filename)
        fold_sizes[best_fold] += size
        fold_site_sizes[best_fold][site] += size

    universe = set(range(len(records)))
    folds: list[tuple[list[int], list[int]]] = []
    for filenames in fold_files:
        val_indices = sorted(
            index for index, record in enumerate(records) if record.filename in filenames
        )
        folds.append((sorted(universe - set(val_indices)), val_indices))
    return folds
