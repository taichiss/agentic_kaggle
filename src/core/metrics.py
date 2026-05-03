"""Evaluation metrics for BirdCLEF.

Primary metric: macro ROC-AUC (excluding classes with no true positives).
Auxiliary: per-class AUC, per-taxon AUC.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


def macro_roc_auc(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    class_names: Sequence[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute competition-style macro ROC-AUC.

    Classes with no true positives are excluded from the average,
    matching the Kaggle evaluation metric.

    Parameters
    ----------
    y_true : (N, C) binary matrix
    y_pred : (N, C) probability matrix
    class_names : optional list of C class names

    Returns
    -------
    macro_auc : float
    per_class : dict mapping class name → AUC (NaN if excluded)
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    per_class: dict[str, float] = {}
    valid_aucs: list[float] = []

    for i, name in enumerate(class_names):
        col_true = y_true[:, i]
        if col_true.sum() == 0:
            per_class[name] = float("nan")
            continue
        if col_true.sum() == len(col_true):
            # All positive — AUC is undefined but not penalised
            per_class[name] = float("nan")
            continue
        auc = float(roc_auc_score(col_true, y_pred[:, i]))
        per_class[name] = auc
        valid_aucs.append(auc)

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return macro_auc, per_class


# ---------------------------------------------------------------------------
# Taxon-level summary
# ---------------------------------------------------------------------------

# 2026 taxonomy groups
TAXON_GROUPS: dict[str, str] = {}  # populated by load_taxonomy()


def load_taxonomy(taxonomy_csv: str) -> dict[str, str]:
    """Load taxonomy.csv and return {species: taxon_group} mapping."""
    global TAXON_GROUPS  # noqa: PLW0603
    df = pd.read_csv(taxonomy_csv)
    mapping = dict(
        zip(
            df["primary_label"],
            df["class_name"].str.split(" - ").str[0],
            strict=True,
        )
    )
    # Fallback: use 'class' column if present
    if "class" in df.columns:
        mapping = dict(zip(df["primary_label"], df["class"], strict=True))
    TAXON_GROUPS.update(mapping)
    return mapping


def per_taxon_auc(
    per_class_auc: dict[str, float],
    taxon_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Aggregate per-class AUCs by taxon group.

    Parameters
    ----------
    per_class_auc : {species: auc} from macro_roc_auc()
    taxon_map : {species: taxon_group}

    Returns
    -------
    {taxon_group: mean AUC over valid species}
    """
    if taxon_map is None:
        taxon_map = TAXON_GROUPS

    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    for species, auc in per_class_auc.items():
        if np.isnan(auc):
            continue
        group = taxon_map.get(species, "Unknown")
        buckets[group].append(auc)

    return {group: float(np.mean(aucs)) for group, aucs in sorted(buckets.items())}
