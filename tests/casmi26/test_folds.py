from __future__ import annotations

from pathlib import Path

import pytest

FOLD_DATASET_PATH = Path("data/input/external/casmi26/casmi26-identity-disjoint-validation-folds")

pytestmark = pytest.mark.skipif(
    not (FOLD_DATASET_PATH / "structures.parquet").exists(),
    reason="identity-disjoint fold dataset not downloaded",
)


def test_fold_sizes_match_spec():
    """SUMMARY.json の fold サイズ仕様 (55512-55514) を実データで確認。"""
    from casmi26.data import load_identity_folds

    df = load_identity_folds()
    assert len(df) == 277566
    for col in ("fold_identity", "fold_scaffold"):
        sizes = df.groupby(col).size().to_dict()
        assert set(sizes.keys()) == {0, 1, 2, 3, 4}
        for v in sizes.values():
            assert 55512 <= v <= 55514


def test_fold_split_covers_all_structures():
    from casmi26.data import load_identity_folds
    from casmi26.data.folds import split_fold

    df = load_identity_folds()
    a = split_fold(df, fold=0, kind="identity")
    assert a.val_structure_count + a.train_structure_count == len(df)
    assert not (a.val_inchikey14 & a.train_inchikey14), "identity fold must be disjoint"


def test_scaffold_fold_split_is_disjoint():
    from casmi26.data import load_identity_folds
    from casmi26.data.folds import split_fold

    df = load_identity_folds()
    a = split_fold(df, fold=0, kind="scaffold")
    assert not (a.val_inchikey14 & a.train_inchikey14)


def test_split_summary_has_expected_axes():
    from casmi26.data import load_split_summary

    df = load_split_summary()
    axes = set(df["split"].unique().tolist())
    assert {"library", "instrument_type", "instrument_family", "analogue"} <= axes
