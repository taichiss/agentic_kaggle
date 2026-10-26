"""CASMI26 identity-disjoint CV fold ローダ。

正本仕様:
    `data/input/external/casmi26/casmi26-identity-disjoint-validation-folds/`
    (Dmitrii Gluzdov, RDKit 2026.03.3, train SHA256 9423...bcd98)

同一 inchikey14 の分子が train / val に分かれない identity-disjoint split と、
Bemis-Murcko scaffold を同一 fold に閉じる scaffold-disjoint split の 2 系列を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

FOLD_DATASET_PATH = Path("data/input/external/casmi26/casmi26-identity-disjoint-validation-folds")
TRAIN_PARQUET_SHA256_EXPECTED = "9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98"

FoldKind = Literal["identity", "scaffold"]


@dataclass(frozen=True)
class FoldAssignment:
    """1 fold の train / validation 仕分け結果 (inchikey14 base)。"""

    kind: FoldKind
    fold: int
    val_inchikey14: frozenset[str]
    train_inchikey14: frozenset[str]
    val_structure_count: int
    train_structure_count: int


def load_identity_folds(
    kind: FoldKind = "identity",
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """structures.parquet を読み、inchikey14 と fold ラベルを DataFrame で返す。

    返り値カラム: inchikey14, scorer_key, smiles, formula, mono_mass,
                  scaffold, identity_group, scaffold_group, fold_identity, fold_scaffold,
                  n_spectra, n_enveda_spectra, n_pos, libs
    """
    root = root or FOLD_DATASET_PATH
    path = root / "structures.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"identity fold dataset not found at {path}. "
            f"download: kaggle datasets download -d dmitriigluzdov/casmi26-identity-disjoint-validation-folds"
        )
    df = pd.read_parquet(path)
    if "fold_identity" not in df.columns or "fold_scaffold" not in df.columns:
        raise ValueError(f"unexpected schema at {path}: {df.columns.tolist()}")
    _ = kind  # kind はカラム選択用の意味だが DataFrame 全列を返す（呼び出し側で使い分け）
    return df


def split_fold(df: pd.DataFrame, fold: int, *, kind: FoldKind = "identity") -> FoldAssignment:
    """load_identity_folds の結果と fold 番号から FoldAssignment を作る。

    train.parquet 側の inchikey14 を join することで train/val に分けられる。
    """
    col = f"fold_{kind}"
    if col not in df.columns:
        raise ValueError(f"missing column {col} in dataframe")
    val_df = df[df[col] == fold]
    train_df = df[df[col] != fold]
    return FoldAssignment(
        kind=kind,
        fold=fold,
        val_inchikey14=frozenset(val_df["inchikey14"].tolist()),
        train_inchikey14=frozenset(train_df["inchikey14"].tolist()),
        val_structure_count=len(val_df),
        train_structure_count=len(train_df),
    )


def load_split_summary(root: Path | None = None) -> pd.DataFrame:
    """library / instrument_type / analogue などの split 概要を返す。"""
    root = root or FOLD_DATASET_PATH
    path = root / "split_summary.parquet"
    if not path.exists():
        raise FileNotFoundError(f"split summary not found at {path}")
    return pd.read_parquet(path)
