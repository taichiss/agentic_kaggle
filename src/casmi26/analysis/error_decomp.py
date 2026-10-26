"""CASMI26 誤差分解 (kaggle_flow_v1.md §4 Week 1 誤差分解ゲート)。

パイプラインの各段で「正解構造がどこで落ちたか」を分子単位で判定し、
Class 1/2/3 ラベル付きで MRR@25 の損失を段ごとに配分する。

段の既定値 (strategy/todo.md Phase D):
    pool       — 正解が候補 pool に入っているか (順位は問わない)
    fp         — FP 予測スコアで top-N (既定 100) に入っているか
    simulator  — forward MS overlay 後に top-25 に入っているか
    ranker     — 最終ランカーで rank 1 を取れているか (最終提出の順位)

クラス定義 (doc/discussion/2026/casmi26/summary.md §3.1):
    class1 — reference スペクトルが学習側ライブラリにある (library hit 可能)
    class2 — 構造は候補 DB にあるが reference スペクトルはない
    class3 — 候補 DB にも存在しない (de novo 生成が必要)

入力の段出力は long format の DataFrame:
    molecule_id, candidate_key, rank (1-based, 小さいほど上位)
candidate_key は正解と同じ canonical_inchikey14 であること
(生の inchikey14 は 1.15% の構造で一致しない。`attach_canonical_keys` で変換する)。

損失の配分:
    分子 u の損失は 1 - 1/rank_u (圏外なら 1)。最終 rank 1 の分子は損失 0。
    rank 1 でない分子は「最初に通過条件を満たさなかった段」に損失を全額計上する。
    段 s の loss_share = その段に計上された損失の合計 / U は、
    「段 s 以降を完璧にしてその分子群を全部 rank 1 にしたときの MRR 上限増分」。
    全段の loss_share の合計は 1 - MRR と一致する。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from casmi26.metrics.breakdown import K_MAX, canonical_inchikey14

CLASS_LABELS = ("class1", "class2", "class3")
TOP1 = "top1"

# 最終順位のバケット (metric_sensitivity.md の感度区分に合わせる)
RANK_BINS: tuple[tuple[str, int, int], ...] = (
    ("rank1", 1, 1),
    ("rank2-3", 2, 3),
    ("rank4-10", 4, 10),
    ("rank11-25", 11, K_MAX),
)
MISS = "miss"


@dataclass(frozen=True)
class StageSpec:
    """パイプラインの 1 段。

    pass_rank が None の段は「候補に含まれているか」だけを見る (pool 段)。
    整数なら「正解の順位が pass_rank 以内か」を通過条件にする。
    """

    name: str
    pass_rank: int | None


def default_stages(fp_top_n: int = 100) -> tuple[StageSpec, ...]:
    """todo.md Phase D の 4 段構成。最後の段が最終提出の順位を表す。"""
    return (
        StageSpec("pool", None),
        StageSpec("fp", fp_top_n),
        StageSpec("simulator", K_MAX),
        StageSpec("ranker", 1),
    )


@dataclass
class ErrorDecomposition:
    """誤差分解の結果。

    per_molecule: 分子ごとの段別順位、最終順位、クラス、計上先バケット
    stage_table: 計上先バケット × クラスの分子数と loss_share
    ceilings: 各段を通過した分子の割合 (= それ以降の段が完璧なときの MRR 上限)
    """

    mrr: float
    count: int
    stages: tuple[StageSpec, ...]
    per_molecule: pd.DataFrame
    stage_table: pd.DataFrame
    rank_table: pd.DataFrame
    ceilings: dict[str, float] = field(default_factory=dict)


def attach_canonical_keys(
    df: pd.DataFrame, *, smiles_col: str = "smiles", key_col: str = "candidate_key"
) -> pd.DataFrame:
    """SMILES 列から canonical_inchikey14 列を追加する (一意 SMILES ごとに 1 回だけ計算)。"""
    unique = df[smiles_col].dropna().unique()
    mapping = {s: canonical_inchikey14(s) for s in unique}
    out = df.copy()
    out[key_col] = out[smiles_col].map(mapping)
    return out


def assign_classes(
    truth_keys: Mapping[str, str | None],
    *,
    reference_keys: Iterable[str],
    database_keys: Iterable[str],
) -> dict[str, str]:
    """molecule_id → class ラベル。

    reference_keys: 学習側 (fold の train 部分) にスペクトルがある構造キー
    database_keys: 候補 DB (PubChem / COCONUT 等) に収録されている構造キー
    reference にある構造は DB の有無に関わらず class1。
    """
    ref = set(reference_keys)
    db = set(database_keys)
    labels: dict[str, str] = {}
    for mol_id, key in truth_keys.items():
        if key is not None and key in ref:
            labels[mol_id] = "class1"
        elif key is not None and key in db:
            labels[mol_id] = "class2"
        else:
            labels[mol_id] = "class3"
    return labels


def _truth_rank(stage_df: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """各 molecule_id について正解キーの最小順位を返す (なければ欠損)。"""
    missing = {"molecule_id", "candidate_key", "rank"} - set(stage_df.columns)
    if missing:
        raise ValueError(f"stage output is missing columns: {sorted(missing)}")
    merged = stage_df[["molecule_id", "candidate_key", "rank"]].merge(
        truth[["molecule_id", "truth_key"]], on="molecule_id", how="inner"
    )
    hits = merged[merged["candidate_key"] == merged["truth_key"]]
    return hits.groupby("molecule_id")["rank"].min()


def _rank_bin(rank: float) -> str:
    if pd.isna(rank):
        return MISS
    for name, lo, hi in RANK_BINS:
        if lo <= rank <= hi:
            return name
    return MISS


def decompose(
    truth: Mapping[str, str | None],
    stage_outputs: Mapping[str, pd.DataFrame],
    *,
    stages: Sequence[StageSpec] | None = None,
    classes: Mapping[str, str] | None = None,
    k_max: int = K_MAX,
) -> ErrorDecomposition:
    """誤差分解を実行する。

    truth: molecule_id → 正解の canonical_inchikey14 (母数 U はこの件数)
    stage_outputs: 段名 → long format の段出力。stages に含まれる段は全て必須
    stages: 上流から順に並べた段。最後の段の順位が最終提出の順位
    classes: molecule_id → クラスラベル。省略時は全分子 "all"
    """
    stages = tuple(stages) if stages is not None else default_stages()
    if not stages:
        raise ValueError("stages must not be empty")
    absent = [s.name for s in stages if s.name not in stage_outputs]
    if absent:
        raise ValueError(f"stage outputs missing for: {absent}")
    if not truth:
        raise ValueError("truth must not be empty")

    truth_df = pd.DataFrame({"molecule_id": list(truth.keys()), "truth_key": list(truth.values())})
    per = truth_df.copy()
    per["class"] = (
        per["molecule_id"].map(classes).fillna("unlabeled") if classes is not None else "all"
    )

    for spec in stages:
        per[f"rank_{spec.name}"] = per["molecule_id"].map(
            _truth_rank(stage_outputs[spec.name], truth_df)
        )
        # 正解キーが作れなかった分子はどの段でもヒットしない
        per.loc[per["truth_key"].isna(), f"rank_{spec.name}"] = float("nan")

    final_col = f"rank_{stages[-1].name}"
    final = per[final_col].where(per[final_col] <= k_max)
    per["final_rank"] = final
    per["rr"] = (1.0 / final).fillna(0.0)
    per["loss"] = 1.0 - per["rr"]
    per["rank_bin"] = per["final_rank"].map(_rank_bin)

    def first_failure(row: pd.Series) -> str:
        if row["final_rank"] == 1:
            return TOP1
        for spec in stages:
            rank = row[f"rank_{spec.name}"]
            if pd.isna(rank):
                return spec.name
            if spec.pass_rank is not None and rank > spec.pass_rank:
                return spec.name
        return stages[-1].name

    per["bucket"] = per.apply(first_failure, axis=1)
    # 前段で落ちたのに最終で拾えた分子 (fusion 系の救済) を可視化する
    per["recovered"] = (per["bucket"] != TOP1) & (per["bucket"] != stages[-1].name)
    per["recovered"] &= per["final_rank"].notna()

    n = len(per)
    mrr = float(per["rr"].sum() / n)

    bucket_order = [s.name for s in stages] + [TOP1]
    stage_table = (
        per.groupby(["bucket", "class"])
        .agg(n=("molecule_id", "size"), loss_sum=("loss", "sum"), recovered=("recovered", "sum"))
        .reset_index()
    )
    stage_table["loss_share"] = stage_table["loss_sum"] / n
    stage_table["bucket"] = pd.Categorical(
        stage_table["bucket"], categories=bucket_order, ordered=True
    )
    stage_table = stage_table.sort_values(["bucket", "class"]).reset_index(drop=True)
    stage_table["bucket"] = stage_table["bucket"].astype(str)

    bin_order = [b[0] for b in RANK_BINS] + [MISS]
    rank_table = (
        pd.crosstab(per["class"], per["rank_bin"])
        .reindex(columns=bin_order, fill_value=0)
        .rename_axis(columns=None)
    )
    rank_table["n"] = rank_table.sum(axis=1)
    rank_table["mrr_in_class"] = per.groupby("class")["rr"].mean()
    rank_table["contribution"] = per.groupby("class")["rr"].sum() / n

    ceilings: dict[str, float] = {}
    passed = pd.Series(True, index=per.index)
    for spec in stages[:-1]:
        rank = per[f"rank_{spec.name}"]
        ok = rank.notna() if spec.pass_rank is None else rank <= spec.pass_rank
        passed &= ok
        ceilings[spec.name] = float(passed.mean())

    return ErrorDecomposition(
        mrr=mrr,
        count=n,
        stages=stages,
        per_molecule=per,
        stage_table=stage_table,
        rank_table=rank_table,
        ceilings=ceilings,
    )


def _md_table(df: pd.DataFrame, floats: Mapping[str, str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floats[c]) if c in floats else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_markdown(result: ErrorDecomposition, *, title: str = "CASMI26 誤差分解") -> str:
    """結果を Markdown レポートにする。"""
    by_stage = (
        result.stage_table.groupby("bucket", sort=False)
        .agg(n=("n", "sum"), recovered=("recovered", "sum"), loss_share=("loss_share", "sum"))
        .reset_index()
    )
    by_stage = by_stage[by_stage["bucket"] != TOP1].sort_values("loss_share", ascending=False)

    stage_desc = ", ".join(
        f"{s.name}" + ("(含有)" if s.pass_rank is None else f"(top-{s.pass_rank})")
        for s in result.stages
    )
    rank_df = result.rank_table.reset_index().rename(columns={"index": "class"})
    ceil_df = pd.DataFrame(
        {"stage": list(result.ceilings.keys()), "ceiling": list(result.ceilings.values())}
    )

    parts = [
        f"# {title}",
        "",
        f"- 分子数 U: {result.count}",
        f"- MRR@{K_MAX}: {result.mrr:.4f}",
        f"- 段構成 (上流→下流): {stage_desc}",
        "",
        "## 1. 段ごとの損失 (loss_share 降順)",
        "",
        "loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。"
        "その段以降を完璧にしたときの MRR 上限増分。",
        "",
        _md_table(by_stage, {"loss_share": ".4f"}),
        "",
        "## 2. 段 × クラス",
        "",
        _md_table(result.stage_table, {"loss_sum": ".2f", "loss_share": ".4f"}),
        "",
        "## 3. クラス別の最終順位分布",
        "",
        _md_table(rank_df, {"mrr_in_class": ".4f", "contribution": ".4f"}),
        "",
        "## 4. 段ごとの MRR 上限 (その段まで通過した割合)",
        "",
        _md_table(ceil_df, {"ceiling": ".4f"}),
        "",
    ]
    return "\n".join(parts)
