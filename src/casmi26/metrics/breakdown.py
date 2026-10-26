"""MRR@25 と構成要素の計算 (CASMI26 公式スコアラの再現と因数分解)。

評価指標:
    MRR@25 = (1/U) * Σ_u 1/rank_u

    U は test の unique molecule_id 数、rank_u は分子 u に対する ranked list で
    最初に正解となった構造の順位 (1〜25)。25 候補中に正解がなければ 0 点。

正解判定キー:
    canonical_inchikey14 (RDKit でタウトマー正規化した SMILES から
    MolToInchiKey → 先頭 14 文字)。

    train の `inchikey14` 列は RDKit canonicalization 未適用のため、
    正解判定にはこちらを使う (kernel 分析 `kitopl/casmi-2026-data-pitfalls` 参照)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

K_MAX = 25

_TAUTOMER = rdMolStandardize.TautomerEnumerator()


def canonical_inchikey14(smiles: str) -> str | None:
    """SMILES → canonical InChIKey14。パース失敗時は None。"""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _TAUTOMER.Canonicalize(mol)
    except Exception:
        pass
    inchi_key = Chem.MolToInchiKey(mol)
    if not inchi_key:
        return None
    return inchi_key[:14]


@dataclass(frozen=True)
class SubmissionRow:
    """提出行の 1 分子分 (molecule_id, 候補 SMILES 列、正解 SMILES)。"""

    molecule_id: str
    candidate_smiles: Sequence[str]
    truth_smiles: str


def _first_hit_rank(
    candidate_keys: Sequence[str | None], truth_key: str | None, k_max: int
) -> int | None:
    """truth_key が最初に出た順位 (1-based)。なければ None。"""
    if truth_key is None:
        return None
    for rank, key in enumerate(candidate_keys[:k_max], start=1):
        if key is not None and key == truth_key:
            return rank
    return None


def mrr_at_k(rows: Iterable[SubmissionRow], *, k_max: int = K_MAX) -> float:
    """提出 rows から MRR@k を計算 (0 件の分子は 0 点)。"""
    rows_list = list(rows)
    if not rows_list:
        return 0.0
    total = 0.0
    for row in rows_list:
        truth_key = canonical_inchikey14(row.truth_smiles)
        cand_keys = [canonical_inchikey14(s) for s in row.candidate_smiles]
        rank = _first_hit_rank(cand_keys, truth_key, k_max)
        if rank is not None:
            total += 1.0 / rank
    return total / len(rows_list)


@dataclass(frozen=True)
class BreakdownResult:
    """MRR@k 因数分解結果。

    rank_counts[r] は "正解が rank r で見つかった分子数"。rank_counts[0] は圏外。
    sum / count = MRR@k。

    class_contributions は optional で、クラスラベル (Class 1/2/3 相当) ごとの
    MRR@k 寄与 (sum of 1/rank / U_total) を返す。U_total は全分子数 (母数固定)。
    """

    k_max: int
    count: int
    mrr: float
    top1_accuracy: float
    rank_counts: dict[int, int]
    class_contributions: dict[str, float] | None = None
    class_counts: dict[str, int] | None = None
    class_mrr: dict[str, float] | None = None


def mrr_at_k_breakdown(
    rows: Sequence[SubmissionRow],
    *,
    k_max: int = K_MAX,
    class_labels: Sequence[str] | None = None,
) -> BreakdownResult:
    """提出 rows の MRR@k を因数分解。

    class_labels が与えられれば rows と同長で各分子のクラスラベルを渡し、
    クラスごとの寄与と MRR を返す。全体 MRR への寄与は "クラスの sum(1/rank) / U_total"
    で、U_total は rows 全体の母数 (sum すると全体 MRR と一致)。
    """
    if class_labels is not None and len(class_labels) != len(rows):
        raise ValueError("class_labels length must match rows length")

    rank_counts: dict[int, int] = {0: 0}
    for r in range(1, k_max + 1):
        rank_counts[r] = 0

    class_sum: dict[str, float] = {}
    class_count: dict[str, int] = {}

    total_sum = 0.0
    top1 = 0
    for i, row in enumerate(rows):
        truth_key = canonical_inchikey14(row.truth_smiles)
        cand_keys = [canonical_inchikey14(s) for s in row.candidate_smiles]
        rank = _first_hit_rank(cand_keys, truth_key, k_max)
        rr = 0.0 if rank is None else 1.0 / rank
        total_sum += rr
        if rank == 1:
            top1 += 1
        if rank is None:
            rank_counts[0] += 1
        else:
            rank_counts[rank] += 1

        if class_labels is not None:
            label = class_labels[i]
            class_sum[label] = class_sum.get(label, 0.0) + rr
            class_count[label] = class_count.get(label, 0) + 1

    n = len(rows)
    mrr = total_sum / n if n else 0.0
    top1_acc = top1 / n if n else 0.0

    class_contributions: dict[str, float] | None = None
    class_mrr: dict[str, float] | None = None
    class_counts: dict[str, int] | None = None
    if class_labels is not None and n:
        class_contributions = {k: v / n for k, v in class_sum.items()}
        class_mrr = {
            k: (class_sum[k] / class_count[k]) if class_count[k] else 0.0 for k in class_sum
        }
        class_counts = dict(class_count)

    return BreakdownResult(
        k_max=k_max,
        count=n,
        mrr=mrr,
        top1_accuracy=top1_acc,
        rank_counts=rank_counts,
        class_contributions=class_contributions,
        class_counts=class_counts,
        class_mrr=class_mrr,
    )


def score_submission(
    submission_smiles: dict[str, Sequence[str]],
    truth_smiles: dict[str, str],
    *,
    k_max: int = K_MAX,
) -> BreakdownResult:
    """submission dict と truth dict から MRR@k 因数分解結果を返す。

    submission_smiles[molecule_id] は候補 SMILES 列 (`;` 連結ではなく list)。
    truth_smiles[molecule_id] は正解 SMILES。
    truth にない molecule_id は無視、truth にあって submission にない分子は 0 点扱い。
    """
    rows: list[SubmissionRow] = []
    for mol_id, truth in truth_smiles.items():
        cands = submission_smiles.get(mol_id, [])
        rows.append(
            SubmissionRow(
                molecule_id=mol_id,
                candidate_smiles=list(cands),
                truth_smiles=truth,
            )
        )
    return mrr_at_k_breakdown(rows, k_max=k_max)
