"""CASMI26 Metric Sandbox (kaggle_flow_v1.md §3 Week 0 必須項目)。

MRR@25 の rank 1→k 感度、Class 1/2/3 の寄与、公開 LB 0.467 を到達している kernel
が top-k acc で何%にいるかの逆算、を自分で実測して metric_sensitivity.md に残す。

bio-hub L1 の反省:
    "S1 Metric Sandbox で division FP 1 件 = edge FP 2.7 本相当と実測したのに、
     ILP で division_weight=0.05 と分裂に消極的な設定を選んだ"
→ 感度を実測した根拠は後から直観で覆してはいけない。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from casmi26.metrics import SubmissionRow, mrr_at_k_breakdown

# 400 molecules × 25 候補が公式提出サイズ
N_MOLECULES = 400
K_MAX = 25

# 既知の SMILES 群（RDKit で正しくパースできる単純な化合物）
# これらは InChIKey14 が全部ユニークなので候補 pool として使える
BASE_SMILES = [
    "CCO",  # ethanol
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
    "C1=CC=C(C=C1)C=O",  # benzaldehyde
    "CCC(=O)O",  # propionic acid
    "C(C(=O)O)N",  # glycine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
    "C1=CC(=CN=C1)C(=O)O",  # nicotinic acid
    "O=C1NC(=O)NC(=O)N1",  # cyanuric acid
    "CC1=CC=C(C=C1)C(=O)O",  # p-toluic acid
    "OC(=O)CCC(=O)O",  # succinic acid
    "C1=CC=C2C(=C1)C=CC=C2",  # naphthalene
    "CN(C)C",  # trimethylamine
    "C1CCNCC1",  # piperidine
    "CC(C)O",  # isopropanol
    "C1=CC=C(C=C1)N",  # aniline
    "C(C(=O)O)O",  # glycolic acid
    "O=C(O)C(O)C(O)C(=O)O",  # tartaric acid
    "C1=CC=C(C(=C1)C(=O)O)O",  # salicylic acid
    "CC(=O)NC1=CC=C(C=C1)O",  # paracetamol
    "C1=CC=C2C(=C1)N=CC=C2",  # quinoline
    "C(C(C(C(C(=O)O)O)O)O)O",  # gluconic acid
    "CC(=O)O",  # acetic acid
    "C1=CC=C(C(=C1)C=O)O",  # salicylaldehyde
    "CC(C)CO",  # isobutanol
    "C1=CC(=CC=C1C=O)O",  # p-hydroxybenzaldehyde
    "CCCCO",  # butanol
    "OCC(O)CO",  # glycerol
    "C(=O)N",  # formamide
]


@dataclass(frozen=True)
class SensitivityCurve:
    """ "rank r の正解" を 400 分子中 N 件増やしたとき、MRR はどれだけ動くか" を記録。"""

    rank: int
    delta_mrr_per_molecule: float
    note: str


def exchange_rates() -> list[SensitivityCurve]:
    """1 分子を "miss → rank r" に変えたときの MRR 変化率を列挙。

    MRR の分母は常に 400 なので、1 分子あたりの寄与は 1/(400 * r)。
    "miss → rank r" で +1/(400*r)、"rank r1 → rank r2" で +(1/r2 - 1/r1)/400。
    """
    out = []
    for r in [1, 2, 3, 5, 10, 25]:
        delta = 1.0 / (N_MOLECULES * r)
        out.append(
            SensitivityCurve(
                rank=r,
                delta_mrr_per_molecule=delta,
                note=f"1 分子を miss → rank {r} に移すと全体 MRR は +{delta:.6f}",
            )
        )
    return out


def pairwise_rank_moves() -> list[dict[str, Any]]:
    """既に正解が rank a にあった分子を rank b に移したときの MRR 変化（400 分子想定）。"""
    pairs = [(1, 2), (1, 3), (1, 5), (1, 10), (1, 25), (2, 5), (5, 10), (10, 25)]
    out = []
    for a, b in pairs:
        delta_per = (1.0 / b - 1.0 / a) / N_MOLECULES
        out.append(
            {
                "from_rank": a,
                "to_rank": b,
                "delta_mrr_per_molecule": delta_per,
                "molecules_to_lose_0_01": (
                    None if delta_per == 0 else int(math.ceil(abs(-0.01 / delta_per)))
                ),
                "note": f"rank {a} → {b} に 1 分子落とすと {delta_per:+.6f} / 分子",
            }
        )
    return out


def class_contribution_simulation(seed: int = 0) -> dict[str, Any]:
    """Class 1 / 2 / 3 の寄与分解をシミュレーション。

    Discussion / kernel 分析の推定:
      - Class 1 (リファレンススペクトル存在): ~16-17%、spectral match 単独で高精度
      - Class 2 (構造のみ既知、PubChem retrieval): 大半、公開 kernel の MRR 主成分
      - Class 3 (完全新規): 残り、公開 LB で停滞中

    400 分子を 17% / 68% / 15% に割り振り、公開 LB 0.467 を再現する簡易モデル:
      - Class 1: top1 acc 0.80 (スペクトル検索でほぼ当てる), 外しは rank 2-5
      - Class 2: top1 acc 0.40, 多くは rank 1-10
      - Class 3: top1 acc 0.05, ほぼ miss
    """
    rng = np.random.default_rng(seed)
    sizes = {"class1": 68, "class2": 272, "class3": 60}  # 400 中、合計 400
    assert sum(sizes.values()) == N_MOLECULES

    rows: list[SubmissionRow] = []
    labels: list[str] = []

    # 各クラスのモックランク分布
    class_rank_dists = {
        "class1": {"top1": 0.80, "max_miss_rank": 5},
        "class2": {"top1": 0.40, "max_miss_rank": 20},
        "class3": {"top1": 0.05, "max_miss_rank": 25},
    }

    truth_smiles = BASE_SMILES[0]  # どの分子でも SMILES 自体は同一を使う（評価は rank のみ依存）
    noise_smiles = BASE_SMILES[1:26]  # 25 本の候補 noise

    for cls, n in sizes.items():
        dist = class_rank_dists[cls]
        for _ in range(n):
            u = rng.random()
            if u < dist["top1"]:
                rank = 1
            elif u < dist["top1"] + 0.3:
                rank = rng.integers(2, min(dist["max_miss_rank"], K_MAX) + 1)
            else:
                rank = None  # miss

            cand = list(noise_smiles)
            if rank is not None:
                cand[rank - 1] = truth_smiles
            rows.append(
                SubmissionRow(
                    molecule_id=f"m{len(rows)}",
                    candidate_smiles=cand,
                    truth_smiles=truth_smiles,
                )
            )
            labels.append(cls)

    r = mrr_at_k_breakdown(rows, class_labels=labels)
    return {
        "sizes": sizes,
        "rank_dist": class_rank_dists,
        "overall_mrr": r.mrr,
        "top1_accuracy": r.top1_accuracy,
        "class_mrr": r.class_mrr,
        "class_contributions_to_overall": r.class_contributions,
        "class_counts": r.class_counts,
        "rank_counts": {k: v for k, v in r.rank_counts.items() if v > 0},
    }


def reverse_engineer_top_k_acc_for_lb(target_mrr: float) -> dict[str, Any]:
    """公開 LB 0.467 を達成する "各 rank の分布" を複数パターン提示。

    400 分子で MRR = 0.467 を達成する rank 分布の候補:
      A) 全員 rank 1 の 46.7% 分: n1 = 187, 他 miss → MRR = 187/400 = 0.4675
      B) rank 1 のみでは不足、2 を混ぜる組
      C) rank 1-5 の重みで現実的分布
    """
    out = []
    # A: top-1 のみで MRR 0.467 を作る
    n1_only = int(round(target_mrr * N_MOLECULES))
    out.append(
        {
            "pattern": "A_top1_only",
            "n_rank1": n1_only,
            "n_miss": N_MOLECULES - n1_only,
            "achieved_mrr": n1_only / N_MOLECULES,
            "note": f"全部 rank 1、残り miss。{n1_only}/{N_MOLECULES} = {n1_only / N_MOLECULES:.3f}",
        }
    )
    # B: rank 1 と rank 2 の組み合わせ
    for n1 in [100, 130, 150, 170, 180]:
        n2 = int(round((target_mrr - n1 / N_MOLECULES) * N_MOLECULES * 2))
        if n2 < 0 or n1 + n2 > N_MOLECULES:
            continue
        achieved = (n1 + n2 * 0.5) / N_MOLECULES
        out.append(
            {
                "pattern": "B_rank12_mix",
                "n_rank1": n1,
                "n_rank2": n2,
                "n_miss": N_MOLECULES - n1 - n2,
                "achieved_mrr": achieved,
                "note": f"rank1={n1}, rank2={n2}, miss={N_MOLECULES - n1 - n2}, MRR={achieved:.3f}",
            }
        )
    # C: 一様に rank 1-5 がある場合
    for p1 in [0.30, 0.35, 0.40]:
        n1 = int(round(p1 * N_MOLECULES))
        remaining_mrr = target_mrr - p1
        # 残り (1-p1) 分子を rank 2-5 に一様分布に見立てる
        avg_recip = (1 / 2 + 1 / 3 + 1 / 4 + 1 / 5) / 4
        n_rest = int(round(remaining_mrr / avg_recip * N_MOLECULES))
        if n1 + n_rest > N_MOLECULES:
            continue
        achieved = (n1 + n_rest * avg_recip) / N_MOLECULES
        out.append(
            {
                "pattern": "C_top1_plus_rank2to5",
                "n_rank1": n1,
                "n_rank2to5_uniform": n_rest,
                "n_miss": N_MOLECULES - n1 - n_rest,
                "achieved_mrr": achieved,
                "note": f"rank1={n1}({p1:.0%}), rank2-5 一様 {n_rest} 本",
            }
        )
    return {"target_mrr": target_mrr, "patterns": out}


def rank1_vs_rank2to25_elasticity() -> dict[str, Any]:
    """ "rank 1 の精度を 1% 上げる" vs "rank 2-25 の順位を 1 階段改善する" のどちらが効くか。

    1 分子を rank 1 に移す = +1/(400*1) = +0.0025
    1 分子を rank 25 → 24 に移す = (1/24 - 1/25)/400 = +0.0000042
    → 比率で 600 倍の差。rank 1 を触る変更は低い順位を触る変更の 600 倍効く。
    """
    r1_gain = 1.0 / (N_MOLECULES * 1)
    r25_to_24 = (1 / 24 - 1 / 25) / N_MOLECULES
    r10_to_9 = (1 / 9 - 1 / 10) / N_MOLECULES
    r5_to_4 = (1 / 4 - 1 / 5) / N_MOLECULES
    r3_to_2 = (1 / 2 - 1 / 3) / N_MOLECULES
    r2_to_1 = (1 / 1 - 1 / 2) / N_MOLECULES
    return {
        "rank_moves_vs_rank1_gain": {
            "miss_to_rank1": r1_gain,
            "rank2_to_rank1": r2_to_1,
            "rank3_to_rank2": r3_to_2,
            "rank5_to_rank4": r5_to_4,
            "rank10_to_rank9": r10_to_9,
            "rank25_to_rank24": r25_to_24,
        },
        "ratio_rank1_vs_rank25_to_24": r1_gain / r25_to_24,
        "ratio_rank1_vs_rank10_to_9": r1_gain / r10_to_9,
        "note": (
            "miss→rank1 は rank 25→24 の "
            f"{r1_gain / r25_to_24:.0f} 倍、rank 10→9 の "
            f"{r1_gain / r10_to_9:.0f} 倍効く。"
            "→ 下位の順位微調整は限界効用が急減、rank 1-3 を触る変更が主戦場。"
        ),
    }


def main() -> None:
    out_dir = Path("enveda-casmi26")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "metadata": {
            "metric": "MRR@25",
            "n_molecules": N_MOLECULES,
            "k_max": K_MAX,
            "scoring_key": "canonical_inchikey14 (RDKit tautomer canonical + first 14 chars of InChIKey)",
        },
        "exchange_rates_per_rank": [
            {
                "rank": c.rank,
                "delta_mrr_per_molecule": c.delta_mrr_per_molecule,
                "note": c.note,
            }
            for c in exchange_rates()
        ],
        "pairwise_rank_moves": pairwise_rank_moves(),
        "rank1_vs_rank2to25_elasticity": rank1_vs_rank2to25_elasticity(),
        "class_contribution_simulation": class_contribution_simulation(seed=0),
        "reverse_engineer_lb_0_467": reverse_engineer_top_k_acc_for_lb(0.467),
    }

    out_path = out_dir / "metric_sensitivity.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")

    md_path = out_dir / "metric_sensitivity.md"
    md_path.write_text(_render_markdown(results))
    print(f"wrote {md_path}")


def _render_markdown(r: dict[str, Any]) -> str:
    lines = [
        "# CASMI26 Metric Sensitivity (MRR@25)",
        "",
        "> 本ファイルは `scripts/casmi26_metric_sandbox.py` が自動生成した実測値。",
        "> 計算根拠は `src/casmi26/metrics/breakdown.py`。",
        "",
        "## 1. 指標の基本定義",
        "",
        f"- 分母: {r['metadata']['n_molecules']} molecules (test の unique molecule_id)",
        f"- 1 分子あたり候補: {r['metadata']['k_max']} SMILES",
        f"- 正解判定キー: `{r['metadata']['scoring_key']}`",
        "",
        "## 2. 1 分子を rank r に動かした時の MRR 変化（基本感度）",
        "",
        "| 動かす先の rank | 1 分子あたりの MRR 変化 | 備考 |",
        "|---:|---:|---|",
    ]
    for c in r["exchange_rates_per_rank"]:
        lines.append(f"| {c['rank']} | +{c['delta_mrr_per_molecule']:.6f} | {c['note']} |")
    lines.extend(
        [
            "",
            "**要点:** 1 分子を miss → rank 1 に戻すと **+0.0025 MRR**。400 分子中、"
            "4 分子分の "
            "rank 1 変動 ≒ +0.01 MRR。",
            "",
            "## 3. rank 間の移動（既に順位内にある分子を動かす）",
            "",
            "| from | to | 1 分子あたり MRR 変化 | 0.01 損ねるのに必要な分子数 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for p in r["pairwise_rank_moves"]:
        mol_count = p["molecules_to_lose_0_01"]
        mol_str = f"{mol_count}" if mol_count is not None else "-"
        lines.append(
            f"| {p['from_rank']} | {p['to_rank']} | {p['delta_mrr_per_molecule']:+.6f} | {mol_str} |"
        )
    lines.extend(
        [
            "",
            "## 4. rank 1 を触る変更 vs 下位を触る変更の効き目比較",
            "",
            "| 移動 | 1 分子あたり MRR 変化 |",
            "|---|---:|",
        ]
    )
    for k, v in r["rank1_vs_rank2to25_elasticity"]["rank_moves_vs_rank1_gain"].items():
        lines.append(f"| {k.replace('_', ' ')} | +{v:.6f} |")
    lines.extend(
        [
            "",
            f"- miss→rank1 は rank25→24 の **{r['rank1_vs_rank2to25_elasticity']['ratio_rank1_vs_rank25_to_24']:.0f} 倍**効く",
            f"- miss→rank1 は rank10→9 の **{r['rank1_vs_rank2to25_elasticity']['ratio_rank1_vs_rank10_to_9']:.0f} 倍**効く",
            "",
            "**→ 戦略判断:** rank 1-3 を直す変更を優先。rank 10-25 の順位微調整は限界効用が急減し、公開 kernel の "
            "`RRF offset 3→1` のような "
            "1 knob tuning が +0.001 止まりなのはこれが根拠。",
            "",
            "## 5. Class 1 / 2 / 3 寄与の合成データシミュレーション",
            "",
            f"- 全体 MRR: **{r['class_contribution_simulation']['overall_mrr']:.4f}**",
            f"- top-1 accuracy: {r['class_contribution_simulation']['top1_accuracy']:.4f}",
            "",
            "### クラスごとの MRR と全体寄与",
            "",
            "| クラス | 分子数 | クラス内 MRR | 全体 MRR への寄与 (sum / U_total) |",
            "|---|---:|---:|---:|",
        ]
    )
    sim = r["class_contribution_simulation"]
    for cls in ("class1", "class2", "class3"):
        n = sim["class_counts"].get(cls, 0)
        cmrr = sim["class_mrr"].get(cls, 0.0)
        ccontrib = sim["class_contributions_to_overall"].get(cls, 0.0)
        lines.append(f"| {cls} | {n} | {cmrr:.4f} | {ccontrib:.4f} |")
    lines.extend(
        [
            "",
            "### このシミュレーションの前提（kernel 分析由来の推定）",
            "",
            f"- Class 1 サイズ: {sim['sizes']['class1']} (17%)、top-1 acc {sim['rank_dist']['class1']['top1']:.2f}",
            f"- Class 2 サイズ: {sim['sizes']['class2']} (68%)、top-1 acc {sim['rank_dist']['class2']['top1']:.2f}",
            f"- Class 3 サイズ: {sim['sizes']['class3']} (15%)、top-1 acc {sim['rank_dist']['class3']['top1']:.2f}",
            "",
            "**要点:** シミュレーション上は全体 MRR が公開 LB 0.467 相当になる。この分解だと "
            "**Class 3 の top-1 を 5% → 20% に上げる** だけで +0.023 の寄与が得られる "
            "（下記 Section 6 参照）。これが Class 3 攻めの経済合理性。",
            "",
            "## 6. 公開 LB 0.467 の逆算（どの rank 分布なら到達可能か）",
            "",
            "| パターン | rank 1 | rank 2 | rank 2-5 | miss | 達成 MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for p in r["reverse_engineer_lb_0_467"]["patterns"]:
        n1 = p.get("n_rank1", "-")
        n2 = p.get("n_rank2", "-")
        n25 = p.get("n_rank2to5_uniform", "-")
        nmiss = p.get("n_miss", "-")
        lines.append(
            f"| {p['pattern']} | {n1} | {n2} | {n25} | {nmiss} | {p['achieved_mrr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "**要点:**",
            "- LB 0.467 は pattern A 想定だと **187 分子 (46.7%) が rank 1 で的中、残り miss** に相当",
            "- 現実には top-1 acc は 30-35% 程度で、残りが rank 2-10 に分散していると推測される",
            "",
            "## 7. 本コンペの採用規律（指標感度に基づく）",
            "",
            "1. **rank 1-3 を直す変更を最優先**（limit of utility は rank 10 で既にほぼ 0）",
            "2. **候補 25 本の下位 (10 位以降) をいじる変更は、必ず 1 位 accuracy を悪化させないか確認**",
            "3. **Class 3 (完全新規) の top-1 を 5% → 20% にする変更が最もレバレッジ高い**（+0.023 MRR 想定）",
            "4. **1 knob tuning (RRF offset 3→1 等) で +0.001 ← これが現状公開 kernel の収穫逓減限界**",
            "",
            "## 8. 次アクション",
            "",
            "- CV に Class 1/2/3 のラベル付けを実装し、クラスごとの MRR を bevy 分析（kernel 分析の推定を実データで裏取り）",
            "- rank 1 の accuracy を直接最適化する listwise ranker (LambdaRank / ListNet) を実装",
            "- Class 3 専用補助の設計を Week 3 で検討（de novo generation / DreaMS retrieval の 20-25 位スロット注入）",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
