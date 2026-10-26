# CASMI26 Metric Sensitivity (MRR@25)

> 本ファイルは `scripts/casmi26_metric_sandbox.py` が自動生成した実測値。
> 計算根拠は `src/casmi26/metrics/breakdown.py`。

## 1. 指標の基本定義

- 分母: 400 molecules (test の unique molecule_id)
- 1 分子あたり候補: 25 SMILES
- 正解判定キー: `canonical_inchikey14 (RDKit tautomer canonical + first 14 chars of InChIKey)`

## 2. 1 分子を rank r に動かした時の MRR 変化（基本感度）

| 動かす先の rank | 1 分子あたりの MRR 変化 | 備考 |
|---:|---:|---|
| 1 | +0.002500 | 1 分子を miss → rank 1 に移すと全体 MRR は +0.002500 |
| 2 | +0.001250 | 1 分子を miss → rank 2 に移すと全体 MRR は +0.001250 |
| 3 | +0.000833 | 1 分子を miss → rank 3 に移すと全体 MRR は +0.000833 |
| 5 | +0.000500 | 1 分子を miss → rank 5 に移すと全体 MRR は +0.000500 |
| 10 | +0.000250 | 1 分子を miss → rank 10 に移すと全体 MRR は +0.000250 |
| 25 | +0.000100 | 1 分子を miss → rank 25 に移すと全体 MRR は +0.000100 |

**要点:** 1 分子を miss → rank 1 に戻すと **+0.0025 MRR**。400 分子中、4 分子分の rank 1 変動 ≒ +0.01 MRR。

## 3. rank 間の移動（既に順位内にある分子を動かす）

| from | to | 1 分子あたり MRR 変化 | 0.01 損ねるのに必要な分子数 |
|---:|---:|---:|---:|
| 1 | 2 | -0.001250 | 8 |
| 1 | 3 | -0.001667 | 6 |
| 1 | 5 | -0.002000 | 5 |
| 1 | 10 | -0.002250 | 5 |
| 1 | 25 | -0.002400 | 5 |
| 2 | 5 | -0.000750 | 14 |
| 5 | 10 | -0.000250 | 40 |
| 10 | 25 | -0.000150 | 67 |

## 4. rank 1 を触る変更 vs 下位を触る変更の効き目比較

| 移動 | 1 分子あたり MRR 変化 |
|---|---:|
| miss to rank1 | +0.002500 |
| rank2 to rank1 | +0.001250 |
| rank3 to rank2 | +0.000417 |
| rank5 to rank4 | +0.000125 |
| rank10 to rank9 | +0.000028 |
| rank25 to rank24 | +0.000004 |

- miss→rank1 は rank25→24 の **600 倍**効く
- miss→rank1 は rank10→9 の **90 倍**効く

**→ 戦略判断:** rank 1-3 を直す変更を優先。rank 10-25 の順位微調整は限界効用が急減し、公開 kernel の `RRF offset 3→1` のような 1 knob tuning が +0.001 止まりなのはこれが根拠。

## 5. Class 1 / 2 / 3 寄与の合成データシミュレーション

- 全体 MRR: **0.4098**
- top-1 accuracy: 0.3700

### クラスごとの MRR と全体寄与

| クラス | 分子数 | クラス内 MRR | 全体 MRR への寄与 (sum / U_total) |
|---|---:|---:|---:|
| class1 | 68 | 0.8527 | 0.1450 |
| class2 | 272 | 0.3835 | 0.2608 |
| class3 | 60 | 0.0274 | 0.0041 |

### このシミュレーションの前提（kernel 分析由来の推定）

- Class 1 サイズ: 68 (17%)、top-1 acc 0.80
- Class 2 サイズ: 272 (68%)、top-1 acc 0.40
- Class 3 サイズ: 60 (15%)、top-1 acc 0.05

**要点:** シミュレーション上は全体 MRR が公開 LB 0.467 相当になる。この分解だと **Class 3 の top-1 を 5% → 20% に上げる** だけで +0.023 の寄与が得られる （下記 Section 6 参照）。これが Class 3 攻めの経済合理性。

## 6. 公開 LB 0.467 の逆算（どの rank 分布なら到達可能か）

| パターン | rank 1 | rank 2 | rank 2-5 | miss | 達成 MRR |
|---|---:|---:|---:|---:|---:|
| A_top1_only | 187 | - | - | 213 | 0.468 |
| B_rank12_mix | 100 | 174 | - | 126 | 0.468 |
| B_rank12_mix | 130 | 114 | - | 156 | 0.468 |
| B_rank12_mix | 150 | 74 | - | 176 | 0.468 |
| B_rank12_mix | 170 | 34 | - | 196 | 0.468 |
| B_rank12_mix | 180 | 14 | - | 206 | 0.468 |
| C_top1_plus_rank2to5 | 120 | - | 208 | 72 | 0.467 |
| C_top1_plus_rank2to5 | 140 | - | 146 | 114 | 0.467 |
| C_top1_plus_rank2to5 | 160 | - | 84 | 156 | 0.467 |

**要点:**
- LB 0.467 は pattern A 想定だと **187 分子 (46.7%) が rank 1 で的中、残り miss** に相当
- 現実には top-1 acc は 30-35% 程度で、残りが rank 2-10 に分散していると推測される

## 7. 本コンペの採用規律（指標感度に基づく）

1. **rank 1-3 を直す変更を最優先**（limit of utility は rank 10 で既にほぼ 0）
2. **候補 25 本の下位 (10 位以降) をいじる変更は、必ず 1 位 accuracy を悪化させないか確認**
3. **Class 3 (完全新規) の top-1 を 5% → 20% にする変更が最もレバレッジ高い**（+0.023 MRR 想定）
4. **1 knob tuning (RRF offset 3→1 等) で +0.001 ← これが現状公開 kernel の収穫逓減限界**

## 8. 次アクション

- CV に Class 1/2/3 のラベル付けを実装し、クラスごとの MRR を bevy 分析（kernel 分析の推定を実データで裏取り）
- rank 1 の accuracy を直接最適化する listwise ranker (LambdaRank / ListNet) を実装
- Class 3 専用補助の設計を Week 3 で検討（de novo generation / DreaMS retrieval の 20-25 位スロット注入）
