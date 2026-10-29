# CASMI26 誤差分解 — fold-safe baseline / np / pool=keep / ref=other

- 分子数 U: 250
- MRR@25: 0.9172
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| ranker | 35 | 0 | 0.0828 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| ranker | class1 | 35 | 20.70 | 0 | 0.0828 |
| top1 | class1 | 215 | 0.00 | 0 | 0.0000 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class1 | 215 | 27 | 7 | 1 | 0 | 250 | 0.9172 | 0.9172 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 1.0000 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.9293
- channel_analog: 0.6018
- channel_frag: 0.1913
- ranker_prior: 0.9405
- analog_prior: 0.7881

## 6. 候補数

- median 140 / mean 156.7 / 0 件 0
