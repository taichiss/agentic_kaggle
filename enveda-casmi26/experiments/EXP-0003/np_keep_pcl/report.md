# CASMI26 誤差分解 — fold-safe baseline / np / pool=keep / ref=none

- 分子数 U: 250
- MRR@25: 0.1833
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| ranker | 232 | 0 | 0.8087 |
| pool | 2 | 0 | 0.0080 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| pool | class2 | 2 | 2.00 | 0 | 0.0080 |
| ranker | class2 | 232 | 202.17 | 0 | 0.8087 |
| top1 | class2 | 16 | 0.00 | 0 | 0.0000 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class2 | 16 | 33 | 69 | 67 | 65 | 250 | 0.1833 | 0.1833 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 0.9920 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.0723
- channel_analog: 0.5112
- channel_frag: 0.2489

## 6. 候補数

- median 88 / mean 104.0 / 0 件 0
