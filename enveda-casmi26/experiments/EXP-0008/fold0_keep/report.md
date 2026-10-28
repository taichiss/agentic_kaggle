# CASMI26 誤差分解 — fold-safe baseline / fold0 / pool=keep / ref=none

- 分子数 U: 400
- MRR@25: 0.1068
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| ranker | 388 | 0 | 0.8932 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| ranker | class2 | 388 | 357.29 | 0 | 0.8932 |
| top1 | class2 | 12 | 0.00 | 0 | 0.0000 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class2 | 12 | 35 | 63 | 99 | 191 | 400 | 0.1068 | 0.1068 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 1.0000 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.0178
- channel_analog: 0.6035
- channel_frag: 0.6179

## 6. 候補数

- median 85 / mean 118.3 / 0 件 0
