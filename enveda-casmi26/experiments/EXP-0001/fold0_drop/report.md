# CASMI26 誤差分解 — fold-safe baseline / fold0 / pool=drop

- 分子数 U: 400
- MRR@25: 0.0003
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| pool | 399 | 0 | 0.9975 |
| ranker | 1 | 0 | 0.0022 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| pool | class3 | 399 | 399.00 | 0 | 0.9975 |
| ranker | class2 | 1 | 0.89 | 0 | 0.0022 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class2 | 0 | 0 | 1 | 0 | 0 | 1 | 0.1111 | 0.0003 |
| class3 | 0 | 0 | 0 | 0 | 399 | 399 | 0.0000 | 0.0000 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 0.0025 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.0000
- channel_analog: 0.0004
- channel_frag: 0.0002

## 6. 候補数

- median 76 / mean 104.0 / 0 件 0
