# CASMI26 誤差分解 — fold-safe baseline / fold0 / pool=keep / ref=other

- 分子数 U: 400
- MRR@25: 0.1002
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| ranker | 384 | 0 | 0.8998 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| ranker | class1 | 1 | 0.50 | 0 | 0.0013 |
| ranker | class2 | 383 | 359.41 | 0 | 0.8985 |
| top1 | class1 | 3 | 0.00 | 0 | 0.0000 |
| top1 | class2 | 13 | 0.00 | 0 | 0.0000 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class1 | 3 | 1 | 0 | 0 | 0 | 4 | 0.8750 | 0.0088 |
| class2 | 13 | 25 | 54 | 91 | 213 | 396 | 0.0924 | 0.0915 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 1.0000 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.0244
- channel_analog: 0.5987
- channel_frag: 0.6172

## 6. 候補数

- median 88 / mean 123.4 / 0 件 0
