# CASMI26 誤差分解 — fold-safe baseline / np / pool=keep / ref=none

- 分子数 U: 250
- MRR@25: 0.2454
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| ranker | 223 | 0 | 0.7506 |
| pool | 1 | 0 | 0.0040 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| pool | class2 | 1 | 1.00 | 0 | 0.0040 |
| ranker | class2 | 223 | 187.65 | 0 | 0.7506 |
| top1 | class2 | 26 | 0.00 | 0 | 0.0000 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class2 | 26 | 41 | 85 | 50 | 48 | 250 | 0.2454 | 0.2454 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 0.9960 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.0527
- channel_analog: 0.5452
- channel_frag: 0.2452

## 6. 候補数

- median 53 / mean 81.8 / 0 件 0
