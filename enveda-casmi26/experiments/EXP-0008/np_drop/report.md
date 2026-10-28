# CASMI26 誤差分解 — fold-safe baseline / np / pool=drop / ref=none

- 分子数 U: 250
- MRR@25: 0.1860
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| ranker | 226 | 0 | 0.7820 |
| pool | 8 | 0 | 0.0320 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| pool | class2 | 1 | 1.00 | 0 | 0.0040 |
| pool | class3 | 7 | 7.00 | 0 | 0.0280 |
| ranker | class2 | 226 | 195.51 | 0 | 0.7820 |
| top1 | class2 | 16 | 0.00 | 0 | 0.0000 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class2 | 16 | 37 | 62 | 76 | 52 | 243 | 0.1913 | 0.1860 |
| class3 | 0 | 0 | 0 | 0 | 7 | 7 | 0.0000 | 0.0000 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 0.9680 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.0748
- channel_analog: 0.4978
- channel_frag: 0.2528

## 6. 候補数

- median 75 / mean 96.9 / 0 件 0
