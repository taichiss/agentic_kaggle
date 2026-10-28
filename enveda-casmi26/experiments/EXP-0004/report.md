# EXP-0004 ライブラリ検索 誤差分解

- 分子数 U: 250
- MRR@25: 0.8822
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| ranker | 44 | 0 | 0.1058 |
| pool | 3 | 0 | 0.0120 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| pool | all | 3 | 3.00 | 0 | 0.0120 |
| ranker | all | 44 | 26.46 | 0 | 0.1058 |
| top1 | all | 203 | 0.00 | 0 | 0.0000 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| all | 203 | 35 | 8 | 1 | 3 | 250 | 0.8822 | 0.8822 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 0.9880 |
