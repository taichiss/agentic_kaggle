# CASMI26 誤差分解 — fold-safe baseline / np / pool=drop

- 分子数 U: 250
- MRR@25: 0.0042
- 段構成 (上流→下流): pool(含有), ranker(top-1)

## 1. 段ごとの損失 (loss_share 降順)

loss_share = その段で最初に落ちた分子の (1 - 1/rank) 合計 / U。その段以降を完璧にしたときの MRR 上限増分。

| bucket | n | recovered | loss_share |
|---|---|---|---|
| pool | 242 | 0 | 0.9680 |
| ranker | 8 | 0 | 0.0278 |

## 2. 段 × クラス

| bucket | class | n | loss_sum | recovered | loss_share |
|---|---|---|---|---|---|
| pool | class3 | 242 | 242.00 | 0 | 0.9680 |
| ranker | class2 | 8 | 6.94 | 0 | 0.0278 |

## 3. クラス別の最終順位分布

| class | rank1 | rank2-3 | rank4-10 | rank11-25 | miss | n | mrr_in_class | contribution |
|---|---|---|---|---|---|---|---|---|
| class2 | 0 | 1 | 5 | 1 | 1 | 8 | 0.1319 | 0.0042 |
| class3 | 0 | 0 | 0 | 0 | 242 | 242 | 0.0000 | 0.0000 |

## 4. 段ごとの MRR 上限 (その段まで通過した割合)

| stage | ceiling |
|---|---|
| pool | 0.0320 |

## 5. 単一チャネルだけで並べたときの MRR@25

- channel_lib: 0.0030
- channel_analog: 0.0089
- channel_frag: 0.0062

## 6. 候補数

- median 52 / mean 80.8 / 0 件 0
