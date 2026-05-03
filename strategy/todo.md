# TODO

## 今やる
- Kaggle kernel `birdclef-2026-full-fit-probe-submit` の実行完了と `submission.csv` 生成を確認する
- `66 files` 全量を使う `site-balanced file 3-fold` を local 主ゲートとして固定し、prior / ResidualSSM / threshold の fold-safe OOF をこの split で回す
- `file GroupKFold 5-fold` を副ゲートとして残し、`probe_pca_blend` を Perch 系の新基準線にする

## 次にやる
- prior、ResidualSSM、threshold 最適化を fold-safe にして、本番 pipeline の OOF macro-AUC を比較する
- `probe_pca_blend` に site/hour prior や温度補正を足して `0.891374` を超えるか試す
- `site holdout 3-fold` の `S22` 偏りを和らげる grouping / weighting を試し、stress test の分散を下げる
- partial files を含めた `66 files` 全量対応で、notebook の reshape 前提を外した評価コードへ整理する
- `scripts/audio_eda.py` に pseudo-label、OOF score、閾値判定の overlay を追加する
- `doc/overview/2026/species_distribution.svg` を見ながら rare class / soundscape-only class を棚卸しし、sampling / class weighting の仮説を作る
- 保存したアノテーションを class / site / recorder 単位で検索・再集計できるようにする

## 保留
- pure site holdout をどの程度 LB 上限確認に使うかは、`0.857185` の実 OOF と S22 偏りを踏まえて再判断する
- distilled SED 系の比較基準線は、validation 正本が固まった後に追加する
- export / runtime 形式の比較は、local CV を固めてから着手する
