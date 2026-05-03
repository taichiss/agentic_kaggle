# 現在の戦略

## コンペ概要
- competition: BirdCLEF+ 2026
- metric: 真陽性が存在しないクラスを除外する macro ROC-AUC
- submission_format: `submission.csv`。各 `row_id` に対して 234 クラス確率を出力する
- constraints: Notebook 提出のみ、CPU 実行 90 分以内、インターネット禁止、公開外部データと事前学習モデルは利用可

## 利用可能データ
- `train_audio`: `35,549` 行、206 クラスに primary ラベル付き focal 音声があるが、分類群分布は大きく偏る
- `train_soundscapes`: ディレクトリ全体では `10,658` files あり、そのうち窓ラベル付きは `66` files
- `train_soundscapes_labels.csv`: 重複除去後 `739` windows、`66` files、`9` sites
- notebook 基準線が使う fully-labeled subset: `708` windows、`59` files、`8` sites
- `taxonomy.csv`: 提出対象 234 クラス

## 確認済み事項
- `notebook.ipynb` の OOF は `filename` 単位 `GroupKFold` で、fully-labeled `59` files を使う
- fully-labeled subset だけだと active classes は `71`。重複除去後の全 labeled windows を使うと `75` まで増える
- partial files は `7` 本あり、追加される site は `S09`、追加クラスは `516975`, `grekis`, `plcjay1`, `thlwre1`
- Perch `SavedModel` をローカル展開し、`scripts/experiment/perch_probe_cv.py` で実学習 CV を再現できるようにした
- 実学習の raw Perch 基準線では、`full59_raw_perch_filegkf5` が `0.729017`、`all66_raw_perch_filegkf5` が `0.736658`
- `raw logits + embedding PCA + balanced logistic probe` をそのまま確率置換すると OOF macro-AUC は悪化した
- `raw Perch` を土台にし、`min_pos >= 3` のクラスだけ `embedding PCA probe` を logit 空間で `alpha=0.4` 混合する補正型 probe にすると、`full59_probe_pca_blend_filegkf5` は `0.886197`、`all66_probe_pca_blend_filegkf5` は `0.891374`
- 上記の改善後でも、`all66_probe_pca_blend_sitebalanced3` は `0.882219` で、file GKF より少し厳しいが十分高い
- `all66_probe_pca_blend_siteholdout3` も実学習では `0.857185` まで回復したが、mean fold AUC は `0.743780` で、`S22` 単独 fold (`477 windows`) の偏りが強い
- site-balanced 3-fold は fold ごとの active classes が `42-53`。notebook 基準の `27-45` より被覆が安定する
- pure site holdout 3-fold は proxy では `0.193321` と厳しすぎ、主ゲート単独採用には不向き
- `scripts/experiment/train_perch_probe_models.py` で 3 パターンの学習済み bundle を `output/models/cv_models/` に保存し、notebook 互換キャッシュ `perch_meta.parquet` / `perch_arrays.npz` も `output/models/` 直下に出力した
- Kaggle Dataset `suzukitaichi/birdclef-2026-perch-probe-cv-models` にアップロード済み。notebook 互換の root cache はそのまま置き、`cv_models/` は Kaggle 側では `cv_models.zip` として配布される
- `notebook.ipynb` は saved full-fit probe bundle を直接ロードして submit できる形に差し替えた。`SUBMIT_BACKEND=\"full_fit_probe\"` で `cv_models.zip` 内の `full_fit/probe_bundle.joblib` を使い、submit 時の再学習はしない
- Kaggle API 用に `scripts/experiment/prepare_kaggle_full_fit_kernel.py` を追加し、kernel slug `suzukitaichi/birdclef-2026-full-fit-probe-submit` を push 済み。現在 status は `RUNNING`
- `cv_models.zip` の archive 内 path が固定でないケースに備え、bundle loader は suffix 探索に修正した。Kernel version 2 を再 push 済み
- Kaggle 実行環境では dataset が `/kaggle/input/datasets/suzukitaichi/birdclef-2026-perch-probe-cv-models/cv_models` に見えるケースがあり、loader はこの mount path と `/kaggle/input` 全体探索 fallback に対応済み。Kernel version 3 を再 push 済み
- Kaggle 側 sklearn 互換性の差で `LogisticRegression.predict_proba()` が落ちるケースがあり、submit 推論は `coef_` / `intercept_` からの手計算に切り替えた。Kernel version 4 を再 push 済み
- ユーザー確認ベースで、Kaggle notebook `birdclef-2026-full-fit-probe-submit` の version 5 は score `0.891` を記録した
- `ashok205/tf-wheels` の output には `tensorflow-2.20.0-*.whl` と `tensorboard-2.20.0-*.whl` があり、Kaggle kernel source として attach できることを確認した
- `notebook.ipynb` は ONNX が見えているときは SavedModel を先に読まないようにし、`labels.csv` も `perch_v2.onnx` の同梱物を第一候補にするよう修正した。ローカル `data/archive (1)` と同じ 3 ファイル構成をそのまま input で扱える
- Kaggle kernel metadata には `ashok205/tf-wheels` を `kernel_sources` として追加した。Kernel version 6 を push 済み
- ローカル EDA 用に `scripts/audio_eda.py` を追加し、`train_audio` / `train_soundscapes` / `test_soundscapes` をブラウザ上で再生、波形、スペクトログラム、窓ラベル付きで見られるようにした
- EDA ツールに soundscape 窓の前後コンテキスト再生、species 出現数/rating 分布、周波数プロファイルと centroid / rolloff / flatness 分析、誤検知メモの JSONL 保存を追加した
- `scripts/export_species_distribution.py` で species 分布 SVG を `doc/overview/2026/species_distribution.svg` に保存できるようにし、現状の被覆は `train_audio 206 species / 35,549 recordings`、labeled soundscape は `75 species / 739 windows / 66 files` と可視化した

## 現在の判断
- `notebook.ipynb` は提出ベースラインとして維持する
- ただし local CV の正本は notebook の `59 full files / file GroupKFold` のみにはしない
- 実学習でも `raw Perch` 単体より `probe_pca_blend` が明確に強いので、Perch 系ベースラインはこの補正型 probe を新しい下限線に置く
- 直近の主ゲートは `66 files` 全量を使う `site-balanced file 3-fold` に置く
- notebook 互換の `file GroupKFold 5-fold` は継続し、回帰確認用の副ゲートとして残す
- 最高 AUC 自体は `all66_probe_pca_blend_filegkf5 = 0.891374` だが、site 分布を見ながら採否判定するため、主ゲートは引き続き `all66_probe_pca_blend_sitebalanced3 = 0.882219` にする
- 3 パターンの bundle と notebook 互換 cache はすでに `output/models/` と Kaggle Dataset に固定したので、以後の比較はこの artifact を基準にする
- notebook 経由の submit は full-fit probe bundle を読む形に統一し、Kaggle 実行では学習時間を test Perch 抽出のみに寄せる
- submit notebook は TF fallback を残しつつ、実運用は ONNX-first + wheel discovery に寄せる
- submit 実績としては、full-fit probe submit notebook の version 5 が `0.891` を出しているので、少なくとも提出経路は成立している
- 定量比較だけでなく、ラベルノイズ、音量差、soundscape 窓の妥当性を素早く監査するために、ローカル音声 EDA ツールを常設する
- EDA は「聞く・見る・数える・メモする」を同じ画面に寄せ、pseudo-label や fold 差分の原因を定性側から潰しやすくする
- class 偏りの定量把握は画面内だけでなく、静的な SVG として保存して long-tail 対応や sampling 方針の共有に使える形にする

## 検証方針
- 主ゲート:
  - `all66_probe_pca_blend_sitebalanced3`
  - 目的は site 分布を意識しつつ file leakage を防ぎ、fold ごとの class support を確保すること
- 副ゲート:
  - `all66_probe_pca_blend_filegkf5`
  - notebook の既存比較線を維持し、変更による退行を見やすくする
- ストレステスト:
  - `all66_probe_pca_blend_siteholdout3`
  - `S22` 偏重の強い site shift 条件で破綻しないかを見る
- raw baseline:
  - `full59_raw_perch_filegkf5`, `all66_raw_perch_filegkf5`
  - notebook 由来の Perch 単体線との差分を見る
- pure site holdout:
  - domain shift の上限確認としては残すが、S22 支配が強いため採否判定の単独根拠にはしない

## 次アクション
- Kaggle kernel `birdclef-2026-full-fit-probe-submit` の実行完了と `submission.csv` 生成を確認する
- Kernel version 6 で `ashok205/tf-wheels` attach と ONNX-first 解決が効いているかを確認する
- version 5 の `0.891` を基準に、version 6 以降で runtime 安定性とスコアの維持・改善を見る
- `scripts/experiment/perch_probe_cv.py` の split 定義を、notebook の ProtoSSM / MLP / ResidualSSM OOF にも適用する
- `probe_pca_blend` に prior や site/hour 後処理を足したときに `0.891374` を超えるか確認する
- prior、ResidualSSM、threshold 最適化を fold-safe に再構成し、本番 pipeline の local AUC を比較する
- EDA ツールに pseudo-label や model score の overlay を追加して、定性監査と local CV の往復を短くする
- 保存したアノテーションを class / site / recorder 観点で再集計し、系統的な誤検知パターンへつなげる
- 保存済み `species_distribution.svg` を見ながら rare class と soundscape-only class の扱いを整理し、sampling / loss weighting 仮説に落とす
