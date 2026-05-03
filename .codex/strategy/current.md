# 現在の戦略

## コンペ概要
- 競技名: BirdCLEF+ 2026
- 評価指標: 真陽性が存在しないクラスを除外する macro ROC-AUC
- 提出形式: `submission.csv`。各 `row_id` は 5 秒窓で、234 クラスすべての確率を出す
- 制約: Notebook 提出のみ、CPU 実行 `90` 分以内、インターネット禁止、GPU 提出は実質使えない、公開外部データと事前学習モデルは利用可

## 利用可能データ
- `train_audio`: 234 クラス中 206 クラスに primary ラベル付き focal 音声あり。ただし分布は `Aves` に大きく偏る
- `train_soundscapes`: 23 サイト、`10,658` 本の 1 分音声。うち `10,592` 本は未ラベルで、hidden test に最も近いドメイン
- `train_soundscapes_labels.csv`: `1,478` 行あるが実質 `739` ユニーク窓。ラベル付きファイルは `66` 本、サイトは `9`
- `taxonomy.csv`: 提出対象 234 クラス = `Aves 162`, `Amphibia 35`, `Insecta 28`, `Mammalia 8`, `Reptilia 1`

## 確認済み事項
- `train_soundscapes_labels.csv` はそのまま使えない。重複行を除くと `1478 -> 739`
- soundscape ラベル列の `primary_label` は実際にはマルチラベル文字列で、`1322/1478` 行に `;` が含まれる
- `train_audio` に primary が存在しない提出クラスは `28` だが、`28` クラスすべてが labeled soundscapes 側には出現する
- 上記 28 クラスの大半は `25` 個の insect sonotype と `3` 個の Amphibia で、末尾クラス対策には soundscape 教師信号が必須
- labeled soundscapes は `66` ファイル・`9` サイトしかないため、行単位 CV やランダム file CV はサイトリークを起こしやすい
- train/test soundscapes は同じ Pantanal 配備由来だが、録音日時は重ならない。hidden test には labeled train にないサイトも含まれる
- `train_audio` と soundscape では分類群バランスが大きく違う。primary 件数は `Aves 34799 / Amphibia 451 / Insecta 199 / Mammalia 99 / Reptilia 1`、一方 soundscape ラベル出現数は `Amphibia 4174 / Insecta 1136 / Aves 824 / Mammalia 84 / Reptilia 26`
- 公開 kernel は大きく `Perch/ONNX + ProtoSSM + prior + MLP probe` 系と、`CNN/SED + Perch distillation + pseudo-label/self-training` 系に分かれる
- discussion 上では、純粋な `Perch + ProtoSSM` 微調整は public LB `0.928-0.933` 付近で頭打ちになりやすい
- Distilled SED 系は同じ CNN バックボーンに対しておおむね `+0.02 LB` 級の改善余地を示している
- `notebook.ipynb` の現行提出設定は、`Perch ONNX + prior + MLP probe + LightProtoSSM + ResidualSSM + 後処理` 系の実用ベースラインで、ユーザー報告の public LB は `0.930`
- `notebook.ipynb` の CV は `filename` 単位 `GroupKFold` を使っており、ランダム split よりは正しいが、現状は `site-aware` ではない
- `notebook.ipynb` の学習/OOF は `59` 本の fully-labeled soundscape file（`708` windows）を中心に回しており、`66` labeled files 全部を使う設計ではない
- `notebook.ipynb` の `run_pipeline_oof()` は `ProtoSSM + MLP` までの OOF であり、prior、ResidualSSM、threshold 最適化まで含んだ完全提出再現 CV ではない
- `probe_pca_blend` の実学習 CV では、`all66_filegkf5=0.891374`、`all66_sitebalanced3=0.882219`、`all66_siteholdout3=0.857185` が出た
- `scripts/experiment/train_perch_probe_models.py` で 3 パターンの bundle を `output/models/cv_models/` に保存し、`perch_meta.parquet` / `perch_arrays.npz` も notebook 互換で `output/models/` に配置した
- Kaggle Dataset `suzukitaichi/birdclef-2026-perch-probe-cv-models` にアップロード済み。Kaggle 側では `cv_models/` は `cv_models.zip` として載る
- `notebook.ipynb` は saved full-fit probe bundle を直接読む submit 版に差し替え、submit 時の再学習を止めた
- `scripts/experiment/prepare_kaggle_full_fit_kernel.py` で Kaggle kernel bundle を生成できる。kernel `suzukitaichi/birdclef-2026-full-fit-probe-submit` は push 済みで status は `RUNNING`
- `cv_models.zip` の archive 内 path が固定でないケースに備え、bundle loader は suffix 探索に修正した。Kernel version 2 を再 push 済み
- Kaggle 実行環境では dataset が `/kaggle/input/datasets/suzukitaichi/birdclef-2026-perch-probe-cv-models/cv_models` に見えるケースがあり、loader はこの mount path と `/kaggle/input` 全体探索 fallback に対応済み。Kernel version 3 を再 push 済み
- Kaggle 側 sklearn 互換性の差で `LogisticRegression.predict_proba()` が落ちるケースがあり、submit 推論は `coef_` / `intercept_` からの手計算に切り替えた。Kernel version 4 を再 push 済み
- ユーザー確認ベースで、Kaggle notebook `birdclef-2026-full-fit-probe-submit` の version 5 は score `0.891` を記録した
- `ashok205/tf-wheels` の output には `tensorflow-2.20.0-*.whl` と `tensorboard-2.20.0-*.whl` があり、Kaggle kernel source として attach できることを確認した
- `notebook.ipynb` は ONNX が見えているときは SavedModel を先に読まず、`labels.csv` も `perch_v2.onnx` の同梱物を第一候補にするよう修正した。ローカル `data/archive (1)` と同じ 3 ファイル構成をそのまま input で扱える
- Kaggle kernel metadata には `ashok205/tf-wheels` を `kernel_sources` として追加した。Kernel version 6 を push 済み
- ver8 用に `output/models/train_audio_mlp_head_bundle.joblib` と `train_audio_mlp_head_meta.json` を生成した。初回 artifact は既存 cache `data/models/train_audio_perch_cache.npz` (`35,549` single-chunk samples) を使い、weighted BCE は `train=0.00490`, `val=0.00888`
- ver8 submit notebook bundle は `output/kaggle_kernel/ver8_train_audio_mlp_submit/` に生成済み。既存 full-fit probe bundle に `train_audio` 適応 MLP head を後段 blend する
- ユーザー確認ベースで、Kaggle notebook `birdclef-2026-full-fit-probe-ver8-submit` の version 2 は score `0.853` を記録した。既存 full-fit probe submit (`0.891`) を下回った
- `scripts/experiment/train_audio_mlp_head.py` は同一 MLP head を `stage1_train_audio` → `stage2_soundscape_*` の順で継続学習する実装になっている。一方で現行 default は `stage1_learning_rate=3e-4`, `stage2_learning_rate=1e-4`, `stage2_val_fraction=0.0` で、初回 ver8 は single-chunk cache + 固定比率 blend だったため target-domain 側の採用判断が弱かった
- `scripts/experiment/train_audio_probe_blend_models.py` を追加し、継続学習 head を fold-safe OOF で評価しつつ `probe + class-wise alpha` を作る経路を実装した
- honest CV (`main_all66_sitebalanced3`) では、single-recording cache ベースの継続学習 head はまだ baseline を超えていない。`stage1=4ep@1e-4 -> stage2=8ep@2e-5` で `probe=0.882219`, `head=0.667842`, `blend=0.869417`。`stage1=0, stage2=8ep@5e-5` でも `head=0.377423`, `blend=0.874181`
- したがって 2026-05-02 時点の submit 最良候補は依然として full-fit probe baseline (`0.891`) であり、継続学習 head は full 5 秒 chunk cache が入るまで採用しない。Kaggle 向け bundle は `alpha=0` の safe fallback にして baseline と同値の出力へ寄せる

## 未確定事項 → 解決済み（2026-05-02 過去解法横断分析）
- ✅ 主力枝 → `20 秒 SED CNN`。根拠: 2025 年 1-10 位全員が SED 採用、1 位は 20 秒チャンクで Amphibia/Insecta を回収
- ✅ pseudo-label 混合比 → batch 全量 (ratio=1.0) で MixUp。根拠: 2025-1 位が ratio 実験で 1.0 が最良と報告
- ✅ `Amphibia/Insecta` 専用枝 → 有効。根拠: 2025-1 位が +0.002-0.003 LB を報告、taxonomy prior 単体より回収力が高い
- ✅ 推論形式 → OpenVINO（量子化なし）。根拠: 2023-2025 の上位が一貫して OpenVINO を採用、ONNX より速い

## 残る未確定事項
- 大規模 XC pretraining の費用対効果（2025-2 位の勝因だが、時間コストが高い）
- PyTorch と TensorFlow (Perch) の共存による kernel サイズ・メモリ圧迫
- 2026 固有の labeled soundscape が 66 ファイル・9 サイトと極端に少ない中での CV 信頼性

## 統合判断
- 2023 の教訓: 奇抜なモデルより、データの正しさ、重複除去、クラス不均衡対策、推論最適化の方が効く
- 2024 の教訓: モデル複雑化より soundscape ドメイン適応が重要。軽量 backbone + pseudo-label + 後処理が強い
- 2025 の教訓: 終盤の差は、unlabeled soundscapes を使った iterative self-training、長めチャンク、分類群専用モデルで付く
- 2026 の現状判断: `Perch` は強い教師・補助枝だが、勝ち切るには soundscape 中心の self-training へ進む必要がある
- ただし、作業の起点としては `notebook.ipynb` を現行ベースラインに採用してよい。理由は、すでに public LB `0.930` を出しており、推論経路・キャッシュ・提出形が実用レベルで固まっているため
- 一方で CV は notebook そのままを最終形にしない。`filename GroupKFold` は最低ラインとして維持しつつ、site-aware split と fold-safe prior 構築を追加する
- 主ゲートは `all66_sitebalanced3`、副ゲートは `all66_filegkf5`、ストレステストは `all66_siteholdout3` で固定する

## 過去解法横断 — 金メダル共通勝因
| 要素 | 2023 | 2024 | 2025 | 2026 状態 |
| --- | --- | --- | --- | --- |
| SED ヘッド | ✅ 1-8 位 | ✅ 上位大半 | ✅ 1-10 位全員 | 🔴 未実装 |
| Iterative Pseudo-labeling | △ | ✅ 勝因 | ✅ 最大勝因 (1 位: 4 回反復) | 🔴 未実装 |
| Soundscape ドメイン適応 | △ | ✅ 勝因 | ✅ 必須 | 🟡 labeled のみ |
| 軽量 CNN backbone | ✅ EfficientNet | ✅ EfficientNet | ✅ EfficientNet + NFNet | 🔴 未実装 |
| 分類群専用モデル | △ | △ | ✅ 1 位: Amphibia/Insecta | 🔴 未実装 |
| 推論最適化 (OpenVINO) | ✅ 必須 | ✅ 必須 | ✅ 必須 | 🟡 ONNX のみ (Perch) |
| 後処理 (smoothing, rank-aware) | ✅ | ✅ | ✅ | 🟡 基本 smoothing のみ |
| 大規模 XC Pretraining | △ | △ | ✅ 2 位の勝因 | 🟡 要検討 |
| 20 秒チャンク | ✗ | ✗ | ✅ 1 位の勝因 | 🔴 未実装 |
| MixUp + Stochastic Depth | △ | △ | ✅ 1 位: +0.005 | 🔴 未実装 |

## 実装 Gap 分析
- 現状で実装済み: データ管理、Perch probe baseline (LB `0.891`)、EDA ツール、戦略文書、品質ゲート
- 金メダルに必要だが未実装のモジュール:
  - `src/models/` — SED ヘッド + timm backbone (EfficientNet-B0/B3/B4, NFNet-L0, RegNetY)
  - `src/data/dataset.py` — 20 秒チャンク Dataset + マルチラベル
  - `src/data/mel_transform.py` — Mel 変換 (n_mels=224, n_fft=4096, hop=1252)
  - `src/data/augmentations.py` — MixUp, Sumix, FilterAugment, SpecAug, Stochastic Depth
  - `src/training/` — 学習ループ, loss (CE/Focal), pseudo-label 生成, iterative self-training
  - `src/inference/sed_inference.py` — sliding window + 重複平均化
  - `src/inference/postprocess.py` — smoothing + rank-aware スケーリング
  - `src/inference/export.py` — PyTorch → ONNX → OpenVINO
  - `src/inference/ensemble.py` — 多段モデル weighted average
  - `src/models/taxon_expert.py` — Amphibia/Insecta 専用モデル
  - `src/core/config.py` — 実験設定 dataclass
  - `src/core/metrics.py` — macro ROC-AUC + per-taxon AUC
  - `src/data/cv_splits.py` — site-aware / file-grouped split（scripts に部分的に存在）
  - `pyproject.toml` — PyTorch, timm, torchaudio, openvino 等の依存追加

## 作業仮説
- 仮説1: 残り 1 か月で期待値が最も高いのは、`2025` 型の `20 秒 SED` を主軸にした iterative pseudo-labeling 路線であり、ProtoSSM の微調整深掘りではない
- 仮説2: `Perch` は捨てず、教師・特徴量・補助アンサンブル枝として使う。ただし主学習枝は export しやすい CNN/SED 学生モデルに寄せる
- 仮説3: `Amphibia/Insecta` 専用枝と soundscape 由来 pseudo label は、一般的な smoothing や taxonomy prior だけより macro AUC 回収力が高い
- 仮説4: 当面の実験基盤は `notebook.ipynb` を採用し、そこから CV の厳密化と unlabeled soundscape 活用を段階的に追加するのが最も効率的
- 仮説5: 2025-1 位の MixUp ratio=1.0 + power transform + WeightedRandomSampler は 2026 にもほぼそのまま適用できる。`10,592` 本の unlabeled soundscapes は 2025 より遥かに大きな pseudo-label 資源
- 仮説6: 異なる self-training iteration のモデルを混合するアンサンブルは shake-up 耐性に効く（2025-1 位の知見）
- 仮説7 (新): Perch frozen embedding を `train_audio` で BirdCLEF 2026 用に適応学習する MLP head は、現ベースライン (soundscape 708 窓のみ) に安全に上積みできる。特に Perch 未対応 31 種、soundscape 陽性が少ない希少種、Amphibia/Insecta/Mammalia に効く。ただし `train_audio` で強く学習しすぎるとドメインギャップ沼にはまるリスクがあるため、最終段は `train_soundscapes` でキャリブレーションする
- 仮説8 (新): Phase 0.5 の本質は `train_audio` head を別枝として足すことではなく、同一 head を低LRで `train_audio` warm start し、その重みを保持したまま `train_soundscapes` で継続学習した target-domain head を作ることにある。最終 blend は補助であり、採用判断は soundscape OOF を主に見る
- 仮説9 (新): `20 秒 SED-B0` へ Perch を最初に入れる形は入力融合ではなく `4 x 5 秒` teacher logits 蒸留が最も筋が良い。`train_audio` の弱ラベルは clip BCE、Perch は時間位置の補助教師として使い、主対象は `203 mapped classes` に置く

## 優先計画 (5 フェーズ)
### Phase 0.5: Perch + train_audio 適応学習 (今すぐ) — 目標: Perch 枝で LB `0.90+`
現ベースラインは `train_soundscapes` の 708 窓だけで probe を学習しており、`train_audio` の教師信号を使っていない。Perch を frozen 特徴抽出器として固定し、`train_audio` で BirdCLEF 234 クラス用の head を warm start し、その同じ重みを `train_soundscapes` で継続学習して target-domain 寄りの head に寄せる。ver8 初回の `0.853` は、single-chunk cache 由来 head を full-fit probe 後段に固定比率で足しただけで、継続学習と soundscape OOF 校正の扱いが弱かったとみなす。

#### 背景・根拠
- Perch の直接マッピングは 203/234 種で、残り 31 種は未対応または属レベル proxy
- `train_audio` には 206 クラスの focal 音声があり、特に `train_soundscapes` で陽性が少ない種の教師信号を補える
- Perch 本体を fine-tune するのではなく、frozen embedding から head を学習するのが安全 (ONNX 推論パスを壊さない)
- `train_audio` に primary が無い 28 クラスは stage1 だけでは学習できず、`train_soundscapes` 継続学習でしか回収できない
- ドメインギャップ (Xeno-canto vs Pantanal soundscape) を考慮すると、`train_audio` は事前適応、`train_soundscapes` は target-domain continuation / calibration と役割分担するのが自然

#### パイプライン原則
- `train_audio` の役割は broad supervision を入れる warm start であり、最終採用の主判定ではない
- `train_soundscapes` の役割は no-primary 28 クラスの注入、分類群バランス補正、target-domain 校正である
- global 固定比率 blend はやめ、class-wise `alpha` / `bias` / `temperature` を soundscape OOF から決める
- `stage2_val_fraction=0.0` の学習結果は artifact 生成には使えても、採用判断には使わない

#### ステップ
1. `train_audio` の full 5 秒 chunk cache を本番生成する
   - 既存 single-chunk cache は捨てず比較線として残すが、採用候補は full-chunk 版を主に使う
2. Stage A: `train_audio` で同一 MLP head を low-LR warm start する
   - 形は既存の `1536-d → LayerNorm → 512 → Dropout → 234` を維持
   - ラベルは `primary=1.0`, `secondary=0.7`
   - まずは `lr=5e-5` または `1e-4`, `3-6 epochs` を主軸にし、過学習よりも broad class separation を優先する
   - `train_audio` grouped val は健全性確認用に残すが、採用判定には使わない
3. Stage B: その同じ head を `train_soundscapes` で継続学習する
   - stage1 の重みを初期値にして optimizer state も含めて継続する
   - `lr=1e-5` から `5e-5` の lower LR を主軸にする
   - `all66` を基本セットにし、no-primary 28 クラス、`Amphibia/Insecta/Mammalia/Reptilia` を強めに扱う
   - この段階で `train_audio` には無い 28 クラスを head に注入する
4. Stage C: soundscape OOF から class-wise 校正を学習する
   - `Perch raw logits`, `probe bundle logits`, `continued head logits` を材料に `alpha`, `bias`, `temperature` をクラスごとに決める
   - 初期形は `final_logit_c = (1 - alpha_c) * probe_logit_c + alpha_c * head_logit_c + bias_c`
   - unmapped 31 種と no-primary 28 クラスは `alpha_c` 上限を高くし、OOF で悪化するクラスは `alpha_c=0` を許容する
5. Stage D: 既存パイプライン (ProtoSSM / ResidualSSM / prior) と統合し、CV / LB を確認する

#### 採用条件
- 主ゲート `all66_sitebalanced3` で現基準 `0.882219` を下回らない
- 副ゲート `all66_filegkf5` で現基準 `0.891374` を大きく崩さない
- public LB は既存 full-fit probe submit `0.891` を最低比較線とする
- no-primary 28 クラスと `Amphibia/Insecta` の per-class / per-taxon AUC が non-negative であること

#### 実験優先度
| # | 実験 | 期待効果 |
| --- | --- | --- |
| 1 | full 5 秒 chunk cache + low-LR `train_audio` warm start | single-chunk 由来のズレを抑える |
| 2 | 同一 head の `train_soundscapes` 継続学習 | no-primary / target-domain 適応 |
| 3 | soundscape OOF 由来の class-wise `alpha` / `bias` / `temperature` | fixed blend を置き換える |
| 4 | `Amphibia/Insecta/Mammalia/Reptilia` 重みの再調整 | macro AUC の底上げ |
| 5 | train_audio 疑似ラベル + train_soundscapes 自己学習 | さらに上積み |

#### 注意点
- `train_audio` で強く学習しすぎると「Xeno-canto には強いが soundscape に弱いモデル」になるため、stage1 は高精度化より warm start に徹する
- `train_soundscapes` は 66 files / 9 sites しかないため、stage2 の良否は random split ではなく site-aware / file-grouped OOF で見る
- no-primary 28 クラスは stage2 が唯一の教師信号なので、stage1 の loss だけで model selection しない
- current single-recording cache では honest CV が baseline 未満なので、良い AUC が出るまでは submit 側に head の寄与を入れない
- Perch 本体の fine-tune はこの段階ではやらない (ONNX パス・CPU 提出との整合性を維持)
- 推論時の追加コストは MLP forward のみで CPU 90 分制約には余裕がある

### Phase 1: SED 基盤構築 (1 週目) — 目標: SED baseline + honest CV
1. `pyproject.toml` に PyTorch/timm/torchaudio/openvino 追加
2. `src/core/config.py` — 実験設定 dataclass
3. `src/core/metrics.py` — macro ROC-AUC + per-taxon AUC
4. `src/data/dataset.py` — 20 秒 clip + マルチラベル + `4 x 5 秒 Perch` teacher target
5. `src/data/mel_transform.py` — Mel 変換 (n_mels=224, hop=1252)
6. `src/data/cv_splits.py` — site-aware / file-grouped / site-holdout split
7. `src/models/sed.py` + `backbones.py` — SED + EfficientNet-B0
8. `src/training/trainer.py` + `losses.py` — weak BCE + distill loss + AdamW + CosineAnnealing
9. `scripts/experiment/train_sed.py` — エントリポイント
10. `train_audio` のみで SED-B0 5-fold 学習 → CV / LB 確認（目標: LB `0.84-0.87`）

### Phase 2: Self-Training 構築 (2-3 週目) — 目標: LB `0.91+`
1. `src/data/augmentations.py` — MixUp (ratio=1.0), Sumix, FilterAugment, SpecAug
2. `src/training/pseudo_label.py` — unlabeled soundscape pseudo-label 生成 + power transform
3. `src/training/self_training.py` — Noisy Student 型 self-training + WeightedRandomSampler
4. Stochastic Depth (`drop_path_rate=0.15`) の追加
5. 2-4 回の iterative self-training
6. backbone 追加: EfficientNet-B3/B4, NFNet-L0, RegNetY-016
7. `src/inference/sed_inference.py` — sliding window + 重複平均化 (TTA 的)
8. `src/inference/postprocess.py` — smoothing `[0.1, 0.2, 0.4, 0.2, 0.1]` + rank-aware

### Phase 3: 専用枝 + アンサンブル (3-4 週目) — 目標: LB `0.93+`
1. XC から Amphibia/Insecta 追加データ取得（~17k samples, ~700 species）
2. `src/models/taxon_expert.py` — Amphibia/Insecta 専用 SED-B0
3. `src/inference/ensemble.py` — 多段モデル weighted average
4. Perch probe を補助枝としてアンサンブルに追加
5. 異なる self-training iteration のモデルを混合
6. delta shift TTA

### Phase 4: 提出固め (4 週目) — 目標: 安全な最終提出
1. `src/inference/export.py` — PyTorch → ONNX → OpenVINO
2. CPU 90 分以内の runtime 検証、タイマー制御
3. Mel キャッシュ共有（全モデルで 1 回計算）
4. ThreadPoolExecutor による並列推論
5. submission.csv の shape / row_id 整合確認
6. 最終 2 提出の選定（Public Best + CV Best）
7. Kaggle kernel 完走テスト

## 検証計画
- CV 方針:
  - 最低ラインは `notebook.ipynb` と同じ `filename` 単位 `GroupKFold`
  - 主ゲートはそれに site 制約を加えた labeled soundscape 窓の site-aware / file-grouped validation
  - `train_audio` 側の grouped CV は表現学習の健全性確認用であり、順位予測の主軸にはしない
  - prior、threshold、Residual 補正を使う場合は fold ごとに train 側だけで再構築する
  - 採用条件は offline macro AUC、分類群別差分、seed 安定性、public LB の小規模確認が一致すること
- オフライン指標:
  - 主指標は competition 準拠の macro ROC-AUC
  - 補助指標は per-class / per-taxon AUC、校正ずれ、pseudo-label 信頼度分布、seed 分散
- 提出前チェック:
  - CPU `90` 分以内
  - mel または embedding キャッシュの再利用
  - 提出候補ごとの設定固定
  - `submission.csv` の shape と `row_id` 整合確認

## 次アクション
- Phase 0.5 は「別枝追加」ではなく「low-LR `train_audio` warm start → lower-LR `train_soundscapes` 継続学習 → soundscape OOF 校正」の 3 段パイプラインとして進める
- 直近の実装/実験は full 5 秒 chunk cache 生成、low-LR sweep、stage2 validation 有効化、class-wise `alpha` 推定に絞る
- `perch_probe_cv.py` / `train_perch_probe_models.py` の CV インフラを再利用し、採用判定は `all66_sitebalanced3` と `all66_filegkf5` で行う
- Perch probe (LB `0.891`) は壊さず、OOF で改善が確認できるまで継続学習 head は submit に混ぜない
- Phase 1 (SED 基盤構築) は Phase 0.5 の比較線を安定化させた後に並行して進める。SED 側の共通 CV 入口は `site_holdout` stress split まで接続済みなので、次は最小 smoke run と summary artifact 確認へ進む
