# BirdCLEF+ 2026 戦略要約

## 1. 目的
このファイルは、`doc/discussion`、`doc/kernel`、`doc/solution` の調査結果を一枚に集約し、残り 1 か月で何に集中するかを明確にするための統合戦略である。

## 2. 2026 データの重要事実
- 提出対象は 234 クラス、評価は macro ROC-AUC
- `train_audio` に primary があるのは 206 クラス、primary が無いのは 28 クラス
- この 28 クラスは完全未学習ではなく、全て `train_soundscapes_labels.csv` に出ている
- `train_soundscapes_labels.csv` は `1478` 行だが、実質は `739` ユニーク窓
- labeled soundscape は `66` ファイル、`9` サイトのみで少ない
- unlabeled soundscape は `10,592` 本あり、ここが最大の未活用資産
- `train_audio` は `Aves` 偏重だが、soundscape では `Amphibia` と `Insecta` の比重が大きい

## 3. discussion からの要点

### 3.1 データ面
- soundscape ラベルは重複除去が必須
- `primary_label` 列は実質マルチラベルで、`;` 分割をしないと誤読する
- 28 no-primary クラスは soundscape 学習でしか回収できない
- labeled soundscape はサイト数が少ないので、site-aware でない CV は信用しにくい
- 一部の train_audio には極端に短い音声、重複音声、再取得した方が良いファイルがある

### 3.2 モデル面
- `Perch` は baseline として強い
- ただし `Perch + ProtoSSM` の微調整だけでは public LB `0.928-0.933` 付近で飽和しやすい
- さらに上を狙うには unlabeled soundscape を pseudo-label で使う方向が有力
- `Perch` 蒸留 SED は軽量 CNN の精度底上げに有効

### 3.3 推論面
- CPU `90` 分制約のため高速化は必須
- ONNX / OpenVINO / TorchScript のいずれかで提出経路を早めに固定する必要がある

## 4. kernel からの要点

### 4.1 主流は 2 系統
- 系統A: `Perch/ONNX + ProtoSSM + metadata prior + MLP probe + threshold/postprocess`
- 系統B: `CNN/SED + Perch distillation + pseudo-label/self-training`

### 4.2 系統Aの評価
- 強い baseline になりやすい
- file-level scaling、event/texture smoothing、site-hour prior などの後処理は整っている
- ただし discussion でも plateau 認識が強く、ここを掘り続ける期待値は高くない
- 現在の手元 baseline `notebook.ipynb` はこの系統で、ユーザー報告 public LB `0.930`
- よって、当面の基準線としては採用価値が高い
- ただし notebook 内 OOF は `filename GroupKFold` が中心で、site-aware 制約や full-pipeline fold-safe prior までは未反映

### 4.3 系統Bの評価
- 学習コストは増えるが、伸び代が大きい
- `Perch` を教師として使えるので、いきなり完全 scratch にならない
- 2025 の winning pattern と整合する

## 5. 過去 3 年の上位解法から再利用すべきもの

### 5.1 2023
- まずデータを正しくする
- SED が強い
- クラス不均衡対策、重複除去、推論最適化は優先度が高い

### 5.2 2024
- soundscape ドメイン適応が勝敗を分けた
- シンプルな EfficientNet 系 backbone でも十分戦える
- smoothing、TTA、rank-aware 系の後処理は依然として有効

### 5.3 2025（最重要参照年）
- iterative noisy-student 型 self-training が最重要
- 20 秒チャンクが Amphibia / Insecta に効いた
- 専用分類群モデルが macro AUC の底上げに効いた

### 5.4 年度横断の金メダル共通勝因

| 要素 | 2023 | 2024 | 2025 | 重要度 |
| --- | --- | --- | --- | --- |
| SED ヘッド | ✅ 全員 | ✅ 大半 | ✅ 全員 | ★★★ |
| Iterative Pseudo-labeling | △ | ✅ | ✅ 最大勝因 | ★★★ |
| Soundscape ドメイン適応 | △ | ✅ | ✅ | ★★★ |
| 軽量 CNN (EfficientNet) | ✅ | ✅ | ✅ | ★★★ |
| 推論最適化 (OpenVINO) | ✅ | ✅ | ✅ | ★★★ |
| 後処理 (smoothing, rank-aware) | ✅ | ✅ | ✅ | ★★☆ |
| 分類群専用モデル | △ | △ | ✅ 1位 | ★★☆ |
| 大規模 XC Pretraining | △ | △ | ✅ 2位 | ★★☆ |
| 20 秒チャンク | ✗ | ✗ | ✅ 1位 | ★★☆ |
| MixUp + Stochastic Depth | △ | △ | ✅ 1位 | ★★☆ |

### 5.5 2025-1 位の具体的な手法パラメータ（最優先参照）
- **SED**: GEM freq pooling, repeated 3-channel mel
- **Mel**: (sample_rate=32000, n_mels=224, fmin=0, fmax=16000, n_fft=4096, hop=1252, top_db=80)
- **チャンク**: 20 秒 → Image size = (3, 224, 512)
- **学習**: epochs=15, CE loss, lr=5e-4→1e-6, AdamW (wd=1e-4), CosineAnnealing (restart=5ep), MixUp p=0.5
- **Self-training**: MixUp ratio=1.0 (全 batch), Stochastic Depth drop_path=0.15, epochs=25-35
- **Pseudo-label**: power transform (>1) で noise 除去, WeightedRandomSampler (sum of max probs)
- **反復**: 4 回で飽和 (power: 1→0.65→0.55→0.6)
- **専用枝**: Amphibia/Insecta 専用 SED-B0 (XC ~17k samples, ~700 species), bs=128, epochs=40
- **推論**: sliding window + 重複平均化, smoothing [0.1,0.2,0.4,0.2,0.1], delta shift TTA
- **アンサンブル**: 7 モデル (異なる iteration × 異なる backbone), equal weight が Private 最良
- **最適化**: OpenVINO (量子化なし), multiprocess audio loading, mel 共有

## 6. 今回の主戦略

### 主軸
- `20 秒 SED CNN` を主学習枝にする
- `train_audio + labeled soundscapes` で教師モデルを作る
- その教師で unlabeled soundscapes を pseudo-label 化し、2-4 回の iterative self-training を回す

### 直近の基準線
- 直近の作業基盤は `notebook.ipynb` に置く
- これは `Perch ONNX + prior + MLP probe + LightProtoSSM + ResidualSSM + 後処理` の実用構成で、提出経路がすでに通っている
- ただし CV 戦略だけはそのまま採用しない。`filename` grouping を最低ラインとして残しつつ、site-aware split と fold-safe prior に強化する

### 補助枝
- `Perch` は捨てない
- 使い方は baseline、蒸留教師、補助アンサンブル、prior 的な特徴抽出の 4 役

### 専用枝
- `Amphibia/Insecta` 専用モデルを別に持つ
- 28 no-primary クラス対策と、soundscape 分布の偏り補正をここで回収する

### あまり時間を使わないもの
- ProtoSSM 系の細かい微調整だけを延々回すこと
- taxonomy prior だけで no-primary クラスを解決しようとすること
- random CV を信じて重い sweep をすること

## 7. 4 フェーズ ロードマップ

### Phase 1: 基盤構築 (1 週目) — 目標: SED baseline LB `0.84-0.87`
- 依存追加 (PyTorch, timm, torchaudio, openvino)
- 実験設定 config, 評価 metrics, Dataset, Mel 変換
- SED model (GEM freq pooling + EfficientNet-B0)
- 学習ループ (CE, AdamW, CosineAnnealing, AMP)
- `train_audio` のみで SED-B0 5-fold supervised → LB 確認

### Phase 2: Self-Training (2-3 週目) — 目標: LB `0.91+`
- Augmentation (MixUp ratio=1.0, Sumix, SpecAug, FilterAugment)
- Pseudo-label (power transform, confidence filter, WeightedRandomSampler)
- Noisy Student self-training + Stochastic Depth
- 2-4 回反復、backbone 拡充
- Sliding window 推論 + 後処理 (smoothing, rank-aware)

### Phase 3: 専用枝 + アンサンブル (3-4 週目) — 目標: LB `0.93+`
- Amphibia/Insecta 専用 SED-B0 (XC 追加データ)
- 多段モデル weighted average
- Perch 補助枝、delta shift TTA

### Phase 4: 提出固め (4 週目) — 安全な最終提出
- OpenVINO export、CPU 90分検証
- Mel キャッシュ共有、並列推論
- 最終 2 提出選定 (Public Best + CV Best)

## 8. 実装 Gap 分析

### 実装済み
- `data/input/` — 競技データ
- `doc/solution/2023-2025/` — 過去解法 (summary + raw writeup + CSV 比較表)
- `.codex/strategy/` — 戦略文書・実験ログ・TODO
- `scripts/experiment/` — Perch probe CV, Kaggle kernel 生成
- `src/data/audio_catalog.py` — 音声カタログ
- `src/providers/` — 可視化・アノテーション
- `src/inference/audio_eda_server.py` — EDA ツール
- `notebook.ipynb` — Perch baseline 提出ノートブック (LB `0.891`)
- 品質ゲート / CI (ruff, mypy, pre-commit)

### 未実装（金メダルに必要）
| モジュール | 用途 | 参照元 |
| --- | --- | --- |
| `src/models/sed.py` | SED ヘッド + backbone 統合 | 2025 全上位 |
| `src/models/backbones.py` | timm wrapper | 2025 全上位 |
| `src/models/taxon_expert.py` | 分類群専用モデル | 2025-1位 |
| `src/data/dataset.py` | 20秒チャンク + マルチラベル | 2025-1位 |
| `src/data/mel_transform.py` | Mel 変換 | 2025-1位 |
| `src/data/augmentations.py` | MixUp, Sumix, SpecAug, Stochastic Depth | 2025-1/2/5位 |
| `src/data/cv_splits.py` | site-aware CV | 2026 固有 |
| `src/training/trainer.py` | 学習ループ | 全般 |
| `src/training/losses.py` | CE / Focal loss | 2025 全上位 |
| `src/training/pseudo_label.py` | pseudo-label 生成 | 2025-1/2位 |
| `src/training/self_training.py` | iterative noisy student | 2025-1位 |
| `src/inference/sed_inference.py` | sliding window 推論 | 2025-1位 |
| `src/inference/postprocess.py` | smoothing + rank-aware | 2023-2025 全上位 |
| `src/inference/export.py` | ONNX → OpenVINO | 2023-2025 全上位 |
| `src/inference/ensemble.py` | 多段アンサンブル | 2025-1位 |
| `src/core/config.py` | 実験設定 | 全般 |
| `src/core/metrics.py` | 評価指標 | 全般 |

## 9. 結論
残り 1 か月で最も期待値が高いのは、`Perch` を教師・補助枝として活かしつつ、主戦場を `20 秒 SED + unlabeled soundscape self-training` に移すことだ。  
今の段階では、追加の小手先後処理よりも、soundscape 学習量を増やす方がリターンが大きい。

金メダルに必要な実装の約 70% が未実装だが、2025 の上位解法が明確なレシピを示しているため、実装は定型的に進められる。最大のリスクは GPU 計算資源と 90 分推論制約であり、Phase 1 完了時に判断する。
