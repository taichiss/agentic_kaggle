# TODO

## Phase 0.5: Perch + train_audio 適応学習 (今すぐ) — Perch 枝で LB `0.90+`

### 今やる
- ✅ `scripts/experiment/extract_train_audio_embeddings.py` — train_audio 5 秒 chunk cache 抽出器を実装
- ✅ `scripts/experiment/train_audio_mlp_head.py` — 234 クラス MLP head 学習と NumPy 推論 bundle 保存を実装
- ✅ `scripts/experiment/prepare_kaggle_ver8_kernel.py` / `run_ver8_train_audio_submit.py` — ver8 submit notebook bundle 生成と実行導線を実装
- ✅ 既存 single-chunk cache (`data/models/train_audio_perch_cache.npz`, `35,549` samples) で `train_audio_mlp_head_bundle.joblib` を生成
- ✅ Kaggle Dataset version を切って `train_audio_mlp_head_bundle.joblib` を upload
- ✅ ver8 kernel を push して submit 完走確認
- ✅ ver8 初回提出結果は `0.853`。single-chunk head の単純 blend は baseline 未満

### 次にやる
- full 5 秒 chunk cache を本番生成し、single-recording cache を差し替える
- `extract_train_audio_embeddings.py` の full-run 進捗確認方法を整える（現状は長時間無言になりやすい）
- `train_audio` warm start を低LR (`5e-5`, `1e-4`) で回し、stage1 を「高精度化」ではなく「初期化」に寄せる
- 同一 head を引き継いで `train_soundscapes` 継続学習 (`1e-5`, `2e-5`, `5e-5`) を回す
- `stage2_val_fraction=0.0` をやめ、site-aware / file-grouped OOF で採用判断する
- soundscape OOF から class-wise `alpha` を推定し、固定 `0.7/0.3` blend を廃止する
- no-primary 28 クラスと `Amphibia/Insecta` の AUC 差分を採用判定に入れる
- 既存パイプライン (ProtoSSM / ResidualSSM / prior) との統合
- CV (`all66_sitebalanced3`, `all66_filegkf5`) / LB 確認
- Kaggle kernel への MLP head 組み込み・提出

### 現時点の判断
- honest CV で改善が確認できるまで、継続学習 head は submit に混ぜない
- 直近 submit は `alpha=0` safe fallback で full-fit probe baseline を維持する

## Phase 1: SED 基盤構築 (1 週目) — SED baseline + honest CV

### 先行着手
- ✅ `src/data/dataset.py` — `train_audio` 20 秒 clip と `4 x 5 秒 Perch teacher logits` を同時に返すデータ層を追加
- ✅ `src/training/losses.py` / `src/training/trainer.py` — weak clip BCE と `teacher_mask` 付き distillation loss、および 2-stage 学習ループを実装
- ✅ `scripts/experiment/train_sed.py` — full 5 秒 teacher cache を読む SED CV 入口を実装
- ✅ `src/data/cv_splits.py` / `train_sed.py` — `site_holdout` stress split を共通 CV 層に接続
- 次: `train_sed.py` の最小 smoke run を作り、`sed_teacher_cv_summary.json` の生成確認まで進める

### 待ち (Phase 0.5 完了後)
- `pyproject.toml` に PyTorch/timm/torchaudio/openvino 依存追加 → `uv sync`
- GPU 環境を確定する

## Phase 2: Self-Training 構築 (2-3 週目) — LB `0.91+`

### 待ち
- `src/data/augmentations.py` — MixUp (ratio=1.0, weight=0.5), Sumix, FilterAugment, SpecAug
- `src/training/pseudo_label.py` — unlabeled soundscape pseudo-label 生成 + power transform + confidence filtering
- `src/training/self_training.py` — Noisy Student 型 self-training + WeightedRandomSampler
- Stochastic Depth (`drop_path_rate=0.15`) の SED モデルへの追加
- 2-4 回の iterative self-training (pseudo-label → retrain → re-label)
- backbone 追加: EfficientNet-B3/B4, NFNet-L0, RegNetY-016
- `src/inference/sed_inference.py` — sliding window + 重複平均化 (1D segmentation 方式)
- `src/inference/postprocess.py` — smoothing `[0.1, 0.2, 0.4, 0.2, 0.1]` + rank-aware file-level max

## Phase 3: 専用枝 + アンサンブル (3-4 週目) — LB `0.93+`

### 待ち
- `scripts/fetch_xeno_canto.py` — Amphibia/Insecta 追加データ取得 (~17k samples, ~700 species)
- `src/models/taxon_expert.py` — Amphibia/Insecta 専用 SED-B0 (拡張 species で学習)
- `src/inference/ensemble.py` — 多段モデル weighted average (異なる iteration × backbone)
- Perch probe を補助枝としてアンサンブルに追加
- delta shift TTA の実装

## Phase 4: 提出固め (4 週目) — 安全な最終提出

### 待ち
- `src/inference/export.py` — PyTorch → ONNX → OpenVINO (量子化なし)
- CPU 90 分以内の runtime 検証 + タイマー制御
- Mel キャッシュ共有 (全モデルで 1 回計算)
- ThreadPoolExecutor による並列推論
- submission.csv の shape / row_id 整合確認
- 最終 2 提出の選定 (Public Best + CV Best)
- Kaggle kernel 完走テスト

## 保留
- 大規模 XC pretraining (2025-2 位の勝因だが時間コスト高い。Phase 2 完了後に残り時間で判断)
- 外部 PAM soundscape を使う background augmentation
- ablation 整理と working note
