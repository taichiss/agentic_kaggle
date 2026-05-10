---
description: Kaggle CLI でノートブックを bundle 化し kaggle kernels push で提出する。kernel-metadata.json 生成、依存 source attach、push 後の status 確認、submission 生成確認が必要なときに使う。
---

# Kaggle CLI Notebook Submit

## 目的

Kaggle Notebook の提出を、Kaggle CLI を使って再現可能に実行する。

## 手順

1. 対象 notebook と提出 slug を確認する。
2. 必要な bundle を作る。
   - notebook 本体を提出ディレクトリにコピーする。
   - `kernel-metadata.json` を生成する。
3. metadata に以下を入れる。
   - `id`: `owner/slug`
   - `code_file`: 置いた notebook 名
   - `kernel_type`: `notebook`
   - `language`: `python`
   - `enable_internet`: `false`
   - `competition_sources`
   - `dataset_sources`
   - `kernel_sources`
   - `model_sources`
4. 事前検証を通す。
   - notebook の JSON が壊れていないか確認する。
   - 依存や品質ゲートを必要に応じて実行する。
5. Kaggle CLI で push する。
   - `kaggle kernels push -p <bundle_dir>`
6. 状態確認をする。
   - `kaggle kernels status <owner>/<slug>`
7. 実行完了後に確認する。
   - `submission.csv` が生成されたか
   - 必要な source attach や mount path が正しいか
   - runtime が制約内か

## この repo での基準

- submit notebook は `ONNX-first` を基本にする。
- TF は fallback として残す。
- `ashok205/tf-wheels` は `kernel_sources` に載せる。
- Perch 入力は `labels.csv` を ONNX 同梱物から優先解決する。
- bundle 生成は `scripts/experiment/prepare_kaggle_full_fit_kernel.py` を優先する。

## 注意点

- Kaggle の attach 形は一意ではないので、dataset root だけでなく `/kaggle/input` 全体探索の fallback を考慮する。
- zip 内の path は固定とは限らないので、loader は suffix 探索に寄せる。
- cross-version 互換性が怪しい object は `predict_proba()` より属性ベースで推論する。
