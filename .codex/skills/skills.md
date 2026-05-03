name: your-skill-name
description: ここに「何をするスキルか」「いつ使うか」を具体的に書く
---

  # Skill Title

  ## 目的
  （ここから本文。命令形で書く）

  ## 手順
  1. ...
## GPU実行スキル（必須）
- GPU実行時は必ず同一環境でプリフライトを行い、結果を記録する。
- プリフライトがNGなら実行禁止（CPU実行に勝手にフォールバックしない）。
- 実行環境が不一致の疑いがある場合は、ユーザーのWSL bashでの実行を優先し、指示だけを出す。
- `torch.cuda.is_available()` が False の場合は、学習開始しない。

## 追加スキル
- `kaggle-cli-notebook-submit`: Kaggle CLI を使ったノートブック bundle 化、`kaggle kernels push`、status 確認、submission 生成確認の手順。
