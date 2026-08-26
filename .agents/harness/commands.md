# Harness Commands

標準の検証経路です。

```bash
uv run ruff check .
uv run harness_check
uv run pytest
```

新しいコンペを作る場合:

```bash
uv run kaggle-init <competition-slug> \
  --title "<display name>" \
  --metric "<official metric>" \
  --metric-direction maximize \
  --competition-url "https://www.kaggle.com/competitions/<competition-slug>"
```

学習、推論、提出検証のコマンドはコンペごとに異なるため、生成した
`competitions/<slug>/README.md` に固定します。

TRUST-LB の提出からスコア取得まで:

```bash
uv run kaggle-lb submit <competition-slug> \
  --file <submission.csv> \
  --experiment-id EXP-0001 \
  --message "<what changed>"
```

重いローカル CV は標準ゲートではありません。`kaggle-lb submit` は提出ファイルを軽量検査し、
Kaggle の scoring 完了まで polling して `strategy/lb-submissions.jsonl` に結果を追記します。
