# Competitions

コンペ固有の資産は `competitions/<competition-slug>/` に置きます。

手作業で空ディレクトリを作る代わりに、ルートから次を実行してください。

```bash
uv run kaggle-init <competition-slug> \
  --title "<competition title>" \
  --metric "<official metric>" \
  --metric-direction maximize \
  --competition-url "<official URL>"
```

生成された `competition.toml` を公式ルールとデータ説明で確認してから実験を始めます。
複数コンペに共通しそうな処理も、二つ以上で実利用されるまでは各コンペ内に置きます。
