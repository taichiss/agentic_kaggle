# {{TITLE}}

- Kaggle slug: `{{SLUG}}`
- Metric: `{{METRIC}}` (`{{METRIC_DIRECTION}}`)
- Official page: {{COMPETITION_URL}}

## Setup

1. `competition.toml` を公式情報で更新する。
2. 配布データを `data/input/` に配置する。
3. `docs/overview/` に評価、提出、ルール、データ schema の要約を残す。
4. `strategy/current.md` に最初の仮説と CV 計画を書く。

## Reproducible commands

コンペに合わせて次を具体的なコマンドへ置き換えてください。

```bash
# smoke test
uv run python competitions/{{SLUG}}/src/train.py --config competitions/{{SLUG}}/configs/baseline.toml --smoke

# train / evaluate
uv run python competitions/{{SLUG}}/src/train.py --config competitions/{{SLUG}}/configs/baseline.toml

# inference / submission validation
uv run python competitions/{{SLUG}}/src/infer.py --config competitions/{{SLUG}}/configs/baseline.toml
```

存在しないコマンドを実装済みとして扱わず、実装時にこの節とテストを同期してください。
