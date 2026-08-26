# Current Strategy: {{TITLE}}

## Competition Contract

- slug: `{{SLUG}}`
- metric: `{{METRIC}}`
- direction: `{{METRIC_DIRECTION}}`
- submission schema: unverified
- critical constraints: unverified

## Confirmed Facts

- 公式情報で確認した事実だけを記載する。

## Open Questions

- [ ] 評価実装をローカルで再現できるか。
- [ ] CV 分割が test 分布と競技構造を反映するか。

## Working Hypotheses

| id | hypothesis | evidence | falsification | status |
| --- | --- | --- | --- | --- |
| H001 | baseline を定義する | none | CV と提出形式を再現できない | proposed |

## Priority Plan

1. データ schema と提出形式を検証する。
2. 最小の再現可能 baseline と CV を作る。
3. 誤差分析から変更軸を一つずつ試す。

## Validation Plan

- split:
- local metric:
- leakage checks:
- submission checks:

## Next Actions

- [ ] `competition.toml` の `unknown` を公式情報で更新する。
