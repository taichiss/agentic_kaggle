# Architecture Decision Records

このディレクトリは、プラットフォーム全体または複数の実装へ影響する継続的な設計判断の
正本です。ADR は `NNNN-kebab-case-title.md` の連番で追加します。

## Index

| ADR | Status | Decision |
| --- | --- | --- |
| [0001](0001-adr-driven-agent-development.md) | Accepted | ADR と実行可能ハーネスを開発の中心にする |
| [0002](0002-competition-isolated-workspaces.md) | Accepted | コンペ固有資産を独立ワークスペースへ分離する |
| [0003](0003-reproducible-experiment-contract.md) | Accepted | 実験を設定・証拠・判断の単位で記録する |
| [0004](0004-trust-leaderboard-workflow.md) | Accepted | 軽量検証後に Leaderboard を主フィードバックとして使う |

## ADR Template

```markdown
# ADR NNNN: Title

- Status: Proposed
- Date: YYYY-MM-DD
- Decision owners: Maintainers
- Supersedes: None
- Superseded by: None

## Context

何が問題で、どの制約と選択肢があるか。

## Decision

何を選び、何を選ばなかったか。

## Consequences

利点、欠点、移行、対象外。

## Validation

決定が守られていることを、テスト・ハーネス・評価・運用でどう確認するか。
```

Accepted ADR の決定を変更する場合は本文の履歴を書き換えず、新しい ADR を追加して
`Supersedes` / `Superseded by` を更新します。
