# ADR 0001: ADR 駆動と実行可能ハーネスを採用する

- Status: Accepted
- Date: 2026-08-26
- Decision owners: Maintainers
- Supersedes: None
- Superseded by: None

## Context

Kaggle 開発では、調査、EDA、CV、学習、推論、提出が短い周期で変化する。会話や単一の
戦略文書だけに判断を残すと、エージェントや担当者が変わったときに、現在の契約と一時的な
仮説の区別がつかない。文章指示だけではディレクトリ境界や記録漏れも機械的に検出できない。

一方、すべての実験判断を ADR にすると更新コストが高く、探索速度を損なう。永続的な設計と
コンペ内の短命な仮説を分ける必要がある。

## Decision

- リポジトリ全体、複数コンペ、継続的な開発契約へ影響する判断を ADR に記録する。
- `AGENTS.md` は短い入口とし、詳細フローは `.agents/`、設計判断は `docs/adr/` を正本とする。
- ADR の構造、index、コンペ境界を `harness_check` で検証し、CI で常時実行する。
- 単一コンペの短命な仮説と実験判断は、そのコンペの `strategy/` に記録する。
- Accepted ADR の判断変更は、新しい ADR で置き換えて履歴を保つ。

## Consequences

設計の理由、実装、検証方法を後から追跡でき、エージェントは作業開始時に現在の契約を復元
できる。規則の一部を自動検査できる一方、ADR index の更新と、設計変更か実験判断かの分類が
必要になる。局所的な実装詳細まで ADR 化しない判断も求められる。

## Validation

- `harness_check` が ADR の命名、metadata、必須セクション、index 登録を検査する。
- CI が `ruff`、`harness_check`、`pytest` を実行する。
- PR レビューで変更と関連 ADR、受入条件、検証結果の対応を確認する。
