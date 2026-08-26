# ADR 0002: コンペ固有資産を独立ワークスペースへ分離する

- Status: Accepted
- Date: 2026-08-26
- Decision owners: Maintainers
- Supersedes: None
- Superseded by: None

## Context

複数の Kaggle コンペを同じリポジトリで扱うと、評価指標、データ schema、外部データ規則、
CV、モデル、提出形式が混ざりやすい。過去コンペの戦略を無条件に流用すると、リークや無効な
提出、ルール違反につながる。一方、共通の開発規約や初期化処理を毎回コピーすると保守できない。

## Decision

- プラットフォーム本体を `src/agentic_kaggle/`、コンペ固有資産を
  `competitions/<competition-slug>/` に分離する。
- 各コンペは `competition.toml` を公式仕様の機械可読な入口とする。
- 知識は `docs/overview/`、`docs/discussion/`、`docs/kernel/` に出典種別で分ける。
- 戦略、設定、実装、テスト、評価 fixture、Notebook をコンペ内に対応付ける。
- 生データ、artifact、提出物はコンペ内に配置できるが Git 管理外とする。
- `templates/competition/` と `kaggle-init` で同じ最小構成を生成する。

## Consequences

コンペ切替時の前提混入を減らし、コンペ単位で削除・移植・レビューできる。共通化は複数
コンペで実利用された、コンペ知識を持たない処理に限定される。単一コンペだけの処理を早期に
プラットフォームへ昇格しないため、最初は重複が残る可能性がある。

## Validation

- `harness_check` が slug と manifest の一致、必須ディレクトリ、戦略文書を検査する。
- `kaggle-init` のテストがテンプレート展開と既存ディレクトリ保護を確認する。
- `.gitignore` の契約をハーネスで確認する。
