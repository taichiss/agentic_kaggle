# ADR 0003: 再現可能な実験契約を採用する

- Status: Accepted
- Date: 2026-08-26
- Decision owners: Maintainers
- Supersedes: None
- Superseded by: None

## Context

Kaggle のスコア改善は、コードだけでなくデータ版、分割、seed、特徴量、学習環境、後処理、
提出生成に依存する。Notebook の出力やチャットだけでは、改善を再現できず、CV と LB の乖離や
リークの原因も追跡できない。すべての生成物を Git に入れることは容量とデータ規約上できない。

## Decision

- 実験前の設定はコンペの `configs/` に置き、再実行コマンドを README に記載する。
- 実験後の要約は `strategy/experiments.md` に一行追加し、必要なら詳細文書へリンクする。
- 最低限、experiment id、日付、仮説、データ版、commit、設定、seed/fold、CV、LB、artifact、
  解釈、次の行動を記録する。
- 大きな artifact は Git 管理外に置き、パス、checksum または外部 run id を記録する。
- 比較可能な実験では同じ split と metric 実装を使い、変更した軸を明記する。
- LB を確認していない場合は空欄ではなく `not_submitted` と記載する。

## Consequences

結果を再現しやすくなり、エージェントが失敗実験を重複して実行する可能性を下げられる。
artifact 自体は別途保管が必要で、完全再現にはデータ取得権限と計算環境が必要になる。
探索初期にも最低限の記録コストが発生する。

## Validation

- コンペテンプレートが必須列を持つ実験台帳を提供する。
- PR レビューで挙動変更と実験台帳の同期を確認する。
- 各コンペは必要に応じて `evals/` に metric、split、submission の回帰 fixture を追加する。
