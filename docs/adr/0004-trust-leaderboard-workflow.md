# ADR 0004: TRUST-LB の提出・スコア確認ワークフローを採用する

- Status: Accepted
- Date: 2026-08-26
- Decision owners: Maintainers
- Supersedes: None
- Superseded by: None

## Context

対象とする運用では、複雑なローカルテストや大規模 CV より Kaggle Leaderboard の結果を主な
フィードバックとして信頼する。手動提出だけに依存すると、異なるコンペへの誤提出、列や空
ファイルの事故、どの実験がどのスコアだったかの記録漏れが起きる。一方、Kaggle の提出回数は
有限であり、エージェントが暗黙に提出してはいけない。

## Decision

- `kaggle-lb submit` を、ローカル提出ファイルから LB score 取得までの標準入口にする。
- 提出前の標準ゲートは、ファイル存在、非空 CSV、manifest で宣言した列、認証と参加状態の
  確認に限定する。フル CV や重いモデルテストは必須にしない。
- submit 前後の submission 一覧を比較して Kaggle submission ref を特定し、完了、失敗、
  timeout まで polling する。
- `publicScore` を通常の反復フィードバックとする。`privateScore` が未公開なら欠損のまま扱う。
- 結果は `strategy/lb-submissions.jsonl` に追記し、実験 ID、commit、提出ファイルの SHA-256、
  submission ref、status、score を関連付ける。
- 実際の提出は、利用者またはエージェントが明示的に `kaggle-lb submit` を実行した場合だけ行う。

## Consequences

CLI だけで提出とスコア回収を完了でき、LB 中心の反復を短くできる。ローカル CV が保証する
リーク検出、分散推定、提出回数節約は弱くなるため、LB の shake-up と overfitting リスクを
受け入れる。提出前の軽量検査は予測品質を保証せず、Competition Rules の承諾は Kaggle Web
上で事前に必要になる。

## Validation

- subprocess を偽装した単体テストで、提出特定、完了、失敗、timeout、台帳追記を確認する。
- `harness_check` が manifest の metric、方向、submission 列設定を検査する。
- 実コンペへの E2E 提出は submission quota を消費するため CI では実行しない。
