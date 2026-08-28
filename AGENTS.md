# Agent Instructions

このファイルは、エージェントが `agentic_kaggle` で作業するときの入口です。

## 目的

このリポジトリは、特定コンペに依存しない Kaggle 用 ADR 駆動開発プラットフォームです。
コンペ固有の事実、戦略、コード、評価は `competitions/<slug>/` に閉じ込めます。

## 作業開始時

1. `.agents/development-workflow.md` を読む。
2. `docs/adr/README.md` と関連 ADR を確認する。
3. 対象コンペの `competition.toml`、`strategy/current.md`、`strategy/experiments.md` を読む。
4. 公式情報、配布データ、再現済み結果、第三者情報を区別する。

対象コンペが未指定、または切り替わった場合は、古いコンペの前提を流用せず、コンペ名、
評価指標、提出形式、制約、利用可能データを先に確定します。

## コンペ別ポインター

- BioHub の nnU-Net / temporal backbone A/B、EXP-0007 の評価・最終化手順:
  [`competitions/biohub-cell-tracking-during-development/docs/overview/backbone-ab.md`](competitions/biohub-cell-tracking-during-development/docs/overview/backbone-ab.md)

## 必須ルール

- Python の環境構築と実行には `uv` を使う。
- 継続的な設計判断は `docs/adr/` に記録する。
- コンペ固有資産をプラットフォーム本体の `src/agentic_kaggle/` に入れない。
- 生データ、モデル重み、予測、提出物、API key、`kaggle.json`、`.env` をコミットしない。
- Notebook は探索用とし、再利用処理は対象コンペの `src/` へ移す。
- 実験には設定、seed、fold、データ版、commit、CV/LB、解釈、次の行動を残す。
- 長時間学習の前に小さいデータで入出力と評価経路を確認する。
- TRUST-LB 運用では重いローカル評価を必須にせず、明示された場合だけ `kaggle-lb submit` で
  提出し、取得した score を台帳へ記録する。
- 大きな変更より、小さく検証可能な変更を優先する。

## ADR が必要な変更

次の変更は原則として ADR を追加または置換します。

- リポジトリ全体または複数コンペへ影響する構成・依存・データ境界
- CV、評価、実験記録、提出検証の標準契約
- エージェントの継続的な作業ルールやハーネス判定
- 既存 Accepted ADR と異なる方針

単純なバグ修正、局所リファクタ、既存 ADR の実装だけなら新規 ADR は不要です。

## 検証

```bash
uv run ruff check .
uv run harness_check
uv run pytest
```

上記はプラットフォームの軽量検証です。コンペごとのフル CV や長時間テストは標準ゲートに
含めません。

実行していない検証を成功と記載しません。PR の作成・更新前は
`.agents/pull-request-guidelines.md` を確認します。
