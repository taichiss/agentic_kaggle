# BirdCLEF Workspace

## Claude Code ユーザーへ
主要ルールは `CLAUDE.md`（Claude Code 用）と `.codex/codex.md`（Codex/Antigravity 用）の 2 か所に管理されている。
Claude Code を使う場合は `CLAUDE.md` を参照する。カスタムコマンド `/kaggle-submit` は `.claude/commands/kaggle-submit.md` に定義されている。

- ユーザー向けの返答は日本語で行う。
- Python 環境、依存追加、実行コマンドは `uv` を使う。
- コード変更の前に短い計画を明示し、1回の変更では1機能または1論点だけ進める。
- コンペ固有の作業前に `.codex/strategy/current.md` で `competition`, `metric`, `submission_format`, `constraints`, `available data` を確認する。
- 新しい事実、仮説、実験結果が出たら `.codex/strategy/current.md`, `.codex/strategy/experiments.md`, `.codex/strategy/todo.md` を更新する。

## Pointers
- Claude Code 主要ルール: `CLAUDE.md`
- Claude Code 設定: `.claude/settings.json`
- Claude Code PostToolUse フック: `.claude/hooks/post-tool-use.sh`
- ワークスペース運用ルール (Codex/Antigravity): `.codex/codex.md`
- 品質ゲートの実行口: `scripts/quality_gate.py`
- アーキテクチャ制約: `scripts/archgate.py`
- PostToolUse 整形フック (Codex): `.codex/hooks/post-tool-use.sh`
- pre-commit 設定: `.pre-commit-config.yaml`
- CI 設定: `.github/workflows/ci.yml`

## Commands
- Bootstrap: `bash scripts/bootstrap_quality.sh`
- Format: `uv run python scripts/quality_gate.py format`
- Lint: `uv run python scripts/quality_gate.py lint`
- Typecheck: `uv run python scripts/quality_gate.py typecheck`
- Architecture: `uv run python scripts/quality_gate.py arch`
- Tests: `uv run python scripts/quality_gate.py test`
- Full local gate: `uv run python scripts/quality_gate.py all`

## Don't
- 公式情報を会話だけで終わらせない。`doc/overview/` か `.codex/strategy/` に残す。
- 思いつきでトップレベルの新規ディレクトリを増やさない。`data/`, `doc/`, `src/`, `tests/`, `.codex/strategy/` を使う。
- 禁止レイヤーへ直接 import しない。外部I/Oや横断的関心事は `providers/` 経由に寄せる。
