# Codex Hooks

このディレクトリには、Kaggle 用の最速フィードバック層で使う実行スクリプトを置く。

## Layers
- `PostToolUse`: `.codex/hooks/post-tool-use.sh`
- `pre-commit`: `.pre-commit-config.yaml`
- `CI`: `.github/workflows/ci.yml`

## Intent
- フォーマットは最速層で自動修正する。
- lint / typecheck / architecture check は pre-commit に寄せる。
- テスト全体は CI に寄せる。

## Note
- ローカルの Codex CLI は `0.118.0` で、バイナリ文字列上は `pre-tool-use`, `post-tool-use`, `session-start`, `user-prompt-submit`, `stop` のイベント名を持つことを確認した。
- このリポジトリではグローバル `~/.codex/config.toml` を自動変更しない。
- この環境では実行権限の変更ができなかったため、PostToolUse では `bash .codex/hooks/post-tool-use.sh` として配線する。
- 配線後も、失敗時に作業を止めたくないためフックスクリプトは常に `0` で終了する。
