# Discussion取得とOKF生成

Kaggle公式Python SDKでこのコンペのDiscussion一覧、本文と全ページの返信を読み取り専用で取得します。返信は入れ子を保持し、件数を再帰的に確認します。生成するOKF 0.2の入口は `knowledge/<run-id>/index.md` です。第三者の投稿は未検証の主張として扱い、内容を自動で事実認定しません。

リポジトリルートで実行します。認証は標準のKaggle設定、または秘密を含む環境ファイルを使用します。

```bash
uv run --env-file ../.env python competitions/enveda-casmi26-molecule-id-mass-spectra/src/sync_discussions.py --run-id 20261027-initial
uv run python competitions/enveda-casmi26-molecule-id-mass-spectra/src/sync_discussions.py --run-id 20261027-initial --build-only
```

環境ファイルは環境に合わせて指定し、Gitへ追加しません。同じIDで取得を再実行すると、APIを全件再照会し、そのIDのsnapshotと生成済み索引を置換します。履歴を保持するには新しいIDを指定してください。`--build-only` は保存済みmanifestだけから生成します。個別トピックの失敗は記録し、次回の再実行で再取得します。個別取得内容が変わるので再実行時は人手で付けた要約を再確認してください。自動生成ファイル以外は削除しません。

生の一覧ページ、本文・返信JSONとmanifestは `data/knowledge/<run-id>/` に保存し、Git管理外です。追跡するMarkdownは原文URL、取得日時、SHA-256、件数、欠落を記録し、本文・返信を転載しません。終了コード1は一覧の終端・件数の不一致、API失敗、本文や返信の欠落など取得不完全を示します。エラー内容は秘密の露出を避け例外クラスのみ記録します。投稿や返信が取得中に変化するとページ間で重複・欠落し得るため、一覧の終端と公称件数、および各トピックの返信数を確認しても完全性を保証するものではありません。

SDKは省略された `commentCount` を0として扱うため、0はAPIの明示値と区別できません。返信本文の欠落は削除・非公開・APIでの省略のどれかを区別できず、該当IDを不足として記録します。
