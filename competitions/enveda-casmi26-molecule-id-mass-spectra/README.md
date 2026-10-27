# Enveda CASMI 2026 - Molecule ID From Mass Spectra

質量スペクトルから分子構造の候補を順位付けする Kaggle コンペの作業ディレクトリです。
[公式ページ](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)を
参照し、確認日と未確認事項を `competition.toml` と `docs/overview/` に記録します。

- 評価: MRR@25（大きいほど良い）。各分子の候補 SMILES を順位順にセミコロンで区切る。
- Code Competition: `submission.csv` を Notebook から生成。CPU / GPU とも最大 9 時間、提出 Notebook のインターネットは無効。
- 公開されている外部データと事前学習モデルの利用が許可される。詳細な条件は利用前に公式ルールで確認する。
- 締切: 2026-12-14 23:59 UTC。変更の可能性があるため公式ページで再確認する。

## 構成

- `competition.toml`: 公式仕様と未確認項目
- `docs/overview/`, `docs/discussion/`, `docs/kernel/`: 出典別の調査記録
- `strategy/`: 現在方針、TODO、実験台帳
- `configs/`, `src/`, `tests/`, `evals/`, `notebooks/`: 設定、コード、検証、探索
- `data/`, `artifacts/`, `submissions/`: ローカルデータと生成物。Git 管理外

## 最初の作業

公式データと sample submission を確認し、入力の schema、ID、出力列、行数と
評価上の候補の扱いを確定する。その後、小さなデータで入出力と評価経路を検証し、
再現可能なベースラインを作る。現時点で学習・推論コマンドは未実装。
実データ、重み、予測、提出物、認証情報はコミットしない。
