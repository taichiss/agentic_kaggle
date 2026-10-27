# 公式情報の要約

確認日: 2026-10-27。出典: [Kaggle Competition Overview](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)。
このページの情報は変更され得るため、実験や提出の前に最新の公式情報を確認する。

- 質量スペクトルに対して候補となる分子構造を順位付けする。評価指標は MRR@25、方向は最大化。
- 提出列は `molecule_id` と `smiles`。各 molecule_id は一度だけ現れ、欠損や重複 ID は認められない。各分子の SMILES 候補は順位順にセミコロン区切りで最大 25 件記載し、25 件未満も許される。
- 照合には RDKit 2026.03.3 による tautomer canonicalization と InChIKey の先頭 14 文字を用いる。
- 提出は Notebook による `submission.csv`。CPU と GPU はそれぞれ最大 9 時間、提出 Notebook のインターネットは無効。
- 公開外部データと公開事前学習モデルが許可される。利用条件の詳細は [Rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules) を確認する。
- 締切は 2026-12-14 23:59 UTC。

## 未確認

sample submission の行数、スペクトルと分子の ID の対応、入力ファイル構造、
重複・不正 SMILES の扱い、外部資源の詳細条件、提出回数制限、
CV に適した分割単位。公式 Data、Rules と実データで確認してから実装へ反映する。
