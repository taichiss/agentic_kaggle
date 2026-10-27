# Current Strategy: Enveda CASMI 2026 - Molecule ID From Mass Spectra

## Competition Contract

- 指標: MRR@25、最大化。SMILES を順位順にセミコロン区切りで最大 25 件出力する。
- 照合: RDKit 2026.03.3 の tautomer canonicalization と InChIKey の先頭 14 文字。
- Notebook 提出、`submission.csv`、CPU / GPU 最大 9 時間、インターネット無効。
- 公式情報と未確認事項は [概要](../docs/overview/README.md) と [設定](../competition.toml) に記録する。

## Working Hypothesis

まず、公式評価と提出形式に沿った候補生成・順位付けの最小ベースラインを構築し、
検証データで MRR@25 と候補のカバレッジを測る。データを未確認のため、モデル、
特徴量、分割方法の選択は保留する。

## Validation Plan

1. 公式ファイルと sample submission の schema を確認し、行と ID の対応を固定する。
2. 評価手順と RDKit のバージョン・正規化の詳細を確認して、小さな合成例で検証する。
3. リークを防ぎ test 分布を反映する分割単位をデータ調査後に決める。
4. 学習・推論の前に最小例で候補の順位、出力形式、重複と無効値を確認する。
5. 各実験で設定、seed、分割、データ版、commit、結果を `experiments.md` に残す。

現時点でデータ調査、ベースライン、CV、LB 提出はいずれも未実施。
