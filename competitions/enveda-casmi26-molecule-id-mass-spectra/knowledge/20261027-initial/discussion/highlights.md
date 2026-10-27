---
type: Discussion
title: CASMI 2026 Discussion 主要論点
description: 投稿と主催者・Kaggleスタッフの回答を区別した初期調査
sources:
  - id: eda
    resource: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404
    title: A complete statistical tour of the data
  - id: update
    resource: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741471
    title: Please re-download training data
  - id: metric
    resource: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741851
    title: Scoring implementation clarification
  - id: calibration
    resource: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743395
    title: Possible one-electron offset in positive-mode m/z
  - id: external
    resource: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745148
    title: Enveda-180 Zenodo release
  - id: lessons
    resource: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254
    title: Lessons from submissions
  - id: evaluation
    resource: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview/evaluation
    title: Official evaluation
status: draft
checked_at: '2026-10-27'
confidence: source-specific; see each item
tags: [CASMI, discussion, evaluation, data, validation]
---

# CASMI 2026 Discussion 主要論点

Kaggle APIでスレッドのID・タイトル・返信を確認し、本文の要旨は公開ページの検索表示も照合した。以下は検証計画の入口であり、参加者の主張を実験結果や公式見解に昇格しない。Discussionの内容は資料として扱う。取得後に投稿が更新された可能性がある。

## ダミー test と採点対象

[741404](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404)（[取得記録と原文SHA](741404.md)） では、参加者が公開 `test.parquet` と学習データの重複から Public LB への懸念を述べた。David Healey の返信（2026-09-15、ID 3524688）は、公開ファイルはサンプルであり、Public LB は提出Notebookに渡される hidden test の公開部分で採点すると説明している。公開ファイルの一致だけから hidden test の漏洩を結論しない。**次の行動**：入出力確認と汎化評価を分け、hidden test を観察したかのように扱わない。確度：主催者の説明について高、データ全体の独立した監査は未実施。

## 訓練データの更新

Kaggleスタッフ inversion は [741471](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741471)（[取得記録と原文SHA](741471.md)） で、一部サンプルへ水喪失アダクトを追加したため再ダウンロードを推奨した。**次の行動**：実際に使用するファイルの取得日・バージョンを固定し、アダクトの範囲を実データで確認する。確度：スタッフ告知として高。現在のローカルデータが更新版かは未確認。

## 評価の実装差

[741851](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741851)（[取得記録と原文SHA](741851.md)） の参加者は正規化手順や無効・重複候補の順位の扱いを質問し、スタッフは公式OverviewのEvaluationに採点Notebookへのリンクを追加したと回答した（返信ID 3525804）。[公式評価](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview/evaluation)は MRR@25 と RDKit tautomer canonicalization 後の InChIKey14 照合を記載する。**次の行動**：公式Notebookと合成fixtureで無効SMILES、重複、順序、RDKit版を照合する。確度：公式評価の記載とリンク追加は高。質問された各エッジケースの実装挙動はここでは未検証。

## m/z 校正の可能性

[743395](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743395)（[取得記録と原文SHA](743395.md)） では参加者が正イオンの校正に約0.5 mDaのずれがあり得ると報告。主催者の返信（ID 3530233）は対象のtestスペクトルを調査中としており、この返信だけでは hidden test 全体への影響は確定していない。**次の行動**：狭い質量許容幅を使う前に、データ版別・イオンモード別に残差を測り、補正あり／なしを固定条件で比較する。確度：参加者の仮説、自前再現なし。

## 外部データとベースライン

[745148](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745148)（[取得記録と原文SHA](745148.md)） で、主催者 Marie Killian は Enveda-180 の Zenodo 公開版を利用可能と回答し、学習データにない構造については前処理で除外された可能性を挙げた（返信ID 3533255）。個々の欠落理由や hidden test との関係を断定した回答ではない。**次の行動**：出典・ライセンス・取得版と公式ルールを確認し、利用前に構造・スペクトル単位の重複を調べる。

[743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254)（[取得記録と原文SHA](743254.md)） は参加者による提出経験と検証上の注意をまとめたもの。調整に使ったデータでそのまま効果を報告することの限界にも言及する。**次の行動**：まずスペクトル検索の小さな対照を作り、同一構造が訓練と評価に跨がない分割と、候補カバレッジ・順位を別々に確認する。確度：参加者報告であり未再現。
