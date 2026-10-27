# EXP-0001: fold-safe baseline 再現 + 誤差分解 (Week 1 Phase D)

- **日付:** 2026-10-27
- **コード:** `scripts/casmi26_baseline_cv.py`, `src/casmi26/baseline/`, `src/casmi26/analysis/error_decomp.py`
- **元 kernel:** `haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline` (V17, LB 0.300–0.336)
- **seed:** 20260930 (fold0 の分子抽出とスペクトル抽出)
- **LB:** 未提出

## 何を再現したか

V17 の 4 チャネルのうち、リークなしで組めるものだけを移植した。

| V17 の部品 | 本実験 | 理由 |
|---|---|---|
| Channel 1 ライブラリ entropy 類似度 (±8.5 ppm) | 移植 (同一パラメータ) | — |
| Channel 2 analog 伝播 (±200 Da, sim^4 × Tanimoto) | 移植。Tanimoto は Morgan r2 2048-bit | 6,930-bit 複合指紋の学習コード非依存化 |
| Channel 3 MetFrag-lite (1–2 結合切断) | 移植 (同一) | — |
| Channel 4 FPNet | **不使用** | 公開重みは train 全体で学習済み → CV リーク (local_evidence.md §2) |
| HistGBM ランカー (`rank_train.npz`) | **不使用**。固定重み `lib + analog + 0.25 × frag` | 学習データが train 全体由来でリーク |
| 候補 pool COCONUT ∪ train 構造 | 移植。train 構造はホールドアウト identity group を `--pool-heldout` で制御 | Validation Protocol step 2 |

ライブラリと analog 代表スペクトルからは、ホールドアウト構造の identity group に属する全スペクトルを除外 (NP: 59,424 本、fold0: 約 51 万本)。

## 検証 split

| split | 分子 | スペクトル | 定義 |
|---|---:|---:|---|
| `np` | 250 | 1,128 | enveda-np-examples の全構造 (既知 adduct のみ)。天然物パネル 250 分子と同数 |
| `fold0` | 400 | 2,284 | identity fold 0 のうち timsTOF スペクトルを持つ構造から 400 抽出、1 分子最大 9 本 |

`pool=keep` はホールドアウト構造を候補 pool に残す (既知構造・reference スペクトルなし = Class 2 の模擬)。
`pool=drop` は pool からも除く。`coconut-casmi26-candidates` は **train 構造を除いた COCONUT** (train 277,566 構造との重複 0 件) なので、drop にするとほぼ全分子が Class 3 になる。天然物パネルでは 250 中 249 分子が実際の COCONUT に収録されているため、**hidden test に近いのは keep**。

## 結果

| 条件 | MRR@25 | top-1 | pool カバー率 | ranker 段の損失 | analog 単独 | frag 単独 | lib 単独 |
|---|---:|---:|---:|---:|---:|---:|---:|
| np / keep | **0.1944** | 0.072 | 0.996 | 0.802 | **0.535** | 0.245 | 0.053 |
| fold0 / keep | **0.0955** | 0.033 | 1.000 | 0.905 | 0.598 | **0.617** | 0.017 |
| np / drop | 0.0042 | 0.000 | 0.032 | 0.028 | 0.009 | 0.006 | 0.003 |
| fold0 / drop | 0.0003 | 0.000 | 0.003 | 0.002 | 0.000 | 0.000 | 0.000 |

各条件の詳細は `<条件>/report.md`、分子ごとの段別順位は `<条件>/per_molecule.parquet`。

## 解釈

1. **損失はほぼ全部ランカー段にある (keep 条件)。** pool カバー率は 99.6–100% で、候補窓 (median 53–88 件) に正解が入っている。MRR の取りこぼしは 0.80–0.90 がランカー段に計上された。
2. **固定重みの融合が単一チャネルより大幅に悪い。** analog 単独 0.535 → 融合 0.194 (np)、frag 単独 0.617 → 融合 0.096 (fold0)。
   原因はライブラリチャネル: 94–99% の分子にライブラリ一致 (類似度 median 約 0.6) があるが、identity-disjoint CV では正解の reference は必ず除外済みなので、一致先は常に train 側の別異性体。融合 top-1 の 66–76% がライブラリ top-1 と同じ候補になっており、正解を押しのけている。
   V17 が C1_HI=0.60 のゲートや `lvmax` 系特徴量付きの学習済みランカーを持つのはこのため。**この 0.19 / 0.10 は V17 の再現値ではなく、「学習済みランカーなしの 3 チャネル融合」の値**として扱う。
3. **drop 条件はデータセットの作り由来の最悪ケース。** Class 3 の割合を見積もるには、COCONUT 全体 (train 構造を含む) か PubChemLite の候補 pool が要る。

## 指標感度との整合

metric_sensitivity.md §4 の「rank 1–3 を直す変更を優先」と整合: keep 条件では rank 2–10 に 112 分子 (np) / 84 分子 (fold0) が溜まっており、ここを rank 1 に上げる変更 (ランカー) が最もレバレッジが高い。

## 次に試すこと

1. **学習済みランカーの導入 (roadmap #1 と同じ作業)。** fold-train 側から query を作り (ライブラリはさらにその query の identity group を除く nested 構成)、HistGBM pointwise と LightGBM LambdaRank を同じ特徴量で比較する。単一チャネルの結果から、少なくとも analog 単独の 0.535 (np) を超えるはず。
2. ライブラリ一致の扱い: `lib − lvmax`、`lvmax` などの相対特徴量を入れて、ゲートを学習させる。
3. Class 3 比率の見積もり: train 構造を含む COCONUT か PubChemLite pool で drop 条件を作り直す。
