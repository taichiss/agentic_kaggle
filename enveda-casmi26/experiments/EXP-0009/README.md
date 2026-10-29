# EXP-0009: PubChem 本体を候補 pool に入れ、文献量の prior で並べる

- **日付:** 2026-10-29
- **コード:**
  - 索引: `scripts/casmi26_build_pubchem_index.py` (完全版)、`scripts/casmi26_slim_pubchem_index.py` (Kaggle 用の軽量版)、`src/casmi26/baseline/pubchem.py`
  - CV: `scripts/casmi26_baseline_cv.py --extra-pool pubchemlite --pool-window corrected --pubchem-top 100 [--pubchem-index ...]`
  - 解析: `scripts/casmi26_pubchem_coverage.py`、`scripts/casmi26_rank_variants.py`
- **前提:** EXP-0008 で PubChemLite (38.6 万構造) を足しても LB はほぼ動かなかった (A 0.200 / B 0.248)。公開 kernel では、PubChem にしかない候補を外すと Public LB が 0.464 → 0.409 に下がる

## 何をしたか

1. **PubChem 本体の索引** (PubChem FTP の Extras: CID-Mass / CID-SMILES / CID-SID / CID-PMID、計 5.2 GB)
   - 荷電種と複数成分を除いた 100–1500 Da の中性構造 **1 億 1,040 万件**を質量順に並べた。作成 7 分、ピークメモリ 10 GB、ディスク 11 GB
   - 文献量の prior は公開 kernel と同じ 0.25 × (ln(1 + SID 件数) + ln(1 + PubMed 件数))
2. **候補への組み込み:** query ごとに窓 (EXP-0008 と同じ ±max(5 ppm, 2 mDa)、m/z のずれ補正) を引き、prior の上位 N = 100 件を EXP-0008 の pool に足す
3. **並べ方** (重みは決め打ち、検証データで調整していない): 各 query 内でスコアを標準化してから prior を足す
   - A + prior: z(固定融合) + prior
   - B + prior: z(analog) + prior
   - **S1: z(analog) + 0.5 × z(断片) + prior**
   - **S2: prior だけ** (スペクトルを使わない)
4. **Kaggle 用の軽量版索引:** 1 mDa ごとに prior の上位 400 件だけ残す (4,420 万件、4.0 GB)。窓の上位 100 件の prior の並びは、CV と配布 test の 1,050 窓中 1,047 窓で完全版と一致 (違う 3 窓も境目付近の構造だけ)。CV の数字も完全版とほぼ同じ (下表)

## 結果 1: 正解は PubChem の窓の中で文献量の何位か (パイプラインを回す前の確認)

| | NP (250 分子) | fold0 (drug-like、400 分子) |
|---|---:|---:|
| 窓内の PubChem 構造数 (中央値) | 4,160 | 8,434 |
| 正解が PubChem の窓にある | 99.6% | 98.0% |
| 正解が文献量 **1 位** | **65.6%** | 0.2% |
| 上位 10 件に入る | 95.6% | 0.5% |
| 上位 100 件に入る | 99.6% | 2.5% |
| 上位 1,000 件に入る | 99.6% | 23% |

NP の天然物は文献量だけで 66% が 1 位 (天然物パネルの「正解の 66% は同じ分子式の中で最も文献が多い」と一致)。fold0 の enveda-180 合成化合物は PubChem には載っているが文献がほとんどない

## 結果 2: CV (PubChem 上位 100 件を追加、軽量版索引)

| 並べ方 | NP Class 2 (keep) | NP Class 1 (other) | fold0 (keep) |
|---|---:|---:|---:|
| A 固定融合 | 0.167 | 0.917 | 0.104 |
| B analog 単独 | 0.478 | 0.603 | 0.561 |
| A + prior | 0.351 | **0.941** | 0.096 |
| B + prior | 0.724 | 0.791 | 0.554 |
| **S1: B + 断片 + prior** | 0.734 | 0.797 | **0.626** |
| **S2: prior だけ** | **0.778** | 0.778 | 0.003 |
| (参考) EXP-0008 の B (PubChem なし) | 0.521 | 0.632 | 0.604 |

- NP pool 外 (drop) では、正解が候補に入る割合が 96.8% → **100%**、B + prior は 0.721
- PubChem を足すだけ (prior なし) だと希釈で少し下がる (B: NP 0.521 → 0.478、fold0 0.604 → 0.561)。prior を足して初めて上がる
- 完全版索引での値: NP Class 2 の B + prior 0.721、S1 0.734、S2 0.778 / fold0 の S1 0.626 (軽量版とほぼ同じ)

## LB 提出 — 提出前の予測 (2026-10-29 記入、結果を見る前に固定)

| 提出 | kernel |
|---|---|
| S1: B + 断片 + prior (本命) | `yuitomurano/casmi26-exp0009-s1-analog-frag-prior` |
| S2: prior だけ (診断用) | `yuitomurano/casmi26-exp0009-s2-prior-only` |

**予測の作り方:** EXP-0002 のクラス構成 (Class 1 = 0.148、Class 2 = 0.285、残り 0.567 は pool 外) に NP の CV の値を当てはめる。この計算は EXP-0008 の 2 本を A 0.191 (実際 0.200)、B 0.242 (実際 0.248) と再現しており、pool 外の寄与はほぼ 0 だった

- S1 ≈ 0.148 × 0.797 + 0.285 × 0.734 = **0.327** + (PubChem で救える pool 外の分) 
- S2 ≈ 0.148 × 0.778 + 0.285 × 0.778 = **0.337** + (同上)

**幅:** NP ホールドアウトはよく研究された天然物で、文献量の prior に有利 (天然物パネルの README にも同じ注意)。hidden test の分子の文献量が少なければ、prior の効果は小さくなる。fold0 のような文献のない化合物ばかりなら S2 は 0 に近づく

- **S1 の予測: 0.25〜0.43** (中心 0.33)
- **S2 の予測: 0.05〜0.45** (中心 0.34)。幅が広いのは、S2 が「hidden test の分子がどれだけ文献に載っているか」をそのまま測るため

**判定:**
- S1 − 0.248 (EXP-0008 B) > +0.03 → PubChem + 文献量 prior の路線を採用する
- S2 / 0.337 を「hidden test の文献量が NP ホールドアウトの何割か」の目安にする。S2 ≥ 0.30 なら prior を主軸に、S2 < 0.15 なら prior は補助にとどめ、スペクトル側の改善を優先する
- S1 と S2 の大小で、スペクトル (analog・断片) が prior に上乗せできているかを見る (NP の CV では S2 > S1、fold0 では S1 ≫ S2)
