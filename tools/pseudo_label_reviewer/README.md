# 疑似ラベル検証ツール — 使い方マニュアル

> **BirdCLEF 2026** プロジェクト用の音声ベース疑似ラベル品質チェックアプリです。  
> ブラウザ上で疑似ラベル音声と参照音声を並べて聴き比べ、**○ / × / スキップ** で判定を記録します。

---

## クイックスタート

```bash
# リポジトリをクローン済みの場合
cd /home/dev/kaggle/BirdCLEF+2026

# サーバー起動（デフォルト設定で OK）
uv run python tools/pseudo_label_reviewer/server.py
```

ブラウザで **http://\<サーバーのIP\>:7890** を開く。

---

## 前提条件

| 項目 | 内容 |
|---|---|
| Python | 3.11 以上（`uv` 管理） |
| 音声データ | `data/input/BirdCLEF+ 2026/train_soundscapes/*.ogg` が存在すること |
| CSV | `data/PresudeLabel/insect_ia_primary_checklist_priority_top300.csv` が存在すること |

---

## セットアップ（初回のみ）

```bash
# 1. uv がない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 依存ライブラリのインストール
cd /path/to/BirdCLEF+2026
uv sync

# ※ fastapi / uvicorn が未インストールの場合
uv add fastapi "uvicorn[standard]"
```

---

## サーバーの起動

### デフォルト設定

```bash
uv run python tools/pseudo_label_reviewer/server.py
```

- **バインド**: `0.0.0.0:7890`（全インターフェース公開）
- **CSV**: `data/PresudeLabel/insect_ia_primary_checklist_priority_top300.csv`
- **音声**: `data/input/BirdCLEF+ 2026/train_soundscapes/`
- **判定保存先**: `output/pseudo_label_review/judgments.json`

### カスタム設定

```bash
uv run python tools/pseudo_label_reviewer/server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --csv data/PresudeLabel/my_checklist.csv \
  --audio-dir /mnt/data/soundscapes \
  --save-dir /mnt/results
```

### LAN での共有（WSL 環境の場合）

WSL の IP は起動時のログに表示されます（例: `Listening : http://0.0.0.0:7890`）。  
同一 LAN 内の他 PC からは **http://\<WSLマシンのLAN-IP\>:7890** でアクセスできます。

```bash
# WSL の LAN IP を確認
ip addr show eth0 | grep 'inet '
```

> **Windows ファイアウォールの注意**  
> Windows ファイアウォールでポート 7890 の受信を許可していない場合、  
> 他 PC からのアクセスがブロックされることがあります。  
> `設定 → Windows セキュリティ → ファイアウォール → 受信の規則` で追加してください。

---

## アプリの使い方

### 画面構成

```
┌──────────────────────────────────────────────────────┐
│ ヘッダー: ロゴ / Tier フィルタ / 判定数 / 進捗 / CSV出力│
├──────────────────┬───────────────────────────────────┤
│  疑似ラベル音声  │       参照音声（正解ラベル）        │
│                  │                                    │
│  ・メタ情報      │  ・正解窓数 / 季節×時間帯          │
│  ・スコアバー    │  ・参照ファイル一覧（クリックで選択）│
│  ・プレイヤー    │  ・プレイヤー                      │
│  ・【波形】      │  ・【波形 + 窓マーカー】            │
├──────────────────┴───────────────────────────────────┤
│  判定エリア: ○正しい / ×誤り / —スキップ   ←→ナビ  │
└──────────────────────────────────────────────────────┘
```

### 判定方法

| ボタン | 意味 |
|---|---|
| **○ 正しい** | 疑似ラベルが正確にその種を検出している |
| **× 誤り** | 疑似ラベルが誤っている（別の種・背景ノイズ等） |
| **— スキップ** | 判断が難しいため保留 |

判定すると**自動で次のレコードに移動**し、サーバーに保存されます。

### キーボードショートカット

| キー | 操作 |
|---|---|
| `O` | ○ 正しい |
| `X` | × 誤り |
| `S` | スキップ |
| `←` / `→` | 前のレコード / 次のレコード |
| `Space` | 疑似ラベル音声を再生/停止 |

### 波形の見方

- **青い波形（左）** → 疑似ラベル音声（確認対象）
- **緑の波形（右）** → 参照音声（正解ラベルが付いた録音）
- **黄色の縦線** → 参照音声中の正解ラベルが付いた5秒窓の位置
- **波形クリック** → その位置にジャンプして再生

### Tier フィルタ

ヘッダーの `Tier A` / `Tier B` などのチップをクリックすると、優先度別に絞り込めます。

| Tier | 意味 |
|---|---|
| **A** | IA強 / 汎用複数モデルが支持 → 採用可能性が高い |
| **B** | IA強 / 汎用モデルが否定 → 要確認 |
| **C** | IA強 / 汎用モデルが中立 → 要確認 |
| **D** | IA弱 → リジェクト候補 |

---

## 判定結果の確認・エクスポート

### サーバー上の JSON（リアルタイム保存）

```bash
cat output/pseudo_label_review/judgments.json
```

### CSV ダウンロード

ブラウザ右上の **📥 CSV出力** ボタン、または:

```
http://<IP>:7890/api/export-csv
```

出力形式:

| 列名 | 内容 |
|---|---|
| `row_index` | 元 CSV の行番号 |
| `pseudo_soundscape_file` | 疑似ラベルファイル名 |
| `label` | 推定種コード |
| `common_name` | 和名/英名 |
| `priority_tier` | 優先度 Tier |
| `judgment` | `correct` / `wrong` / `skip` / `""（未判定）` |
| `reason` | 推定根拠 |

---

## API エンドポイント一覧

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `GET /` | — | アプリ本体 HTML |
| `GET /api/rows` | — | CSV データ（JSON） |
| `GET /api/audio/{filename}` | — | 音声ファイルのストリーミング |
| `GET /api/audio-list` | — | 存在する音声ファイル名一覧 |
| `GET /api/judgments` | — | 保存済み判定データ取得 |
| `POST /api/judgments` | JSON | 判定データを保存 |
| `GET /api/export-csv` | — | 判定結果 CSV ダウンロード |
| `GET /api/status` | — | サーバー設定・パスの確認 |

---

## トラブルシューティング

### 「音声ファイルが見つかりません」

`/api/audio-list` にアクセスして、ファイル名一覧を確認してください。  
CSV の `pseudo_soundscape_file` 列のファイル名と一致しているか確認します。

### 「波形の読み込みに失敗」

- ブラウザのコンソール（F12）でエラー内容を確認
- サーバーが音声ファイルを配信できているか `/api/status` で確認

### ポートが使われている

```bash
# 別ポートで起動
uv run python tools/pseudo_label_reviewer/server.py --port 8080
```

### 判定が保存されない

```bash
# 保存ディレクトリの権限確認
ls -la output/pseudo_label_review/
```

---

## ファイル構成

```
tools/pseudo_label_reviewer/
├── server.py      # FastAPI サーバー（音声配信・判定保存）
├── index.html     # フロントエンド（SPA）
└── README.md      # このマニュアル
```

---

## 更新履歴

| 日付 | 変更 |
|---|---|
| 2026-05-31 | 初版（FastAPI サーバー化、波形表示、サーバー側判定保存） |
