#!/usr/bin/env python3
"""
BirdCLEF 2026 疑似ラベル検証サーバー
======================================
使い方:
    uv run python tools/pseudo_label_reviewer/server.py
    # または
    uv run python tools/pseudo_label_reviewer/server.py --port 7890 --host 0.0.0.0

オプション:
    --host         バインドアドレス (default: 0.0.0.0)
    --port         ポート番号      (default: 7890)
    --csv          CSVファイルパス (default: data/PresudeLabel/insect_ia_primary_checklist_priority_top300.csv)
    --audio-dir    音声フォルダ    (default: data/input/BirdCLEF+ 2026/train_soundscapes)
    --save-dir     判定保存フォルダ(default: output/pseudo_label_review)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── デフォルトパス ────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CSV = REPO_ROOT / "data/PresudeLabel/insect_ia_primary_checklist_priority_top300.csv"
DEFAULT_AUDIO_DIR = REPO_ROOT / "data/input/BirdCLEF+ 2026/train_soundscapes"
DEFAULT_SAVE_DIR = REPO_ROOT / "output/pseudo_label_review"

APP_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("reviewer")

# ── 設定（起動引数で上書き） ──────────────────────────────
class Config:
    csv_path: Path = DEFAULT_CSV
    audio_dir: Path = DEFAULT_AUDIO_DIR
    save_dir: Path = DEFAULT_SAVE_DIR

cfg = Config()

# ── アプリ ────────────────────────────────────────────────
app = FastAPI(title="BirdCLEF 2026 疑似ラベル検証", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CSV 読み込み ─────────────────────────────────────────────
def load_csv() -> list[dict]:
    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")
    rows = []
    with open(cfg.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("pseudo_soundscape_file"):
                rows.append(dict(r))
    return rows

# ── エンドポイント ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = APP_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/api/rows")
def get_rows():
    """CSVの全行をJSON配列で返す"""
    try:
        rows = load_csv()
        return JSONResponse({"rows": rows, "total": len(rows)})
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    """音声ファイルをストリーミング配信"""
    # パストラバーサル防止: ファイル名のみ許可
    safe_name = Path(filename).name
    path = cfg.audio_dir / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Audio not found: {safe_name}")
    return FileResponse(
        path=str(path),
        media_type="audio/ogg",
        headers={"Accept-Ranges": "bytes"},
    )

@app.get("/api/audio-list")
def audio_list():
    """存在する音声ファイル名一覧"""
    if not cfg.audio_dir.exists():
        return JSONResponse({"files": [], "error": f"audio_dir not found: {cfg.audio_dir}"})
    files = sorted(p.name for p in cfg.audio_dir.glob("*.ogg"))
    return JSONResponse({"files": files, "count": len(files)})

# ── 判定の保存・読み込み ──────────────────────────────────────
class JudgmentPayload(BaseModel):
    judgments: dict[str, str]  # row_index(str) → "correct"|"wrong"|"skip"

def _judgment_path() -> Path:
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    return cfg.save_dir / "judgments.json"

@app.get("/api/judgments")
def get_judgments():
    p = _judgment_path()
    if not p.exists():
        return JSONResponse({"judgments": {}})
    return JSONResponse(json.loads(p.read_text()))

@app.post("/api/judgments")
def save_judgments(payload: JudgmentPayload):
    p = _judgment_path()
    data = {"judgments": payload.judgments}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    log.info(f"Saved {len(payload.judgments)} judgments → {p}")
    return JSONResponse({"status": "ok", "saved": len(payload.judgments)})

@app.get("/api/export-csv")
def export_csv():
    """判定結果をCSVとしてダウンロード"""
    rows = load_csv()
    p = _judgment_path()
    judgments: dict[str, str] = {}
    if p.exists():
        judgments = json.loads(p.read_text()).get("judgments", {})

    lines = ["row_index,pseudo_soundscape_file,label,common_name,priority_tier,judgment,reason"]
    for i, r in enumerate(rows):
        j = judgments.get(str(i), "")
        reason = r.get("reason", "").replace('"', '""')
        lines.append(
            f'{i},"{r["pseudo_soundscape_file"]}","{r["label"]}","{r["common_name"]}",'
            f'"{r["priority_tier"]}","{j}","{reason}"'
        )
    content = "\n".join(lines)
    from fastapi.responses import Response
    return Response(
        content=content.encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pseudo_label_review_result.csv"},
    )

@app.get("/api/status")
def status():
    return {
        "csv": str(cfg.csv_path),
        "csv_exists": cfg.csv_path.exists(),
        "audio_dir": str(cfg.audio_dir),
        "audio_dir_exists": cfg.audio_dir.exists(),
        "save_dir": str(cfg.save_dir),
    }

# ── 起動 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="疑似ラベル検証サーバー")
    parser.add_argument("--host",      default="0.0.0.0", help="バインドアドレス (default: 0.0.0.0)")
    parser.add_argument("--port",      default=7890, type=int, help="ポート番号 (default: 7890)")
    parser.add_argument("--csv",       default=str(DEFAULT_CSV), help="CSVファイルパス")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="音声フォルダ")
    parser.add_argument("--save-dir",  default=str(DEFAULT_SAVE_DIR), help="判定保存フォルダ")
    args = parser.parse_args()

    cfg.csv_path  = Path(args.csv)
    cfg.audio_dir = Path(args.audio_dir)
    cfg.save_dir  = Path(args.save_dir)

    log.info(f"CSV       : {cfg.csv_path}")
    log.info(f"Audio dir : {cfg.audio_dir}")
    log.info(f"Save dir  : {cfg.save_dir}")
    log.info(f"Listening : http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
