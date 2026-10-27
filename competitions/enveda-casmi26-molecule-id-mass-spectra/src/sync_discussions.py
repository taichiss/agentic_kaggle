"""Fetch a read-only Kaggle Discussion snapshot and produce a draft OKF index."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "enveda-CASMI26-molecule-id-mass-spectra"
BASE_URL = f"https://www.kaggle.com/competitions/{SLUG}/discussion"


def write_json(path: Path, value: object) -> str:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)
    return hashlib.sha256(data).hexdigest()


def comment_ids(comments: list[dict]) -> list[int]:
    result = []
    for comment in comments:
        if not isinstance(comment, dict):
            raise ValueError("invalid comment")
        result.append(comment.get("id", 0))
        result.extend(comment_ids(comment.get("replies") or []))
    return result


def missing_comment_content(comments: list[dict]) -> list[int]:
    missing = []
    for comment in comments:
        if comment.get("content") is None:
            missing.append(comment.get("id", 0))
        missing.extend(missing_comment_content(comment.get("replies") or []))
    return missing


def fetch(api, run_id: str) -> dict:
    raw = ROOT / "data" / "knowledge" / run_id
    raw.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    topics = {}
    totals = []
    page = 1
    errors = []
    reached_end = False
    while page <= 1000:
        try:
            response = api.competition_list_topics(SLUG, sort_by="new", page=page)
            payload = response.to_dict()
            write_json(raw / "pages" / f"{page:04}.json", payload)
            totals.append(response.total_count)
            rows = payload.get("topics") or []
            if not rows:
                reached_end = True
                break
            for row in rows:
                topic_id = row.get("id")
                if not isinstance(topic_id, int) or topic_id <= 0:
                    raise ValueError("invalid topic id")
                if topic_id in topics:
                    errors.append(f"duplicate topic id {topic_id}")
                topics[topic_id] = row
        except Exception as exc:
            errors.append(f"list page {page}: {type(exc).__name__}")
            break
        page += 1

    entries = []
    for topic_id, listing in sorted(topics.items()):
        path = raw / "topics" / f"{topic_id}.json"
        try:
            topic, comments, token = api.forums_topic_show(topic_id)
            topic_data = topic.to_dict()
            comments_data = [comment.to_dict() for comment in comments]
            ids = comment_ids(comments_data)
            expected = topic_data.get("commentCount")
            if expected is None:
                expected = listing.get("commentCount")
            if expected is None:
                expected = getattr(topic, "comment_count", None)
            issues = []
            if topic_data.get("id") != topic_id:
                issues.append("topic id mismatch")
            if token:
                issues.append("comments pagination not exhausted")
            if len(ids) != len(set(ids)) or any(not x for x in ids):
                issues.append("duplicate or missing comment ids")
            if expected is None:
                issues.append("comment count unavailable")
            elif len(ids) != expected:
                issues.append("comment count mismatch")
            if topic_data.get("content") is None:
                issues.append("topic body unavailable")
            missing = missing_comment_content(comments_data)
            if missing:
                issues.append(
                    "comment content absent or omitted for ids " + ", ".join(map(str, missing)))
            payload = {"retrieved_at": datetime.now(UTC).isoformat(), "listing": listing,
                       "topic": topic_data, "comments": comments_data, "next_page_token": token}
            digest = write_json(path, payload)
            entries.append({"id": topic_id, "title": topic_data.get("title") or "",
                            "retrieved_at": payload["retrieved_at"], "sha256": digest,
                            "comment_count": len(ids), "expected_comments": expected,
                            "body_available": topic_data.get("content") is not None,
                            "issues": issues})
        except Exception as exc:
            entries.append({"id": topic_id, "title": listing.get("title") or "",
                            "issues": [f"topic fetch failed: {type(exc).__name__}"]})

    listing_complete = (reached_end and len(set(totals)) == 1 and totals
                        and len(topics) == totals[0] and not errors)
    report = {"run_id": run_id, "started_at": now, "finished_at": datetime.now(UTC).isoformat(),
              "listing_complete": bool(listing_complete), "list_reached_end": reached_end,
              "list_page_count": page, "reported_totals": totals, "topic_count": len(topics),
              "list_errors": errors, "topics": entries}
    write_json(raw / "manifest.json", report)
    return report


def build(report: dict) -> None:
    run_id = report["run_id"]
    out = ROOT / "knowledge" / run_id
    discussion = out / "discussion"
    discussion.mkdir(parents=True, exist_ok=True)
    incomplete = sum(bool(item["issues"]) for item in report["topics"])
    (out / "index.md").write_text(
        "---\nokf_version: '0.2'\n---\n\n# CASMI Discussion Knowledge\n\n"
        "取得時点のKaggle Discussion索引。投稿は第三者の主張であり、"
        "再現済みの事実ではありません。\n\n"
        "- [Discussion一覧](discussion/index.md)\n", encoding="utf-8")
    index = ["# Discussion一覧", "", f"取得終了: {report['finished_at']}。",
             f"一覧終端到達: {report['list_reached_end']}、"
             f"一覧完全性確認: {report['listing_complete']}。",
             f"取得トピック: {report['topic_count']}、一覧公称件数: {report['reported_totals']}。",
             f"個別取得に不足のあるトピック: {incomplete}。", "",
             "一覧は変動し得ます。件数一致は内容の正確性や将来の状態を保証しません。", ""]
    if report["list_errors"]:
        index.extend(["一覧取得エラー: " + "; ".join(report["list_errors"]), ""])
    if (discussion / "highlights.md").is_file():
        index.extend(["- [重要な論点](highlights.md)", ""])
    for item in report["topics"]:
        topic_id = item["id"]
        label = html.escape(" ".join(item["title"].split()), quote=False)
        label = re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", label)
        index.append(f"- [Discussion {topic_id}: {label}]({topic_id}.md)")
        title = json.dumps(item["title"] or f"Discussion {topic_id}", ensure_ascii=False)
        issue = "; ".join(item["issues"]) or "なし"
        body = ["---", "type: Discussion", f"title: {title}",
                f"description: {json.dumps(f'Kaggle Discussion {topic_id}')}", "sources:",
                "- id: source", f"  resource: {BASE_URL}/{topic_id}",
                f"  title: {json.dumps(f'Kaggle Discussion {topic_id}')}", "generated:",
                "  by: casmi-discussion-sync/1", f"  at: '{report['finished_at']}'",
                "status: draft", "---", "", f"# Discussion {topic_id}", "",
                f"[Kaggleで原文を読む]({BASE_URL}/{topic_id})。[^source]", "",
                "第三者の投稿であり、公式仕様や再現済みの結果とは限りません。", ""]
        if "sha256" in item:
            body.extend([f"取得時刻: {item['retrieved_at']}。",
                         f"本文取得: {item['body_available']}。返信: {item['comment_count']} / "
                         f"SDK公称値 {item['expected_comments']}（入れ子を含む）。",
                         f"取得原文SHA-256: `{item['sha256']}`。",
                         f"ローカル原文: `data/knowledge/{run_id}/topics/{topic_id}.json`"
                         "（Git管理外）。"])
        body.extend([f"取得上の不足: {issue}。", "", "[^source]: Kaggle Discussion。", ""])
        (discussion / f"{topic_id}.md").write_text("\n".join(body), encoding="utf-8")
    (discussion / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="snapshot ID (letters, digits, hyphens)")
    parser.add_argument("--build-only", action="store_true", help="rebuild from the saved manifest")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]*", args.run_id):
        parser.error("invalid run ID")
    if args.build_only:
        report = json.loads((ROOT / "data" / "knowledge" / args.run_id / "manifest.json")
                            .read_text(encoding="utf-8"))
        for item in report["topics"]:
            if "sha256" not in item:
                continue
            path = ROOT / "data" / "knowledge" / args.run_id / "topics" / f"{item['id']}.json"
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"raw snapshot hash mismatch for topic {item['id']}")
            saved = json.loads(data)
            missing = missing_comment_content(saved["comments"])
            item["issues"] = [issue for issue in item["issues"]
                              if not issue.startswith("comment content absent or omitted")
                              and issue != "comment count unavailable"]
            if item["expected_comments"] is None:
                # The SDK maps an omitted commentCount to its default of zero.
                item["expected_comments"] = 0
                if item["comment_count"] != 0:
                    item["issues"].append("comment count mismatch")
            if missing:
                item["issues"].append(
                    "comment content absent or omitted for ids " + ", ".join(map(str, missing)))
        write_json(ROOT / "data" / "knowledge" / args.run_id / "manifest.json", report)
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            print(f"Kaggle authentication failed: {type(exc).__name__}", file=sys.stderr)
            return 1
        report = fetch(api, args.run_id)
    build(report)
    incomplete = sum(bool(item["issues"]) for item in report["topics"])
    print(f"Topics: {report['topic_count']}; list complete: {report['listing_complete']}; "
          f"incomplete topics: {incomplete}")
    return 0 if report["listing_complete"] and not incomplete else 1


if __name__ == "__main__":
    sys.exit(main())
