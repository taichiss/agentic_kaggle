import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "sync_discussions.py"
spec = importlib.util.spec_from_file_location("sync_discussions", SOURCE)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class Object:
    def __init__(self, value):
        self.value = value
        self.total_count = value.get("totalCount", 0)

    def to_dict(self):
        return self.value


class Api:
    def competition_list_topics(self, slug, sort_by, page):
        assert slug == sync.SLUG
        assert sort_by == "new"
        rows = [{"id": 42, "title": "a [link]", "commentCount": 2}] if page == 1 else []
        return Object({"topics": rows, "totalCount": 1})

    def forums_topic_show(self, topic_id):
        assert topic_id == 42
        topic = Object({"id": 42, "title": "a [link]", "content": "private source text",
                        "commentCount": 2})
        comments = [Object({"id": 11, "content": "source reply", "replies": [{"id": 12}]})]
        return topic, comments, ""


def test_snapshot_retains_nested_comments_and_reports_missing_content(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    report = sync.fetch(Api(), "run-1")
    assert report["listing_complete"]
    entry = report["topics"][0]
    assert entry["comment_count"] == 2
    assert entry["issues"] == ["comment content absent or omitted for ids 12"]
    raw = json.loads((tmp_path / "data/knowledge/run-1/topics/42.json").read_text())
    assert raw["comments"][0]["replies"][0]["id"] == 12
    sync.build(report)
    note = (tmp_path / "knowledge/run-1/discussion/42.md").read_text(encoding="utf-8")
    index = (tmp_path / "knowledge/run-1/discussion/index.md").read_text(encoding="utf-8")
    assert "private source text" not in note
    assert entry["sha256"] in note
    assert r"a \[link\]" in index


def test_list_failure_is_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "ROOT", tmp_path)

    class Broken(Api):
        def competition_list_topics(self, slug, sort_by, page):
            if page == 2:
                raise RuntimeError("secret should not be recorded")
            return super().competition_list_topics(slug, sort_by, page)

    report = sync.fetch(Broken(), "run-2")
    assert not report["listing_complete"]
    assert report["list_errors"] == ["list page 2: RuntimeError"]
    manifest = (tmp_path / "data/knowledge/run-2/manifest.json").read_text(encoding="utf-8")
    assert "secret should not be recorded" not in manifest


def test_sdk_default_zero_comment_count(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "ROOT", tmp_path)

    class NoReplies(Api):
        def forums_topic_show(self, topic_id):
            topic = Object({"id": topic_id, "title": "Zero", "content": "Text"})
            topic.comment_count = 0
            return topic, [], ""

    report = sync.fetch(NoReplies(), "run-3")
    assert report["topics"][0]["expected_comments"] == 2  # list says 2
    assert "comment count mismatch" in report["topics"][0]["issues"]

    class ZeroOnList(NoReplies):
        def competition_list_topics(self, slug, sort_by, page):
            rows = [{"id": 42, "title": "Zero"}] if page == 1 else []
            return Object({"topics": rows, "totalCount": 1})

    entry = sync.fetch(ZeroOnList(), "run-4")["topics"][0]
    assert entry["expected_comments"] == 0
    assert entry["issues"] == []


def test_invalid_comment_structure_rejected():
    with pytest.raises(ValueError):
        sync.comment_ids([None])
