#!/usr/bin/env python3
"""Rebuild the static CASMI dashboard from its curated metadata and ledgers.

Pairs are deliberately explicit: score pairs from unrelated splits are not
comparable. No submission files, private local paths, or raw data are published.
"""
from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRATEGY = ROOT / "strategy"
DIST = ROOT / "site-dashboard" / "dist"
SUBMISSION_FIELDS = (
    "experiment_id", "date", "description", "ref", "kernel", "kernel_version",
    "status", "public_score", "private_score", "recorded_at",
)
EXPERIMENT_TABLE_COLUMNS = (
    "id", "date", "hypothesis", "data version", "commit", "config",
    "seed/fold", "cv", "lb", "artifact/run", "takeaway", "next",
)


def jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(item)
    return rows


def score(value, context):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}: invalid score {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context}: non-finite score")
    return result


def recorded_order(value, ref):
    if not value:
        return float("-inf")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Submission {ref}: invalid recorded_at {value!r}") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.timestamp()


def latest_submissions(rows):
    latest = {}
    for row in rows:
        if not row.get("ref"):
            raise ValueError("Submission without ref")
        ref = str(row["ref"])
        clean = {key: row.get(key) for key in SUBMISSION_FIELDS}
        clean["ref"] = ref
        for key in ("public_score", "private_score"):
            clean[key] = score(clean[key], f"submission {ref} {key}")
        old = latest.get(ref)
        if old and clean["experiment_id"] != old["experiment_id"]:
            raise ValueError(f"Submission {ref} has conflicting experiment IDs")
        # A later timeout must not replace a known score.
        rank = (clean["public_score"] is not None, recorded_order(clean.get("recorded_at"), ref))
        if old is None or rank > (
            old["public_score"] is not None,
            recorded_order(old.get("recorded_at"), ref),
        ):
            latest[ref] = clean
    return latest


def add_uncurated_experiments(history, markdown):
    """Expose new experiment table rows; curated summaries retain precedence."""
    known = {part.strip() for item in history for part in item["id"].split("/")}
    in_main_table = False
    for line in markdown.splitlines():
        if not line.startswith("|"):
            in_main_table = False
            continue
        cols = [
            part.strip().replace(r"\|", "|")
            for part in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        ]
        if tuple(part.lower() for part in cols) == EXPERIMENT_TABLE_COLUMNS:
            in_main_table = True
            continue
        if not in_main_table:
            continue
        if all(re.fullmatch(r":?-{3,}:?", part) for part in cols):
            continue
        if not cols[0].startswith("EXP-"):
            in_main_table = False
            continue
        if len(cols) < len(EXPERIMENT_TABLE_COLUMNS):
            raise ValueError(f"Malformed experiment table row: {line[:120]}")
        exp_id = cols[0]
        if exp_id in known:
            continue
        history.append({"id": exp_id, "title": cols[2], "result": f"CV: {cols[7]} / LB: {cols[8]}",
                        "limitation": f"未レビューの台帳行。条件・解釈は原文参照。{cols[10]}",
                        "source": "strategy/experiments.md"})
        known.add(exp_id)
    # Some experiments are recorded only as detailed headings, not table rows.
    for match in re.finditer(r"^###\s+(EXP-[A-Za-z0-9-]+)\s*[:：]\s*(.+)$", markdown, re.MULTILINE):
        exp_id, title = match.groups()
        if exp_id not in known:
            history.append({"id": exp_id, "title": title.strip(), "result": "詳細は実験台帳を参照",
                            "limitation": "未レビューの台帳見出し。結果・条件は原文を確認。",
                            "source": "strategy/experiments.md"})
            known.add(exp_id)
    return history


def resolve_initial_metadata_experiments(pairs, cv, submissions, metric):
    """Join the historical ektarr ablation to its explicitly selected split."""
    result = []
    for pair in pairs:
        ref = str(pair["submission_ref"])
        submission = submissions.get(ref)
        if not submission or submission.get("experiment_id") != pair["experiment_id"]:
            raise ValueError(f"{ref}: missing or mismatched initial submission")
        matching = [
            row for row in cv
            if row.get("experiment_id") == pair["experiment_id"]
            and row.get("split_id") == pair["split_id"]
            and row.get("metric") == metric
        ]
        if len(matching) != 1:
            raise ValueError(f"{ref}: initial CV match is not unique")
        entry = dict(pair)
        entry["local_mrr"] = matching[0]["value"]
        entry["public_lb"] = submission["public_score"]
        entry["comparison_kind"] = matching[0].get("comparison_kind")
        result.append(entry)
    return result


def resolve_score_plot_groups(groups, cv, submissions, metric):
    """Resolve separately plotted validation cohorts from explicit ledger IDs."""
    result = []
    seen_groups = set()
    for group in groups:
        group_id = group.get("id")
        if (not group_id or group_id in seen_groups or not group.get("title")
                or not group.get("description")):
            raise ValueError(f"Invalid or repeated score plot group {group_id!r}")
        seen_groups.add(group_id)
        if not group.get("points"):
            raise ValueError(f"{group_id}: score plot group has no points")
        points = []
        seen_refs = set()
        for point in group["points"]:
            ref = str(point.get("submission_ref") or "")
            experiment_id = point.get("experiment_id")
            if not ref or ref in seen_refs or not experiment_id or not point.get("label"):
                raise ValueError(f"{group_id}: invalid or repeated score plot point {ref!r}")
            seen_refs.add(ref)
            if point.get("split_id") != group_id:
                raise ValueError(f"{ref}: score plot point must use its group's split")
            matching = [row for row in cv if row.get("experiment_id") == experiment_id
                        and row.get("split_id") == group_id and row.get("metric") == metric]
            if len(matching) != 1:
                raise ValueError(f"{ref}: score plot CV match is not unique")
            submission = submissions.get(ref)
            if not submission or submission.get("experiment_id") != experiment_id:
                raise ValueError(f"{ref}: missing or mismatched score plot submission")
            points.append({
                "experiment_id": experiment_id,
                "label": point["label"],
                "submission_ref": ref,
                "local_mrr": matching[0]["value"],
                "public_lb": submission["public_score"],
                "lb_status": submission.get("status"),
            })
        result.append({"id": group_id, "title": group["title"],
                       "description": group["description"], "points": points})
    return result


def build():
    curated = json.loads((STRATEGY / "dashboard-curated.json").read_text(encoding="utf-8"))
    cv = jsonl(STRATEGY / "cv-results.jsonl")
    for row in cv:
        for field in ("experiment_id", "split_id", "metric"):
            if not row.get(field):
                raise ValueError(f"Local CV entry missing {field}")
        row["value"] = score(row.get("value"), f"{row['experiment_id']} local CV")
    latest = latest_submissions(jsonl(STRATEGY / "lb-submissions.jsonl"))
    curated["score_plot_groups"] = resolve_score_plot_groups(
        curated.get("score_plot_groups", []), cv, latest, curated["metric"]
    )
    curated["initial_metadata_experiments"] = resolve_initial_metadata_experiments(
        curated.get("initial_metadata_experiments", []), cv, latest, curated["metric"]
    )
    hypotheses = curated["hypotheses"]
    if not hypotheses:
        raise ValueError("At least the baseline pair must be configured")
    seen = set()
    for pair in hypotheses:
        ref = str(pair["submission_ref"])
        if ref in seen:
            raise ValueError(f"Repeated dashboard submission reference {ref}")
        seen.add(ref)
        if not pair.get("split_id") or pair["split_id"] != hypotheses[0].get("split_id"):
            raise ValueError(f"{ref}: every plotted pair must use the baseline's local split")
        if "local_cv_ref" in pair:
            selector = pair["local_cv_ref"]
            if (
                selector.get("experiment_id") != pair["experiment_id"]
                or selector.get("split_id") != pair["split_id"]
                or selector.get("metric") != curated["metric"]
            ):
                raise ValueError(
                    f"{ref}: local_cv_ref must identify the experiment, "
                    "matching split and metric"
                )
            matches = [row for row in cv if all(row.get(k) == v for k, v in selector.items())]
            if len(matches) != 1:
                raise ValueError(
                    f"{ref}: local_cv_ref matched {len(matches)} rows, expected exactly one"
                )
            pair["local_mrr"] = score(matches[0].get("value"), f"{ref} local CV")
        pair["local_mrr"] = score(pair.get("local_mrr"), f"{ref} local MRR")
        if pair["local_mrr"] is None:
            raise ValueError(f"{ref}: local score missing")
        sub = latest.get(ref)
        if sub and sub.get("experiment_id") != pair["experiment_id"]:
            raise ValueError(f"{ref}: pair and submission experiment IDs differ")
        pair["public_lb"] = sub["public_score"] if sub else None
        pair["lb_status"] = (sub.get("status") or "unknown") if sub else "unknown"
        pair["lb_recorded_at"] = sub.get("recorded_at") if sub else None
    for pair in hypotheses:
        pair["local_delta"] = round(pair["local_mrr"] - hypotheses[0]["local_mrr"], 10)
    cv_fields = (
        "experiment_id", "split_id", "metric", "value", "comparison_kind",
        "comparison_note", "metrics_sha256",
    )
    curated["local_cv_ledger"] = [{key: row.get(key) for key in cv_fields} for row in cv]
    curated["submissions"] = sorted(
        latest.values(), key=lambda row: (row.get("date") or "", row["ref"])
    )
    curated["history"] = add_uncurated_experiments(
        curated["history"], (STRATEGY / "experiments.md").read_text(encoding="utf-8")
    )
    curated["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return curated


def main():
    data = build()
    # Keep the real Kaggle refs in the local ledgers; publish stable display aliases.
    refs = sorted({str(row["ref"]) for row in data["submissions"]})
    aliases = {ref: f"S-{index:03d}" for index, ref in enumerate(refs, 1)}
    data["submissions"] = [
        {key: value for key, value in row.items() if key not in {"kernel", "kernel_version"}}
        for row in data["submissions"]
    ]
    rendered = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    for ref, alias in aliases.items():
        rendered = re.sub(rf"(?<![A-Za-z0-9]){re.escape(ref)}(?![A-Za-z0-9])", alias, rendered)
    if any(ref in rendered for ref in refs):
        raise ValueError("Raw Kaggle submission reference remains in public dashboard")
    rendered += "\n"
    DIST.mkdir(parents=True, exist_ok=True)
    # Compute/validate all output before touching published files.
    (DIST / "history.json").write_text(rendered, encoding="utf-8")
    (DIST / "history.js").write_text(
        "window.CASMI_INVENTORY = " + rendered.rstrip() + ";\n", encoding="utf-8"
    )
    print(
        f"Dashboard refreshed: {len(data['history'])} experiments, "
        f"{len(data['local_cv_ledger'])} CV rows, "
        f"{len(data['submissions'])} submissions, "
        f"{len(data['hypotheses'])} pairs"
    )


if __name__ == "__main__":
    main()
