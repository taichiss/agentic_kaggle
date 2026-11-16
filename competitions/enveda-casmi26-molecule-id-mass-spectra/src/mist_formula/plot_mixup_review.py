"""Render a compact, evidence-linked overview of the Mixup comparison."""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/derived/mist-mixup-20261116/mixup-review.svg"
CV = ROOT / "strategy/cv-results.jsonl"
LB = ROOT / "strategy/lb-submissions.jsonl"

EXPERIMENTS = [
    ("A", "EXP-0025-mist-12k"),
    ("M1", "EXP-20261116-mixup-m1"),
    ("M2", "EXP-20261116-mixup-m2"),
    ("M3", "EXP-20261116-mixup-m3"),
    ("Strict M1", "EXP-20261117-mixup-m1-ce-strict"),
    ("Weak M1", "EXP-20261117-mixup-m1-weak-085095"),
]
PANELS = [
    ("same-formula-1000-v1", "Fixed 1,000 structures / 25 candidates"),
    (
        "natural-sources-5425-variable-candidates-posthoc-v1",
        "Natural-source diagnostic / 5,425 groups",
    ),
    ("saved-0477-family-80-proxy", "Parent-rank proxy / 80 structures"),
]
COLORS = {
    "A": "#48566a",
    "M1": "#007f70",
    "M2": "#d1663f",
    "M3": "#a94e64",
    "Strict M1": "#0c5c94",
    "Weak M1": "#7455a3",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: int = 1) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
        f'y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'
    )


def label(x: float, y: float, value: str, size: int = 12, color: str = "#263548") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'fill="{color}" font-family="Arial,sans-serif">{html.escape(value)}</text>'
    )


def scaled(value: float, left: float, right: float, lo: float, hi: float) -> float:
    return left + (value - lo) / (hi - lo) * (right - left)


def main() -> None:
    cv = {(row["experiment_id"], row["split_id"]): row for row in load_jsonl(CV)}
    lb = {row["experiment_id"]: row for row in load_jsonl(LB)}
    width, height = 1540, 500
    blocks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Mixup local validation and public leaderboard comparison">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        label(26, 35, "MS/MS Mixup: validation and Public LB", 21),
        label(
            26,
            56,
            "Changes relative to A (0.479 Public LB); intervals are paired bootstrap 95%",
            12,
            "#596879",
        ),
    ]
    y0, dy = 121, 50
    for j, (split, title) in enumerate(PANELS):
        x0 = 26 + j * 377
        blocks.extend(
            [
                f'<rect x="{x0}" y="72" width="360" height="360" '
                'rx="10" fill="white" stroke="#dae1e9"/>',
                label(x0 + 14, 96, title, 13),
            ]
        )
        left, right = x0 + 87, x0 + 329
        lo, hi = -0.028, 0.014
        zero = scaled(0, left, right, lo, hi)
        blocks.append(line(zero, 108, zero, 407, "#9daabd", 1))
        for i, (name, exp) in enumerate(EXPERIMENTS):
            y = y0 + i * dy
            row = cv.get((exp, split))
            blocks.append(label(x0 + 13, y + 4, name, 12, COLORS[name]))
            if row is None:
                blocks.append(label(left, y + 4, "pending local", 11, "#8e9aa8"))
                continue
            a = cv.get(("EXP-0025-mist-12k", split))
            if a is None:
                continue
            delta = float(row["value"]) - float(a["value"])
            ci = row.get("paired_bootstrap_95pct_delta_vs_A")
            if ci is not None:
                x1 = scaled(float(ci[0]), left, right, lo, hi)
                x2 = scaled(float(ci[1]), left, right, lo, hi)
                blocks.extend(
                    [
                        line(x1, y, x2, y, COLORS[name], 2),
                        line(x1, y - 4, x1, y + 4, COLORS[name]),
                        line(x2, y - 4, x2, y + 4, COLORS[name]),
                    ]
                )
            blocks.append(
                f'<circle cx="{scaled(delta, left, right, lo, hi):.1f}" '
                f'cy="{y}" r="4.5" fill="{COLORS[name]}"/>'
            )
            blocks.append(label(left, y + 17, f"{delta:+.4f}", 10, "#596879"))
        blocks.append(label(left, 423, "MRR@25 difference from A", 11, "#596879"))

    x0 = 1157
    blocks.extend(
        [
            f'<rect x="{x0}" y="72" width="357" height="360" '
            'rx="10" fill="white" stroke="#dae1e9"/>',
            label(x0 + 14, 96, "Public LB / same 0.477 parent", 13),
        ]
    )
    left, right = x0 + 100, x0 + 326
    lo, hi = 0.474, 0.484
    base = scaled(0.479, left, right, lo, hi)
    blocks.append(line(base, 108, base, 407, "#9daabd", 1))
    for i, (name, exp) in enumerate(EXPERIMENTS):
        y = y0 + i * dy
        blocks.append(label(x0 + 13, y + 4, name, 12, COLORS[name]))
        row = lb.get(exp)
        score = None if row is None else row.get("public_score")
        if score is None:
            status = "not submitted" if row is None else "submitted / scoring pending"
            blocks.append(label(left, y + 4, status, 11, "#8e9aa8"))
            continue
        score = float(score)
        blocks.extend(
            [
                line(base, y, scaled(score, left, right, lo, hi), y, COLORS[name], 3),
                f'<circle cx="{scaled(score, left, right, lo, hi):.1f}" '
                f'cy="{y}" r="4.5" fill="{COLORS[name]}"/>',
                label(left, y + 17, f"{score:.3f}", 10, "#596879"),
            ]
        )
    blocks.append(label(left, 423, "Displayed score (0.001 precision)", 11, "#596879"))
    blocks.append(
        label(
            26,
            464,
            "The three local panels have different candidate sets and are not interchangeable; "
            "the natural-source panel was built after the first local results.",
            11,
            "#596879",
        )
    )
    blocks.append(
        label(
            26,
            481,
            "Original M1 and Weak M1 include mixed collision-energy acquisitions; "
            "the corrected same-condition M1 is reported separately.",
            11,
            "#596879",
        )
    )
    blocks.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(blocks) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
