import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import refresh


class RefreshTests(unittest.TestCase):
    def test_score_plots_keep_validation_cohorts_separate(self):
        groups = [
            {
                "id": "same-formula", "title": "Three models", "description": "Same candidates",
                "points": [
                    {"experiment_id": "J", "label": "JESTR", "split_id": "same-formula",
                     "submission_ref": "11"},
                    {"experiment_id": "M", "label": "MIST", "split_id": "same-formula",
                     "submission_ref": "12"},
                ],
            },
            {
                "id": "np-holdout", "title": "FPNet", "description": "Different candidates",
                "points": [
                    {"experiment_id": "F", "label": "FPNet", "split_id": "np-holdout",
                     "submission_ref": "13"},
                ],
            },
        ]
        cv = [
            {"experiment_id": exp, "split_id": split, "metric": "MRR@25", "value": value}
            for exp, split, value in (
                ("J", "same-formula", 0.62), ("M", "same-formula", 0.30),
                ("F", "np-holdout", 0.52),
            )
        ]
        submissions = {
            ref: {"experiment_id": exp, "public_score": value, "status": "COMPLETE"}
            for ref, exp, value in (
                ("11", "J", 0.476), ("12", "M", 0.478), ("13", "F", 0.429),
            )
        }
        result = refresh.resolve_score_plot_groups(groups, cv, submissions, "MRR@25")
        self.assertEqual([len(group["points"]) for group in result], [2, 1])
        self.assertEqual(result[0]["points"][1]["public_lb"], 0.478)
        self.assertEqual(result[1]["points"][0]["local_mrr"], 0.52)
        groups[1]["points"][0]["split_id"] = "same-formula"
        with self.assertRaisesRegex(ValueError, "group's split"):
            refresh.resolve_score_plot_groups(groups, cv, submissions, "MRR@25")

    def test_score_plot_requires_unique_cv_and_matching_submission(self):
        groups = [{"id": "split", "title": "Panel", "description": "Contract", "points": [
            {"experiment_id": "E", "label": "Model", "split_id": "split", "submission_ref": "42"}
        ]}]
        cv = [{"experiment_id": "E", "split_id": "split", "metric": "MRR@25", "value": 0.5}]
        with self.assertRaisesRegex(ValueError, "not unique"):
            refresh.resolve_score_plot_groups(
                groups, cv + cv, {"42": {"experiment_id": "E"}}, "MRR@25"
            )
        with self.assertRaisesRegex(ValueError, "mismatched"):
            refresh.resolve_score_plot_groups(groups, cv, {"42": {"experiment_id": "X"}}, "MRR@25")

    def test_score_plot_allows_one_submission_on_two_validation_panels(self):
        groups = [
            {"id": split, "title": split, "description": "Distinct validation cohort",
             "points": [{"experiment_id": "E", "label": "Model", "split_id": split,
                         "submission_ref": "42"}]}
            for split in ("synthetic-1000", "parent-proxy-80")
        ]
        cv = [
            {"experiment_id": "E", "split_id": split, "metric": "MRR@25", "value": value}
            for split, value in (("synthetic-1000", 0.39), ("parent-proxy-80", 0.35))
        ]
        submissions = {"42": {"experiment_id": "E", "public_score": 0.479,
                              "status": "COMPLETE"}}
        resolved = refresh.resolve_score_plot_groups(groups, cv, submissions, "MRR@25")
        self.assertEqual([g["points"][0]["local_mrr"] for g in resolved], [0.39, 0.35])
        self.assertEqual([g["points"][0]["public_lb"] for g in resolved], [0.479, 0.479])

        groups[0]["points"].append(groups[0]["points"][0].copy())
        with self.assertRaisesRegex(ValueError, "repeated score plot point"):
            refresh.resolve_score_plot_groups(groups, cv, submissions, "MRR@25")

    def test_initial_metadata_pairs_require_matching_experiment_and_split(self):
        pairs = [{"experiment_id": "E", "label": "CCS", "submission_ref": "42", "split_id": "full"}]
        cv = [{"experiment_id": "E", "split_id": "full", "metric": "MRR@25", "value": 0.091,
               "comparison_kind": "approximate"}]
        submissions = {"42": {"experiment_id": "E", "public_score": 0.155}}
        result = refresh.resolve_initial_metadata_experiments(pairs, cv, submissions, "MRR@25")
        self.assertEqual((result[0]["local_mrr"], result[0]["public_lb"]), (0.091, 0.155))
        self.assertEqual(result[0]["comparison_kind"], "approximate")
        with self.assertRaisesRegex(ValueError, "not unique"):
            refresh.resolve_initial_metadata_experiments(pairs, cv + cv, submissions, "MRR@25")
        with self.assertRaisesRegex(ValueError, "mismatched"):
            refresh.resolve_initial_metadata_experiments(
                pairs, cv, {"42": {"experiment_id": "X"}}, "MRR@25"
            )

    def test_newer_timeout_does_not_hide_score(self):
        rows = [
            {
                "ref": "42", "experiment_id": "E", "public_score": "0.477",
                "recorded_at": "2026-01-01",
            },
            {
                "ref": "42", "experiment_id": "E", "public_score": None,
                "recorded_at": "2026-01-02",
            },
        ]
        self.assertEqual(refresh.latest_submissions(rows)["42"]["public_score"], 0.477)

    def test_conflicting_submission_identity_fails(self):
        with self.assertRaises(ValueError):
            refresh.latest_submissions([
                {"ref": 1, "experiment_id": "A"},
                {"ref": 1, "experiment_id": "B"},
            ])

    def test_new_table_row_is_visible_without_erasing_curated_summary(self):
        curated = [{"id": "EXP-0001", "title": "reviewed", "result": "R", "limitation": "L"}]
        text = (
            "| id | date | hypothesis | data version | commit | config | seed/fold | "
            "CV | LB | artifact/run | takeaway | next |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| EXP-0001 | date | old | data | commit | config | fold | cv | lb | "
            "artifact | note | next |\n"
            "| EXP-9999 | date | new | data | commit | config | fold | 0.3 | 0.4 | "
            "artifact | note | next |"
        )
        result = refresh.add_uncurated_experiments(copy.deepcopy(curated), text)
        self.assertEqual([r["id"] for r in result], ["EXP-0001", "EXP-9999"])
        self.assertEqual(result[0]["title"], "reviewed")
        self.assertIn("CV: 0.3 / LB: 0.4", result[1]["result"])

    def test_detail_table_experiment_rows_do_not_parse_as_main_ledger(self):
        header = (
            "| id | date | hypothesis | data version | commit | config | seed/fold | "
            "CV | LB | artifact/run | takeaway | next |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        main_row = (
            "| EXP-9999 | date | main | data | commit | config | fold | 0.3 | 0.4 | "
            "artifact | note | next |\n"
        )
        detail = (
            "\n| 実験 | ランダム/難候補 | MRR@25 |\n"
            "| --- | --- | ---: |\n"
            "| EXP-0022-mist-random8 | 8/0 | 0.316447 |\n"
        )
        result = refresh.add_uncurated_experiments([], header + main_row + detail)
        self.assertEqual([row["id"] for row in result], ["EXP-9999"])
        with self.assertRaisesRegex(ValueError, "Malformed experiment table row"):
            refresh.add_uncurated_experiments([], header + "| EXP-broken | only |\n")

    def test_build_pairs_only_explicit_matching_split_and_hides_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "metric": "MRR@25", "history": [], "hypotheses": [{
                    "experiment_id": "E", "submission_ref": "42", "split_id": "common",
                    "local_cv_ref": {
                        "experiment_id": "E", "split_id": "common", "metric": "MRR@25",
                    },
                }],
            }
            (root / "dashboard-curated.json").write_text(json.dumps(source), encoding="utf-8")
            (root / "experiments.md").write_text("", encoding="utf-8")
            (root / "cv-results.jsonl").write_text(
                json.dumps({
                    "experiment_id": "E", "split_id": "common", "metric": "MRR@25",
                    "value": 0.35, "private_path": "/secret",
                }), encoding="utf-8",
            )
            (root / "lb-submissions.jsonl").write_text(
                json.dumps({
                    "ref": 42, "experiment_id": "E", "public_score": "0.477",
                    "file": "/secret",
                }), encoding="utf-8",
            )
            with patch.object(refresh, "STRATEGY", root):
                result = refresh.build()
                self.assertEqual(result["hypotheses"][0]["local_mrr"], 0.35)
                self.assertEqual(result["hypotheses"][0]["public_lb"], 0.477)
                self.assertNotIn("private_path", result["local_cv_ledger"][0])
                self.assertNotIn("file", result["submissions"][0])
                source["hypotheses"][0]["local_cv_ref"]["split_id"] = "other"
                (root / "dashboard-curated.json").write_text(json.dumps(source), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "matching split"):
                    refresh.build()
                source["hypotheses"][0]["local_cv_ref"]["split_id"] = "common"
                source["hypotheses"][0]["local_cv_ref"]["experiment_id"] = "different"
                (root / "dashboard-curated.json").write_text(json.dumps(source), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "matching split"):
                    refresh.build()


if __name__ == "__main__":
    unittest.main()
