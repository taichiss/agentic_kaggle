from __future__ import annotations

import unittest

from scripts.experiment.cv_soundscape_validation import (
    DEFAULT_DATA_DIR,
    active_class_count,
    default_experiments,
    evaluate_experiment,
    filter_fully_labeled_rows,
    folds_for_experiment,
    load_primary_labels,
    load_soundscape_rows,
    rows_for_dataset,
)


class CvSoundscapeValidationTest(unittest.TestCase):
    def test_dataset_counts_match_notebook_and_expanded_view(self) -> None:
        all_rows = load_soundscape_rows(DEFAULT_DATA_DIR)
        full_rows = filter_fully_labeled_rows(all_rows)

        self.assertEqual(len(all_rows), 739)
        self.assertEqual(len({row.filename for row in all_rows}), 66)
        self.assertEqual(len({row.site for row in all_rows}), 9)
        self.assertEqual(active_class_count(all_rows), 75)

        self.assertEqual(len(full_rows), 708)
        self.assertEqual(len({row.filename for row in full_rows}), 59)
        self.assertEqual(len({row.site for row in full_rows}), 8)
        self.assertEqual(active_class_count(full_rows), 71)

    def test_site_balanced_split_has_no_file_leakage(self) -> None:
        all_rows = load_soundscape_rows(DEFAULT_DATA_DIR)
        folds = folds_for_experiment(all_rows, splitter="site_balanced_file", n_splits=3)

        seen_validation_files: set[str] = set()
        all_indices = set()
        for train_indices, val_indices in folds:
            self.assertTrue(val_indices)
            self.assertTrue(set(train_indices).isdisjoint(val_indices))
            val_files = {all_rows[index].filename for index in val_indices}
            self.assertTrue(seen_validation_files.isdisjoint(val_files))
            seen_validation_files.update(val_files)
            all_indices.update(val_indices)

        self.assertEqual(all_indices, set(range(len(all_rows))))

    def test_site_balanced_proxy_gate_improves_over_notebook_proxy(self) -> None:
        labels = load_primary_labels(DEFAULT_DATA_DIR)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        all_rows = load_soundscape_rows(DEFAULT_DATA_DIR)
        full_rows = filter_fully_labeled_rows(all_rows)

        specs = {spec.name: spec for spec in default_experiments()}
        notebook_result = evaluate_experiment(
            spec=specs["notebook_full59_filegkf5_sitehour"],
            rows=rows_for_dataset(all_rows, full_rows, "full59"),
            label_to_idx=label_to_idx,
            n_classes=len(labels),
        )
        site_balanced_result = evaluate_experiment(
            spec=specs["all66_sitebalanced_file3_sitehour"],
            rows=rows_for_dataset(all_rows, full_rows, "all66"),
            label_to_idx=label_to_idx,
            n_classes=len(labels),
        )

        self.assertGreater(site_balanced_result.oof_macro_auc, notebook_result.oof_macro_auc)
        self.assertGreater(
            site_balanced_result.min_fold_active_classes,
            notebook_result.min_fold_active_classes,
        )


if __name__ == "__main__":
    unittest.main()
