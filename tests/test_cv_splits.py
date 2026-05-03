# ruff: noqa: E402
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cv_splits import ClipFoldRecord, file_group_kfold, site_holdout_folds


class CvSplitsTest(unittest.TestCase):
    def test_file_group_kfold_keeps_each_file_in_single_validation_fold(self) -> None:
        records = [
            ClipFoldRecord(filename="a.wav", site="S01"),
            ClipFoldRecord(filename="a.wav", site="S01"),
            ClipFoldRecord(filename="b.wav", site="S01"),
            ClipFoldRecord(filename="c.wav", site="S02"),
            ClipFoldRecord(filename="c.wav", site="S02"),
            ClipFoldRecord(filename="d.wav", site="S03"),
        ]

        folds = file_group_kfold(records, n_splits=3)
        seen_validation_files: set[str] = set()
        all_val_indices: set[int] = set()

        for train_indices, val_indices in folds:
            self.assertTrue(set(train_indices).isdisjoint(val_indices))
            val_files = {records[index].filename for index in val_indices}
            self.assertTrue(seen_validation_files.isdisjoint(val_files))
            seen_validation_files.update(val_files)
            all_val_indices.update(val_indices)

        self.assertEqual(all_val_indices, set(range(len(records))))

    def test_site_holdout_folds_keep_each_site_out_of_train(self) -> None:
        records = [
            ClipFoldRecord(filename="a.wav", site="S01"),
            ClipFoldRecord(filename="b.wav", site="S01"),
            ClipFoldRecord(filename="c.wav", site="S02"),
            ClipFoldRecord(filename="d.wav", site="S02"),
            ClipFoldRecord(filename="e.wav", site="S03"),
            ClipFoldRecord(filename="f.wav", site="S03"),
        ]

        folds = site_holdout_folds(records, n_splits=3)
        seen_validation_sites: set[str] = set()

        for train_indices, val_indices in folds:
            self.assertTrue(val_indices)
            val_sites = {records[index].site for index in val_indices}
            train_sites = {records[index].site for index in train_indices}
            self.assertEqual(len(val_sites), 1)
            self.assertTrue(val_sites.isdisjoint(train_sites))
            seen_validation_sites.update(val_sites)

        self.assertEqual(seen_validation_sites, {"S01", "S02", "S03"})


if __name__ == "__main__":
    unittest.main()
