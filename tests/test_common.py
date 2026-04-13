#!/usr/bin/env python3

from __future__ import annotations

import unittest

from machine_learning_analysis.common import load_csv, resolve_data_path


class CommonTests(unittest.TestCase):
    def test_load_csv_uses_bundled_fixture_for_missing_file(self):
        dataframe = load_csv("course_data.csv")
        self.assertGreaterEqual(len(dataframe), 40)
        self.assertIn("Grade", dataframe.columns)
        self.assertIn("Letter", dataframe.columns)

    def test_resolve_data_path_prefers_bundled_fixture(self):
        resolved_path = resolve_data_path("forestfires.csv")
        self.assertEqual(resolved_path.name, "forestfires.csv")
        self.assertTrue(resolved_path.exists())

    def test_load_csv_raises_for_unknown_dataset(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("definitely_missing_dataset.csv")


if __name__ == "__main__":
    unittest.main()
