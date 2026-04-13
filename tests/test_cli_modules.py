#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ModuleCliTests(unittest.TestCase):
    maxDiff = None

    def run_module(self, module: str, *args: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        output_dir = Path(tempfile.mkdtemp(prefix="mla-test-"))
        command = [
            sys.executable,
            "-m",
            module,
            *args,
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        return result, output_dir

    def test_backpropagation_classifier_runs_with_dummy_course_data(self):
        _, output_dir = self.run_module(
            "machine_learning_analysis.models.backpropagation_nn",
            "--data",
            "course_data.csv",
            "--analysis-type",
            "classification",
            "--cv",
            "3",
            "--max-iter",
            "200",
        )
        self.assertTrue((output_dir / "course_data_backpropagation_classification.json").exists())

    def test_tree_models_runs_with_dummy_course_data(self):
        _, output_dir = self.run_module(
            "machine_learning_analysis.models.tree_models",
            "--data",
            "course_data.csv",
            "--cv",
            "3",
        )
        self.assertTrue((output_dir / "course_data_tree_models.json").exists())
        self.assertTrue((output_dir / "course_data_decision_tree.dot").exists())

    def test_unsupervised_modules_run_with_dummy_data(self):
        _, clustering_output = self.run_module(
            "machine_learning_analysis.unsupervised.hierarchical_clustering",
            "--data",
            "sample_data.csv",
            "--clusters",
            "4",
        )
        self.assertTrue((clustering_output / "sample_data_clustering.json").exists())

        _, pca_output = self.run_module(
            "machine_learning_analysis.unsupervised.pca_analysis",
            "--data",
            "course_data.csv",
            "--components",
            "3",
            "--features",
            "AvgHW",
            "AvgQuiz",
            "AvgLab",
            "MT1",
            "MT2",
            "Final",
            "Participation",
        )
        self.assertTrue((pca_output / "course_data_pca.json").exists())

    def test_visualization_and_accuracy_modules_run_with_dummy_data(self):
        _, eda_output = self.run_module(
            "machine_learning_analysis.visualization.exploratory_plots",
            "--data",
            "course_data.csv",
            "--header",
            "--pairplot",
            "--pairplot-columns",
            "AvgHW",
            "AvgQuiz",
            "Grade",
            "--jointplot",
            "AvgHW",
            "Final",
            "--jointplot-kind",
            "scatter",
            "--regression",
            "AvgHW",
            "Final",
        )
        self.assertTrue((eda_output / "course_data_eda.json").exists())

        _, accuracy_output = self.run_module(
            "machine_learning_analysis.models.neural_network_accuracy",
            "--data",
            "forestfires.csv",
            "--min-neurons",
            "1",
            "--max-neurons",
            "3",
            "--cv",
            "3",
        )
        self.assertTrue((accuracy_output / "forestfires_nn_accuracy.json").exists())

    def test_classification_modules_run_with_dummy_data(self):
        _, knn_output = self.run_module(
            "machine_learning_analysis.models.knn_classifier",
            "--data",
            "HW3_data.csv",
            "--cv",
            "3",
            "--plot-decision-boundary",
        )
        self.assertTrue((knn_output / "HW3_data_knn.json").exists())

        _, svm_output = self.run_module(
            "machine_learning_analysis.models.svm_classifier",
            "--data",
            "sample_data.csv",
            "--cv",
            "3",
            "--plot-decision-boundary",
        )
        self.assertTrue((svm_output / "sample_data_svm.json").exists())

        _, pnn_output = self.run_module(
            "machine_learning_analysis.models.probabilistic_nn",
            "--data",
            "course_data.csv",
            "--method",
            "gaussian_nb",
            "--cv",
            "2",
        )
        self.assertTrue((pnn_output / "course_data_probabilistic_nn.json").exists())

    def test_fetch_zinc_help_works_without_network(self):
        result = subprocess.run(
            [sys.executable, "-m", "machine_learning_analysis.data_sources.fetch_zinc", "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Fetch ZINC15 compound descriptors", result.stdout)


if __name__ == "__main__":
    unittest.main()
