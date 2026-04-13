# Machine Learning Analysis

This repository contains command-line tools for classical machine learning analysis tasks on CSV datasets. It is now organized as a small Python package instead of a flat set of ad hoc scripts.

## Repository Layout

```text
machine_learning_analysis/
  common.py
  data_sources/
    fetch_zinc.py
  models/
    backpropagation_nn.py
    generalized_regression_nn.py
    knn_classifier.py
    neural_network_accuracy.py
    probabilistic_nn.py
    svm_classifier.py
    tree_models.py
  unsupervised/
    hierarchical_clustering.py
    pca_analysis.py
  visualization/
    exploratory_plots.py
README.md
requirements.txt
```

## What Changed

- Renamed all scripts to consistent snake_case module names.
- Grouped files by responsibility: `models`, `unsupervised`, `visualization`, and `data_sources`.
- Moved shared helpers into [machine_learning_analysis/common.py](machine_learning_analysis/common.py).
- Removed interactive prompts and hard-coded CSV filenames.
- Standardized all tools around `--data`, `--output-dir`, and `--random-state`.
- Added validation for required columns and deterministic model behavior.
- Saved metrics to JSON and plots to `outputs/` instead of relying on transient console output.

## Requirements

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Optional dependency:

- `neupy` is only required for `generalized_regression_nn.py` and for `probabilistic_nn.py --method pnn`.

## Running Modules

Run the tools as Python modules from the repository root:

```bash
python3 -m machine_learning_analysis.<group>.<module> --data path/to/dataset.csv
```

Shared arguments:

- `--data`: input CSV file path.
- `--output-dir`: directory for plots and reports. Defaults to `outputs`.
- `--random-state`: seed for deterministic model behavior. Defaults to `42`.

Use `--help` on any module to inspect its full interface.

## Modules

### machine_learning_analysis.models.backpropagation_nn

Train a multilayer perceptron for regression or classification.

```bash
python3 -m machine_learning_analysis.models.backpropagation_nn \
  --data data/course_data.csv \
  --analysis-type classification \
  --features AvgHW AvgQuiz AvgLab MT1 MT2 Final Participation \
  --target Letter
```

### machine_learning_analysis.models.tree_models

Compare decision tree, random forest, and gradient boosting classifiers.

```bash
python3 -m machine_learning_analysis.models.tree_models \
  --data data/course_data.csv \
  --target Letter \
  --features AvgHW AvgQuiz AvgLab MT1 MT2 Final Participation
```

### machine_learning_analysis.models.generalized_regression_nn

Train a generalized regression neural network using KMeans-derived prototypes.

```bash
python3 -m machine_learning_analysis.models.generalized_regression_nn \
  --data data/course_data.csv \
  --target Grade \
  --prototypes 12 \
  --std 0.1
```

### machine_learning_analysis.unsupervised.hierarchical_clustering

Run KMeans and agglomerative clustering on two selected features and export a dendrogram.

```bash
python3 -m machine_learning_analysis.unsupervised.hierarchical_clustering \
  --data data/sample_data.csv \
  --features X1 X2 \
  --clusters 8
```

### machine_learning_analysis.models.knn_classifier

Train a k-nearest-neighbors classifier and optionally render a decision boundary.

```bash
python3 -m machine_learning_analysis.models.knn_classifier \
  --data data/HW3_data.csv \
  --features X1 X2 \
  --target Y \
  --neighbors 5 \
  --plot-decision-boundary
```

### machine_learning_analysis.models.neural_network_accuracy

Sweep hidden-layer sizes for an MLP and plot the resulting scores.

```bash
python3 -m machine_learning_analysis.models.neural_network_accuracy \
  --data data/forestfires.csv \
  --analysis-type regression \
  --target area \
  --min-neurons 1 \
  --max-neurons 50
```

### machine_learning_analysis.unsupervised.pca_analysis

Compute principal components and export transformed component values.

```bash
python3 -m machine_learning_analysis.unsupervised.pca_analysis \
  --data data/course_data.csv \
  --components 3 \
  --features AvgHW AvgQuiz AvgLab MT1 MT2 Final Participation \
  --scale
```

### machine_learning_analysis.visualization.exploratory_plots

Generate pairplots, jointplots, and simple linear regression diagnostics.

```bash
python3 -m machine_learning_analysis.visualization.exploratory_plots \
  --data data/course_data.csv \
  --header \
  --pairplot \
  --jointplot AvgHW Final \
  --regression AvgHW Final
```

### machine_learning_analysis.models.probabilistic_nn

Train either Gaussian Naive Bayes or a probabilistic neural network using per-class KMeans prototypes.

```bash
python3 -m machine_learning_analysis.models.probabilistic_nn \
  --data data/course_data.csv \
  --target Letter \
  --method gaussian_nb \
  --prototypes-per-class 4
```

### machine_learning_analysis.models.svm_classifier

Train a support vector classifier and optionally render a decision boundary.

```bash
python3 -m machine_learning_analysis.models.svm_classifier \
  --data data/sample_data.csv \
  --features X1 X2 \
  --target Y \
  --kernel linear \
  --plot-decision-boundary
```

### machine_learning_analysis.data_sources.fetch_zinc

Fetch compound descriptors from ZINC15 for a numeric ID range and export them to CSV.

```bash
python3 -m machine_learning_analysis.data_sources.fetch_zinc 1000 1010 --output outputs/zinc_subset.csv
```

## Validation

Syntax validation can be run with:

```bash
python3 -m compileall machine_learning_analysis
```

Because the repository does not include the original CSV datasets, end-to-end model execution still depends on providing real input files and installing the required packages.

## Notes

- Reports and plots are written under `outputs/` by default.
- Generated artifacts are ignored by git.