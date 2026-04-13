# Machine Learning Analysis

This repository contains command-line tools for classical machine learning analysis tasks on CSV datasets. The original codebase was a collection of interactive, dataset-specific scripts. It has been refactored into importable Python 3 modules with consistent CLI arguments, reproducible model training, input validation, and structured outputs.

## What Changed

- Removed interactive `input()` prompts and hard-coded CSV filenames.
- Standardized all scripts around `--data`, `--output-dir`, and `--random-state`.
- Added validation for required columns and safer filesystem handling.
- Made model training reproducible through explicit random seeds.
- Saved metrics to JSON and plots to `outputs/` instead of relying on transient console output.
- Modernized `fetchZinc.py` to Python 3 and a CSV-based export flow.
- Added dependency documentation and git ignore rules for generated files.

## Requirements

Install the dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Optional dependency:

- `neupy` is only required for `GeneralizedRegressionNN.py` and for `ProbabilisticNN.py --method pnn`.

## Common Usage Pattern

Most scripts follow this structure:

```bash
python3 <script>.py --data path/to/dataset.csv --output-dir outputs
```

Shared arguments:

- `--data`: input CSV file path.
- `--output-dir`: directory for plots and reports. Defaults to `outputs`.
- `--random-state`: seed for deterministic model behavior. Defaults to `42`.

Use `--help` on any script to see its full interface.

## Scripts

### BackpropagationNN.py

Train a multilayer perceptron for regression or classification.

Example:

```bash
python3 BackpropagationNN.py \
  --data data/course_data.csv \
  --analysis-type classification \
  --features AvgHW AvgQuiz AvgLab MT1 MT2 Final Participation \
  --target Letter
```

Output: JSON report with cross-validation and test metrics.

### DecisionTree.py

Compare three tree-based classifiers: decision tree, random forest, and gradient boosting.

Example:

```bash
python3 DecisionTree.py \
  --data data/course_data.csv \
  --target Letter \
  --features AvgHW AvgQuiz AvgLab MT1 MT2 Final Participation
```

Outputs:

- Feature-importance plot
- Graphviz `.dot` export for the fitted decision tree
- JSON metrics report

### GeneralizedRegressionNN.py

Train a generalized regression neural network using KMeans-derived prototypes.

Example:

```bash
python3 GeneralizedRegressionNN.py \
  --data data/course_data.csv \
  --target Grade \
  --prototypes 12 \
  --std 0.1
```

Output: JSON regression report.

### Heirarchical_Clustering.py

Run KMeans and agglomerative clustering on two selected features and export a dendrogram.

Example:

```bash
python3 Heirarchical_Clustering.py \
  --data data/sample_data.csv \
  --features X1 X2 \
  --clusters 8
```

Outputs:

- Combined cluster comparison plot
- Dendrogram plot
- JSON silhouette summary

### kNN.py

Train a k-nearest-neighbors classifier and optionally render a decision boundary when using two features.

Example:

```bash
python3 kNN.py \
  --data data/HW3_data.csv \
  --features X1 X2 \
  --target Y \
  --neighbors 5 \
  --plot-decision-boundary
```

### NN_accuracy.py

Sweep hidden-layer sizes for an MLP and plot the resulting test scores.

Example:

```bash
python3 NN_accuracy.py \
  --data data/forestfires.csv \
  --analysis-type regression \
  --target area \
  --min-neurons 1 \
  --max-neurons 50
```

Outputs:

- Accuracy sweep plot
- JSON report containing all tested neuron counts

### PCA.py

Compute principal components on selected numeric columns and export transformed component values.

Example:

```bash
python3 PCA.py \
  --data data/course_data.csv \
  --components 3 \
  --features AvgHW AvgQuiz AvgLab MT1 MT2 Final Participation \
  --scale
```

Outputs:

- CSV of principal component scores
- Explained variance plot
- JSON report with covariance, correlation, and component loadings

### Pairplot_Jointplot.py

Generate pairplots, jointplots, and simple linear regression diagnostics.

Example:

```bash
python3 Pairplot_Jointplot.py \
  --data data/course_data.csv \
  --header \
  --pairplot \
  --jointplot AvgHW Final \
  --regression AvgHW Final
```

### ProbabilisticNN.py

Train either Gaussian Naive Bayes or a probabilistic neural network using per-class KMeans prototypes.

Example:

```bash
python3 ProbabilisticNN.py \
  --data data/course_data.csv \
  --target Letter \
  --method gaussian_nb \
  --prototypes-per-class 4
```

Output: JSON classification report.

### svm.py

Train a support vector classifier and optionally render a decision boundary.

Example:

```bash
python3 svm.py \
  --data data/sample_data.csv \
  --features X1 X2 \
  --target Y \
  --kernel linear \
  --plot-decision-boundary
```

### fetchZinc.py

Fetch compound descriptors from ZINC15 for a numeric ID range and export them to CSV.

Example:

```bash
python3 fetchZinc.py 1000 1010 --output outputs/zinc_subset.csv
```

## Validation

Syntax validation was run with:

```bash
python3 -m compileall .
```

Because the repository does not include the original CSV datasets, end-to-end model execution was not run during this refactor.

## Notes

- `Heirarchical_Clustering.py` retains its original filename for backward compatibility, even though the spelling is non-standard.
- Reports and plots are written under `outputs/` by default.
- Generated artifacts are ignored by git.