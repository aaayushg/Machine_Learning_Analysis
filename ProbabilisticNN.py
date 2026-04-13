#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

from ml_analysis_utils import (
    build_parser,
    dataset_stem,
    ensure_output_dir,
    load_csv,
    print_dataframe_summary,
    require_columns,
    serialize_scores,
    write_json_report,
)

try:
    from neupy.algorithms import PNN
except ImportError:  # pragma: no cover - dependency availability varies by environment
    PNN = None


DEFAULT_EXCLUDED_COLUMNS = {"Name", "Grade", "Letter", "Percentile"}


def parse_args():
    parser = build_parser("Train a probabilistic neural network or Gaussian Naive Bayes baseline using class prototypes.")
    parser.add_argument("--features", nargs="+", help="Feature columns to include. Defaults to numeric columns excluding metadata.")
    parser.add_argument("--target", default="Letter", help="Classification target column.")
    parser.add_argument("--method", choices=["gaussian_nb", "pnn"], default="gaussian_nb")
    parser.add_argument("--prototypes-per-class", type=int, default=4, help="Maximum prototypes to derive per class.")
    parser.add_argument("--std", type=float, default=0.1, help="PNN smoothing parameter.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--cv", type=int, default=3, help="Number of cross-validation folds.")
    return parser.parse_args()


def infer_features(data: pd.DataFrame, target: str, requested):
    if requested:
        require_columns(data, [*requested, target])
        return list(requested)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in DEFAULT_EXCLUDED_COLUMNS and column != target]


def build_class_prototypes(x_train: pd.DataFrame, y_train: pd.Series, prototypes_per_class: int, random_state: int):
    prototype_frames = []
    prototype_labels = []
    for label in sorted(y_train.unique()):
        class_rows = x_train.loc[y_train == label]
        cluster_count = min(prototypes_per_class, len(class_rows))
        if cluster_count < 1:
            continue
        model = KMeans(n_clusters=cluster_count, n_init=10, random_state=random_state)
        model.fit(class_rows)
        centers = pd.DataFrame(model.cluster_centers_, columns=x_train.columns)
        prototype_frames.append(centers)
        prototype_labels.extend([label] * cluster_count)
    if not prototype_frames:
        raise ValueError("Unable to build class prototypes from the training data.")
    return pd.concat(prototype_frames, ignore_index=True), np.asarray(prototype_labels)


def main() -> None:
    args = parse_args()
    if args.method == "pnn" and PNN is None:
        raise ImportError("neupy is required for --method pnn. Install it with `pip install neupy==0.8.2` or a compatible version.")

    data = load_csv(args.data)
    print_dataframe_summary(data)
    require_columns(data, [args.target])
    features = infer_features(data, args.target, args.features)
    if not features:
        raise ValueError("No usable feature columns were found.")

    x_data = data[features]
    y_data = data[args.target]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_data,
    )

    scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=features, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=features, index=x_test.index)
    x_prototypes, y_prototypes = build_class_prototypes(
        x_train_scaled,
        y_train,
        args.prototypes_per_class,
        args.random_state,
    )

    if args.method == "pnn":
        model = PNN(std=args.std)
        model.train(x_prototypes, y_prototypes)
    else:
        model = GaussianNB()
        model.fit(x_prototypes, y_prototypes)

    cv_scores = cross_val_score(model, x_prototypes, y_prototypes, scoring="accuracy", cv=args.cv)
    predictions = model.predict(x_test_scaled)
    test_accuracy = accuracy_score(y_test, predictions)

    report = {
        "features": features,
        "target": args.target,
        "method": args.method,
        "prototypes_per_class": args.prototypes_per_class,
        "prototype_count": int(len(x_prototypes)),
        "cv_scores": serialize_scores(cv_scores),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std_x2": round(float(cv_scores.std() * 2), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "classes": sorted(str(value) for value in y_data.unique()),
    }
    if args.method == "gaussian_nb":
        report["class_count"] = [int(value) for value in model.class_count_]

    output_dir = ensure_output_dir(args.output_dir)
    report_path = output_dir / f"{dataset_stem(args.data)}_probabilistic_nn.json"
    write_json_report(report, report_path)

    print(f"Cross-validation mean: {report['cv_mean']} (+/- {report['cv_std_x2']})")
    print(f"Test accuracy: {report['test_accuracy']}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
