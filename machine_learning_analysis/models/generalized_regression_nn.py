#!/usr/bin/env python3

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

from machine_learning_analysis.common import (
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
    from neupy.algorithms import GRNN
except ImportError:  # pragma: no cover - dependency availability varies by environment
    GRNN = None


DEFAULT_EXCLUDED_COLUMNS = {"Name", "Letter", "Percentile", "Grade"}


def parse_args():
    parser = build_parser("Train a generalized regression neural network using prototype selection.")
    parser.add_argument("--features", nargs="+", help="Feature columns to include. Defaults to numeric columns excluding metadata.")
    parser.add_argument("--target", default="Grade", help="Regression target column.")
    parser.add_argument("--prototypes", type=int, default=12, help="Number of KMeans prototypes.")
    parser.add_argument("--std", type=float, default=0.1, help="GRNN smoothing parameter.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    return parser.parse_args()


def infer_features(data: pd.DataFrame, target: str, requested: Sequence[str] | None) -> list[str]:
    if requested:
        require_columns(data, [*requested, target])
        return list(requested)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in DEFAULT_EXCLUDED_COLUMNS and column != target]


def build_prototypes(x_train: pd.DataFrame, y_train: pd.Series, prototypes: int, random_state: int):
    cluster_count = min(prototypes, len(x_train))
    model = KMeans(n_clusters=cluster_count, n_init=10, random_state=random_state)
    labels = model.fit_predict(x_train)
    prototype_x = pd.DataFrame(model.cluster_centers_, columns=x_train.columns)
    prototype_y = pd.Series(y_train.to_numpy()).groupby(labels).mean().reindex(range(cluster_count)).to_numpy()
    return prototype_x, prototype_y


def main() -> None:
    args = parse_args()
    if GRNN is None:
        raise ImportError("neupy is required for this script. Install it with `pip install neupy==0.8.2` or a compatible version.")

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
    )

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(x_scaler.fit_transform(x_train), columns=features, index=x_train.index)
    x_test_scaled = pd.DataFrame(x_scaler.transform(x_test), columns=features, index=x_test.index)
    y_train_scaled = y_scaler.fit_transform(y_train.to_frame()).ravel()
    y_test_scaled = y_scaler.transform(y_test.to_frame()).ravel()

    x_prototypes, y_prototypes = build_prototypes(x_train_scaled, pd.Series(y_train_scaled), args.prototypes, args.random_state)
    model = GRNN(std=args.std)
    model.train(x_prototypes, y_prototypes)
    cv_scores = cross_val_score(model, x_train_scaled, y_train_scaled, scoring="r2", cv=args.cv)

    predictions_scaled = np.asarray(model.predict(x_test_scaled)).reshape(-1, 1)
    predictions = y_scaler.inverse_transform(predictions_scaled).ravel()
    test_r2 = r2_score(y_test, predictions)
    scaled_test_r2 = r2_score(y_test_scaled, predictions_scaled.ravel())

    output_dir = ensure_output_dir(args.output_dir)
    report = {
        "features": features,
        "target": args.target,
        "prototypes": int(len(x_prototypes)),
        "std": args.std,
        "cv_scores": serialize_scores(cv_scores),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std_x2": round(float(cv_scores.std() * 2), 4),
        "test_r2": round(float(test_r2), 4),
        "scaled_test_r2": round(float(scaled_test_r2), 4),
    }
    report_path = output_dir / f"{dataset_stem(args.data)}_grnn.json"
    write_json_report(report, report_path)

    print(f"Cross-validation mean: {report['cv_mean']} (+/- {report['cv_std_x2']})")
    print(f"Test R2: {report['test_r2']}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()