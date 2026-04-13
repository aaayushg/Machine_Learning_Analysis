#!/usr/bin/env python3

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from machine_learning_analysis.common import (
    build_parser,
    configure_plotting,
    dataset_stem,
    ensure_output_dir,
    load_csv,
    print_dataframe_summary,
    save_figure,
    write_json_report,
)


def parse_args():
    parser = build_parser("Compute principal components for a numeric dataset.")
    parser.add_argument("--features", nargs="+", help="Feature columns to include. Defaults to all numeric columns.")
    parser.add_argument("--components", type=int, required=True, help="Number of principal components.")
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Apply standard scaling before PCA.",
    )
    parser.add_argument(
        "--index-col",
        type=int,
        default=0,
        help="Column index to use as the dataframe index when loading the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plotting()
    data = load_csv(args.data, index_col=args.index_col)
    print_dataframe_summary(data)

    if args.features:
        missing = [column for column in args.features if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        feature_data = data[args.features]
    else:
        feature_data = data.select_dtypes(include=[np.number])
        if feature_data.empty:
            raise ValueError("No numeric columns found for PCA.")

    if args.components < 1 or args.components > feature_data.shape[1]:
        raise ValueError("--components must be between 1 and the number of selected features.")

    matrix = feature_data.to_numpy()
    covariance = np.cov(matrix, rowvar=False)
    correlation = np.corrcoef(matrix, rowvar=False)
    if args.scale:
        matrix = StandardScaler().fit_transform(matrix)

    pca = PCA(n_components=args.components)
    transformed = pca.fit_transform(matrix)
    component_names = [f"PC{i}" for i in range(1, args.components + 1)]
    transformed_df = pd.DataFrame(transformed, columns=component_names, index=feature_data.index)

    output_dir = ensure_output_dir(args.output_dir)
    transformed_path = output_dir / f"{dataset_stem(args.data)}_pca_components.csv"
    transformed_df.to_csv(transformed_path)

    figure, axis = plt.subplots(figsize=(8, 5))
    explained = pca.explained_variance_ratio_
    axis.bar(component_names, explained)
    axis.set_ylabel("Explained variance ratio")
    axis.set_title("PCA Explained Variance")
    plot_path = output_dir / f"{dataset_stem(args.data)}_pca_explained_variance.png"
    save_figure(figure, plot_path)

    report = {
        "features": feature_data.columns.tolist(),
        "components": args.components,
        "scaled": args.scale,
        "mean": [round(float(value), 6) for value in pca.mean_],
        "explained_variance_ratio": [round(float(value), 6) for value in explained],
        "components_matrix": [[round(float(value), 6) for value in row] for row in pca.components_],
        "covariance_matrix": np.round(covariance, 6).tolist(),
        "correlation_matrix": np.round(correlation, 6).tolist(),
        "transformed_csv": str(transformed_path),
        "plot": str(plot_path),
    }
    report_path = output_dir / f"{dataset_stem(args.data)}_pca.json"
    write_json_report(report, report_path)

    print("Explained variance ratio:")
    print(report["explained_variance_ratio"])
    print(f"Transformed components written to: {transformed_path}")
    print(f"Plot written to: {plot_path}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()