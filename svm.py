#!/usr/bin/env python3

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

from ml_analysis_utils import (
    build_parser,
    configure_plotting,
    dataset_stem,
    ensure_output_dir,
    load_csv,
    print_dataframe_summary,
    require_columns,
    save_figure,
    serialize_scores,
    write_json_report,
)


def parse_args():
    parser = build_parser("Train an SVM classifier and optionally render a decision boundary.")
    parser.add_argument("--features", nargs="+", default=["X1", "X2"], help="Feature columns to include.")
    parser.add_argument("--target", default="Y", help="Classification target column.")
    parser.add_argument("--kernel", choices=["linear", "poly", "rbf", "sigmoid"], default="linear")
    parser.add_argument("--c", type=float, default=1.0, help="Regularization strength.")
    parser.add_argument("--gamma", default="scale", help="Kernel coefficient for rbf, poly, and sigmoid kernels.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree when --kernel=poly.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument(
        "--plot-decision-boundary",
        action="store_true",
        help="Render a decision boundary plot when exactly two features are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plotting()
    data = load_csv(args.data)
    print_dataframe_summary(data)
    require_columns(data, [*args.features, args.target])

    x_data = data[args.features]
    y_data = data[args.target]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_data,
    )

    model = SVC(
        kernel=args.kernel,
        C=args.c,
        gamma=args.gamma,
        degree=args.degree,
        random_state=args.random_state,
    )
    cv_scores = cross_val_score(model, x_train, y_train, cv=args.cv, scoring="accuracy")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions)

    report = {
        "features": args.features,
        "target": args.target,
        "kernel": args.kernel,
        "c": args.c,
        "gamma": args.gamma,
        "degree": args.degree,
        "cv_scores": serialize_scores(cv_scores),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std_x2": round(float(cv_scores.std() * 2), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "support_vector_count": int(model.n_support_.sum()),
        "intercept": [round(float(value), 6) for value in model.intercept_],
    }
    if args.kernel == "linear":
        report["coefficients"] = [[round(float(value), 6) for value in row] for row in model.coef_]

    output_dir = ensure_output_dir(args.output_dir)
    if args.plot_decision_boundary:
        if len(args.features) != 2:
            raise ValueError("Decision boundary plotting requires exactly two features.")

        x_matrix = x_data.to_numpy()
        x_min, x_max = x_matrix[:, 0].min() - 0.5, x_matrix[:, 0].max() + 0.5
        y_min, y_max = x_matrix[:, 1].min() - 0.5, x_matrix[:, 1].max() + 0.5
        step = max((x_max - x_min) / 300, 0.01)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.pcolormesh(xx, yy, zz, shading="auto", cmap="Pastel1")
        scatter = axis.scatter(x_matrix[:, 0], x_matrix[:, 1], c=y_data.astype("category").cat.codes, cmap="tab10")
        axis.set_xlabel(args.features[0])
        axis.set_ylabel(args.features[1])
        axis.set_title(f"SVM Decision Boundary ({args.kernel})")
        figure.colorbar(scatter, ax=axis, label=args.target)
        plot_path = output_dir / f"{dataset_stem(args.data)}_svm_boundary.png"
        save_figure(figure, plot_path)
        report["decision_boundary_plot"] = str(plot_path)
        print(f"Decision boundary plot written to: {plot_path}")

    report_path = output_dir / f"{dataset_stem(args.data)}_svm.json"
    write_json_report(report, report_path)

    print(f"Cross-validation mean: {report['cv_mean']} (+/- {report['cv_std_x2']})")
    print(f"Test accuracy: {report['test_accuracy']}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()