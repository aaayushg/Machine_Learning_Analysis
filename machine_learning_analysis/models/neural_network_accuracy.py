#!/usr/bin/env python3

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from machine_learning_analysis.common import (
    build_parser,
    configure_plotting,
    dataset_stem,
    ensure_output_dir,
    load_csv,
    print_dataframe_summary,
    require_columns,
    save_figure,
    write_json_report,
)


DEFAULT_FEATURES = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]


def parse_args():
    parser = build_parser("Benchmark neural network hidden-layer sizes across a configurable range.")
    parser.add_argument(
        "--analysis-type",
        choices=["regression", "classification"],
        default="regression",
        help="Model family to evaluate.",
    )
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns to include.")
    parser.add_argument("--target", default="area", help="Target column.")
    parser.add_argument("--min-neurons", type=int, default=1, help="Smallest hidden layer size to test.")
    parser.add_argument("--max-neurons", type=int, default=50, help="Largest hidden layer size to test.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "sgd", "adam"],
        default="lbfgs",
        help="Optimization algorithm for the MLP model.",
    )
    return parser.parse_args()


def build_estimator(args, hidden_units: int):
    common_kwargs = {
        "hidden_layer_sizes": (hidden_units,),
        "solver": args.solver,
        "max_iter": 1000,
        "random_state": args.random_state,
    }
    if args.analysis_type == "classification":
        return MLPClassifier(activation="relu", **common_kwargs), "accuracy"
    return MLPRegressor(activation="logistic", **common_kwargs), "r2"


def main() -> None:
    args = parse_args()
    if args.min_neurons > args.max_neurons:
        raise ValueError("--min-neurons must be less than or equal to --max-neurons.")

    configure_plotting()
    data = load_csv(args.data)
    print_dataframe_summary(data)
    require_columns(data, [*args.features, args.target])

    x_data = data[args.features]
    y_data = data[args.target]
    stratify = y_data if args.analysis_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    neuron_counts = list(range(args.min_neurons, args.max_neurons + 1))
    sweep_results = []
    best_result = None
    for hidden_units in neuron_counts:
        estimator, scoring = build_estimator(args, hidden_units)
        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", estimator),
        ])
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=args.cv, scoring=scoring)
        pipeline.fit(x_train, y_train)
        test_score = float(pipeline.score(x_test, y_test))
        result = {
            "hidden_units": hidden_units,
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std_x2": round(float(cv_scores.std() * 2), 4),
            "test_score": round(test_score, 4),
        }
        sweep_results.append(result)
        if best_result is None or test_score > best_result["test_score"]:
            best_result = result

    output_dir = ensure_output_dir(args.output_dir)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(neuron_counts, [item["test_score"] for item in sweep_results], marker="o")
    axis.set_xlabel("Hidden units")
    axis.set_ylabel("Test score")
    axis.set_title(f"MLP {args.analysis_type.title()} Hidden-Unit Sweep")
    plot_path = output_dir / f"{dataset_stem(args.data)}_nn_accuracy.png"
    save_figure(figure, plot_path)

    report = {
        "analysis_type": args.analysis_type,
        "features": args.features,
        "target": args.target,
        "min_neurons": args.min_neurons,
        "max_neurons": args.max_neurons,
        "best_result": best_result,
        "results": sweep_results,
        "plot": str(plot_path),
    }
    report_path = output_dir / f"{dataset_stem(args.data)}_nn_accuracy.json"
    write_json_report(report, report_path)

    print(f"Best result: {best_result}")
    print(f"Plot written to: {plot_path}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()