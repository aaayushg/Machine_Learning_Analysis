#!/usr/bin/env python3

from __future__ import annotations

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
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


DEFAULT_FEATURES = ["AvgHW", "AvgQuiz", "AvgLab", "MT1", "MT2", "Final", "Participation"]


def parse_args():
    parser = build_parser("Train a backpropagation neural network for regression or classification.")
    parser.add_argument(
        "--analysis-type",
        choices=["regression", "classification"],
        required=True,
        help="Choose whether to fit a regressor or classifier.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to include.",
    )
    parser.add_argument("--target", help="Override the default target column.")
    parser.add_argument("--hidden-units", type=int, default=7, help="Number of hidden units.")
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "sgd", "adam"],
        default="lbfgs",
        help="Optimization algorithm for the MLP model.",
    )
    parser.add_argument(
        "--activation",
        choices=["identity", "logistic", "tanh", "relu"],
        default=None,
        help="Activation function. Defaults depend on analysis type.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations.")
    return parser.parse_args()


def build_model(args):
    if args.analysis_type == "regression":
        activation = args.activation or "logistic"
        estimator = MLPRegressor(
            activation=activation,
            solver=args.solver,
            hidden_layer_sizes=(args.hidden_units,),
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        scoring = "r2"
        default_target = "Grade"
        stratify = None
    else:
        activation = args.activation or "relu"
        estimator = MLPClassifier(
            activation=activation,
            solver=args.solver,
            hidden_layer_sizes=(args.hidden_units,),
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        scoring = "accuracy"
        default_target = "Letter"
        stratify = True

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", estimator),
    ])
    return pipeline, scoring, default_target, stratify


def main() -> None:
    args = parse_args()
    data = load_csv(args.data)
    print_dataframe_summary(data)

    pipeline, scoring, default_target, should_stratify = build_model(args)
    target = args.target or default_target

    require_columns(data, [*args.features, target])
    x_data = data[args.features]
    y_data = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_data if should_stratify else None,
    )

    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=args.cv, scoring=scoring)
    pipeline.fit(x_train, y_train)
    test_score = pipeline.score(x_test, y_test)
    fitted_model = pipeline.named_steps["model"]

    report = {
        "analysis_type": args.analysis_type,
        "features": args.features,
        "target": target,
        "cv_scores": serialize_scores(cv_scores),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std_x2": round(float(cv_scores.std() * 2), 4),
        "test_score": round(float(test_score), 4),
        "hidden_units": args.hidden_units,
        "solver": args.solver,
        "activation": fitted_model.activation,
        "loss": round(float(fitted_model.loss_), 6),
        "iterations": int(fitted_model.n_iter_),
        "n_layers": int(fitted_model.n_layers_),
        "n_outputs": int(fitted_model.n_outputs_),
        "output_activation": fitted_model.out_activation_,
    }
    if args.analysis_type == "classification":
        report["classes"] = fitted_model.classes_.tolist()

    output_dir = ensure_output_dir(args.output_dir)
    report_path = output_dir / f"{dataset_stem(args.data)}_backpropagation_{args.analysis_type}.json"
    write_json_report(report, report_path)

    print(f"Cross-validation mean: {report['cv_mean']} (+/- {report['cv_std_x2']})")
    print(f"Test score: {report['test_score']}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()