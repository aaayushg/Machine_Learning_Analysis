#!/usr/bin/env python3

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

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


DEFAULT_FEATURES = ["AvgHW", "AvgQuiz", "AvgLab", "MT1", "MT2", "Final", "Participation"]


def parse_args():
    parser = build_parser("Compare decision-tree-based classifiers and export feature importances.")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns to include.")
    parser.add_argument("--target", default="Letter", help="Classification target column.")
    parser.add_argument("--criterion", choices=["gini", "entropy", "log_loss"], default="gini")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum decision tree depth.")
    parser.add_argument("--forest-estimators", type=int, default=100, help="Number of trees in the random forest.")
    parser.add_argument("--boosting-estimators", type=int, default=100, help="Number of boosting stages.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    return parser.parse_args()


def build_models(args):
    return {
        "decision_tree": tree.DecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,
            random_state=args.random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=args.forest_estimators,
            criterion=args.criterion,
            random_state=args.random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=args.boosting_estimators,
            random_state=args.random_state,
        ),
    }


def main() -> None:
    args = parse_args()
    configure_plotting()
    data = load_csv(args.data, index_col=0)
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

    models = build_models(args)
    report = {
        "target": args.target,
        "features": args.features,
        "models": {},
    }
    output_dir = ensure_output_dir(args.output_dir)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (name, model) in zip(axes, models.items()):
        cv_scores = cross_val_score(model, x_train, y_train, cv=args.cv, scoring="accuracy")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        test_accuracy = accuracy_score(y_test, predictions)
        report["models"][name] = {
            "cv_scores": serialize_scores(cv_scores),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std_x2": round(float(cv_scores.std() * 2), 4),
            "test_accuracy": round(float(test_accuracy), 4),
            "feature_importances": {
                feature: round(float(importance), 6)
                for feature, importance in zip(args.features, model.feature_importances_)
            },
        }

        axis.barh(args.features, model.feature_importances_)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Importance")

    plot_path = output_dir / f"{dataset_stem(args.data)}_tree_feature_importances.png"
    save_figure(figure, plot_path)

    dot_path = output_dir / f"{dataset_stem(args.data)}_decision_tree.dot"
    tree.export_graphviz(
        models["decision_tree"],
        out_file=str(dot_path),
        feature_names=args.features,
        class_names=sorted(str(value) for value in y_data.unique()),
        filled=True,
    )

    report_path = output_dir / f"{dataset_stem(args.data)}_tree_models.json"
    write_json_report(report, report_path)

    for name, metrics_payload in report["models"].items():
        print(
            f"{name}: cv={metrics_payload['cv_mean']} (+/- {metrics_payload['cv_std_x2']}), "
            f"test={metrics_payload['test_accuracy']}"
        )
    print(f"Feature importance plot written to: {plot_path}")
    print(f"Decision tree graph written to: {dot_path}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()