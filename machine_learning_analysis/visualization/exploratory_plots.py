#!/usr/bin/env python3

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from machine_learning_analysis.common import (
    build_parser,
    configure_plotting,
    dataset_stem,
    ensure_output_dir,
    load_csv,
    print_dataframe_summary,
    require_columns,
    write_json_report,
)


def parse_args():
    parser = build_parser("Generate pairplots, jointplots, and simple linear regression diagnostics.")
    parser.add_argument("--header", action="store_true", help="Treat the CSV as having a header row.")
    parser.add_argument(
        "--index-col",
        type=int,
        default=0,
        help="Column index to use as the dataframe index when --header is supplied.",
    )
    parser.add_argument("--pairplot", action="store_true", help="Render a seaborn pairplot.")
    parser.add_argument("--pairplot-columns", nargs="+", help="Columns to include in the pairplot.")
    parser.add_argument("--jointplot", nargs=2, metavar=("X", "Y"), help="Render a jointplot for two columns.")
    parser.add_argument(
        "--jointplot-kind",
        choices=["scatter", "kde", "hist", "hex", "reg", "resid"],
        default="kde",
        help="Jointplot rendering style.",
    )
    parser.add_argument(
        "--regression",
        nargs=2,
        metavar=("X", "Y"),
        help="Fit a simple linear regression with one predictor and one target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plotting()
    header = 0 if args.header else None
    index_col = args.index_col if args.header else None
    data = load_csv(args.data, header=header, index_col=index_col)
    print_dataframe_summary(data)

    output_dir = ensure_output_dir(args.output_dir)
    report = {
        "rows": len(data.index),
        "columns": [str(column) for column in data.columns],
    }

    if args.pairplot:
        pairplot_columns = args.pairplot_columns or list(data.columns)
        require_columns(data, pairplot_columns)
        pair_grid = sns.pairplot(data[pairplot_columns], diag_kind="hist")
        pairplot_path = output_dir / f"{dataset_stem(args.data)}_pairplot.png"
        pair_grid.savefig(pairplot_path)
        plt.close(pair_grid.figure)
        report["pairplot"] = str(pairplot_path)
        print(f"Pairplot written to: {pairplot_path}")

    if args.jointplot:
        x_var, y_var = args.jointplot
        require_columns(data, [x_var, y_var])
        joint_grid = sns.jointplot(data=data, x=x_var, y=y_var, kind=args.jointplot_kind)
        jointplot_path = output_dir / f"{dataset_stem(args.data)}_{x_var}_{y_var}_jointplot.png"
        joint_grid.savefig(jointplot_path)
        plt.close(joint_grid.figure)
        report["jointplot"] = {
            "x": x_var,
            "y": y_var,
            "kind": args.jointplot_kind,
            "path": str(jointplot_path),
        }
        print(f"Jointplot written to: {jointplot_path}")

    if args.regression:
        x_var, y_var = args.regression
        require_columns(data, [x_var, y_var])
        x_values = data[[x_var]].to_numpy()
        y_values = data[y_var].to_numpy()
        model = LinearRegression()
        model.fit(x_values, y_values)
        predictions = model.predict(x_values)
        regression_payload = {
            "x": x_var,
            "y": y_var,
            "slope": round(float(model.coef_.ravel()[0]), 6),
            "intercept": round(float(np.ravel(model.intercept_)[0]), 6) if np.ndim(model.intercept_) else round(float(model.intercept_), 6),
            "mse": round(float(mean_squared_error(y_values, predictions)), 6),
            "r2": round(float(r2_score(y_values, predictions)), 6),
        }
        report["regression"] = regression_payload
        print(f"Regression: y = {regression_payload['slope']} * x + {regression_payload['intercept']}")
        print(f"MSE: {regression_payload['mse']}  R2: {regression_payload['r2']}")

    report_path = output_dir / f"{dataset_stem(args.data)}_eda.json"
    write_json_report(report, report_path)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
