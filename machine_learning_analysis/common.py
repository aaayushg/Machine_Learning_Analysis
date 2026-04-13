#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RANDOM_STATE = 42
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where plots and reports will be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducible splits and models.",
    )
    return parser


def configure_plotting() -> None:
    sns.set_theme(style="darkgrid")


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_stem(data_path: str | Path) -> str:
    return Path(data_path).stem


def load_csv(
    data_path: str | Path,
    *,
    header: int | None = 0,
    index_col: int | None = None,
) -> pd.DataFrame:
    csv_path = Path(data_path)
    resolved_path = resolve_data_path(csv_path)
    return pd.read_csv(resolved_path, header=header, index_col=index_col)


def resolve_data_path(data_path: str | Path) -> Path:
    csv_path = Path(data_path)
    if csv_path.exists():
        return csv_path

    fallback_candidates = [csv_path.name]
    if csv_path.suffix != ".csv":
        fallback_candidates.append(f"{csv_path.name}.csv")
    elif csv_path.stem:
        fallback_candidates.append(f"{csv_path.stem}.csv")

    for candidate in fallback_candidates:
        fallback_path = DEFAULT_DATA_DIR / candidate
        if fallback_path.exists():
            return fallback_path

    raise FileNotFoundError(f"Input CSV not found: {csv_path}")


def require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")


def save_figure(figure: plt.Figure, output_path: str | Path, *, dpi: int = 300) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_json_report(payload: dict, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def print_dataframe_summary(dataframe: pd.DataFrame) -> None:
    print(f"{len(dataframe.index)} rows and {len(dataframe.columns)} columns")
    print("")


def serialize_scores(values: Iterable[float]) -> list[float]:
    return [round(float(value), 4) for value in values]