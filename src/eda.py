from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, skew

from src.data import (
    FAILURE_MODES,
    HORIZON,
    HORIZON_GRID,
    PHYSICS_COLUMNS,
    RAW_PATH,
    SENSOR_COLUMNS,
    TARGET,
    add_horizon_target,
    add_physics,
    chronological_split,
    load_cycles,
    load_raw,
    prepare,
)

NUMERIC_COLUMNS = SENSOR_COLUMNS + PHYSICS_COLUMNS
SEGMENT_SIZE = 1000
ROLLING_WINDOW = 250
AUTOCORRELATION_LAGS = [1, 5, 10, 50]


def load_frame(path: Path = RAW_PATH) -> pd.DataFrame:
    return add_physics(load_cycles(path))


def overview(raw: pd.DataFrame) -> dict:
    return {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "sensor_features": len(SENSOR_COLUMNS),
        "failures": int(raw["Machine failure"].sum()),
        "failure_rate": float(raw["Machine failure"].mean()),
        "missing_values": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "imbalance_ratio": float((raw["Machine failure"] == 0).sum() / raw["Machine failure"].sum()),
    }


def schema(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": raw.columns,
            "Type": [str(dtype) for dtype in raw.dtypes],
            "Unique": [int(raw[column].nunique()) for column in raw.columns],
            "Missing": [int(raw[column].isna().sum()) for column in raw.columns],
        }
    )


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame[NUMERIC_COLUMNS].describe().T
    table["skew"] = [float(skew(frame[column])) for column in NUMERIC_COLUMNS]
    return table.round(3)


def class_balance(frame: pd.DataFrame) -> pd.Series:
    return frame["Machine failure"].map({0: "No Failure", 1: "Failure"}).value_counts()


def mode_counts(raw: pd.DataFrame) -> pd.DataFrame:
    flags = raw[FAILURE_MODES]
    failed = raw["Machine failure"] == 1
    return pd.DataFrame(
        {
            "Mode": FAILURE_MODES,
            "Count": [int(flags[mode].sum()) for mode in FAILURE_MODES],
            "Without Machine Failure": [int((flags[mode].eq(1) & ~failed).sum()) for mode in FAILURE_MODES],
        }
    )


def label_consistency(raw: pd.DataFrame) -> dict:
    flags = raw[FAILURE_MODES].sum(axis=1)
    failed = raw["Machine failure"] == 1
    return {
        "failure_without_mode": int((failed & flags.eq(0)).sum()),
        "mode_without_failure": int((~failed & flags.gt(0)).sum()),
        "multiple_modes": int(flags.gt(1).sum()),
    }


def type_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby("Type")["Machine failure"].agg(["count", "sum", "mean"]).reindex(["L", "M", "H"])
    summary.columns = ["Cycles", "Failures", "Failure Rate"]
    return summary


def class_separation(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    failed = frame["Machine failure"] == 1
    for column in NUMERIC_COLUMNS:
        positive = frame.loc[failed, column]
        negative = frame.loc[~failed, column]
        pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2)
        rows.append(
            {
                "Feature": column,
                "Mean (Failure)": float(positive.mean()),
                "Mean (No Failure)": float(negative.mean()),
                "Cohen d": float((positive.mean() - negative.mean()) / pooled),
                "Mann-Whitney p": float(mannwhitneyu(positive, negative).pvalue),
            }
        )
    return pd.DataFrame(rows)


def correlation(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    labelled = add_horizon_target(frame, horizon).rename(columns={TARGET: f"Failure in next {horizon}"})
    columns = NUMERIC_COLUMNS + ["Machine failure", f"Failure in next {horizon}"]
    return labelled[columns].corr().round(3)


def outlier_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_COLUMNS:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        spread = q3 - q1
        mask = (frame[column] < q1 - 1.5 * spread) | (frame[column] > q3 + 1.5 * spread)
        rows.append(
            {
                "Feature": column,
                "Outliers": int(mask.sum()),
                "Failure Rate (Outliers)": float(frame.loc[mask, "Machine failure"].mean()) if mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def segment_modes(frame: pd.DataFrame, size: int = SEGMENT_SIZE) -> pd.DataFrame:
    segment = ((frame["Cycle_ID"] - 1) // size) * size + 1
    failures = frame[frame["Machine failure"] == 1]
    table = pd.crosstab(segment[failures.index], failures["Failure_Mode"])
    table = table.reindex(sorted(segment.unique()), fill_value=0)
    table.index = [f"{start}-{start + size - 1}" for start in table.index]
    return table


def rolling_failure_rate(frame: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Cycle_ID": frame["Cycle_ID"],
            "Failure Rate": frame["Machine failure"].rolling(window, min_periods=window // 2).mean(),
        }
    )


def autocorrelation(frame: pd.DataFrame, lags=AUTOCORRELATION_LAGS) -> pd.DataFrame:
    columns = NUMERIC_COLUMNS + ["Machine failure"]
    return pd.DataFrame(
        {f"lag {lag}": [float(frame[column].astype(float).autocorr(lag)) for column in columns] for lag in lags},
        index=columns,
    ).round(3)


def horizon_positive_rates(frame: pd.DataFrame, horizons=HORIZON_GRID) -> pd.DataFrame:
    rows = [{"Horizon": 0, "Positive Rate": float(frame["Machine failure"].mean())}]
    for horizon in horizons:
        rows.append({"Horizon": horizon, "Positive Rate": float(add_horizon_target(frame, horizon)[TARGET].mean())})
    return pd.DataFrame(rows)


def split_profile(horizon: int = HORIZON) -> pd.DataFrame:
    train, validation, holdout = chronological_split(prepare(horizon))
    rows = []
    for name, part in (("Train", train), ("Validation", validation), ("Holdout", holdout)):
        rows.append(
            {
                "Split": name,
                "First Cycle": int(part["Cycle_ID"].min()),
                "Last Cycle": int(part["Cycle_ID"].max()),
                "Rows": int(len(part)),
                "Positive Rate": float(part[TARGET].mean()),
                "Failure Events": int(part["Machine failure"].sum()),
                "HDF Events": int((part["Failure_Mode"] == "HDF").sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    raw = load_raw()
    frame = load_frame()
    sections = [
        ("Overview", pd.Series(overview(raw))),
        ("Schema", schema(raw)),
        ("Descriptive statistics", describe(frame)),
        ("Class balance", class_balance(frame)),
        ("Failure modes", mode_counts(raw)),
        ("Label consistency", pd.Series(label_consistency(raw))),
        ("Product grade", type_summary(frame)),
        ("Class separation", class_separation(frame)),
        ("IQR outliers", outlier_counts(frame)),
        ("Correlation", correlation(frame)),
        ("Failure modes per 1000 cycles", segment_modes(frame)),
        ("Autocorrelation", autocorrelation(frame)),
        ("Positive rate by horizon", horizon_positive_rates(frame)),
        ("Chronological split", split_profile()),
    ]
    for title, table in sections:
        print(f"\n=== {title} ===")
        print(table.to_string())


if __name__ == "__main__":
    main()
