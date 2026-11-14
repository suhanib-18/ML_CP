from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
STREAM_PATH = ROOT / "data" / "raw" / "live_telemetry_stream.csv"
MODEL_PATH = ROOT / "models" / "horizon_stacking_pipeline.pkl"
REPORT_PATH = ROOT / "models" / "evaluation_report.json"

HORIZON = 10
HORIZON_GRID = [1, 3, 5, 10, 15]
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.15
OPENING_BLOCK = 0.25
TARGET = "Target"

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
PHYSICS_COLUMNS = [
    "Mechanical_Power_kW",
    "Thermal_Stress_K",
    "Tool_Wear_Strain_Index",
]
GRADE_COLUMNS = ["Type_L", "Type_M", "Type_H"]
FEATURE_COLUMNS = SENSOR_COLUMNS + PHYSICS_COLUMNS + GRADE_COLUMNS
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
EVENT_FLAG_COLUMNS = ["Machine failure", *FAILURE_MODES]

HDF_THERMAL_LIMIT_K = 8.6
HDF_SPEED_LIMIT_RPM = 1380.0
PWF_LOW_W = 3500.0
PWF_HIGH_W = 9000.0
OSF_LIMITS = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
TWF_WEAR_WINDOW = (200.0, 240.0)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Place ai4i2020.csv in {path.parent}.")
    frame = pd.read_csv(path)
    if "UDI" not in frame.columns:
        raise ValueError("The source archive must include UDI so cycles stay in order.")
    return frame.sort_values("UDI").reset_index(drop=True)


def failure_mode(frame: pd.DataFrame) -> pd.Series:
    flags = frame[FAILURE_MODES]
    mode = flags.idxmax(axis=1).where(flags.sum(axis=1) > 0, "Unlabelled")
    return mode.where(frame["Machine failure"] == 1, "None")


def load_cycles(path: Path = RAW_PATH) -> pd.DataFrame:
    frame = load_raw(path)
    frame["Failure_Mode"] = failure_mode(frame)
    frame["Cycle_ID"] = frame["UDI"].astype(int)
    return frame.drop(columns=["UDI", "Product ID"])


def add_physics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Mechanical_Power_kW"] = frame["Torque [Nm]"] * frame["Rotational speed [rpm]"] / 9550.0
    frame["Thermal_Stress_K"] = frame["Process temperature [K]"] - frame["Air temperature [K]"]
    frame["Tool_Wear_Strain_Index"] = frame["Mechanical_Power_kW"] * frame["Tool wear [min]"]
    return frame


def add_horizon_target(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    failures = frame["Machine failure"].to_numpy()
    future = np.lib.stride_tricks.sliding_window_view(failures[1:], horizon).max(axis=1)
    frame = frame.iloc[: len(future)].copy()
    frame[TARGET] = future.astype(int)
    return frame.reset_index(drop=True)


def encode_variant(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    grades = pd.get_dummies(frame["Type"].astype(str), prefix="Type", dtype=float)
    frame = pd.concat([frame, grades], axis=1)
    for column in GRADE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def prepare(horizon: int = HORIZON, path: Path = RAW_PATH) -> pd.DataFrame:
    frame = encode_variant(add_horizon_target(add_physics(load_cycles(path)), horizon))
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Process parameters missing after engineering: {missing}")
    leaked = [column for column in EVENT_FLAG_COLUMNS if column in FEATURE_COLUMNS]
    if leaked:
        raise RuntimeError(f"Event flags leaked into process parameters: {leaked}")
    return frame


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count = len(frame)
    train_end = int(count * TRAIN_FRACTION)
    validation_end = int(count * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def chronological_folds(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    edges = [0, *(int(n_samples * fraction) for fraction in (OPENING_BLOCK, 0.50, 0.75)), n_samples]
    if len(set(edges)) != len(edges):
        raise RuntimeError("Not enough cycles to build chronological folds.")
    folds = [(np.arange(edges[1], edges[2]), np.arange(edges[0], edges[1]))]
    for index in range(1, 4):
        folds.append((np.arange(0, edges[index]), np.arange(edges[index], edges[index + 1])))
    return folds


def legacy_rule_engine_score(frame: pd.DataFrame) -> np.ndarray:
    torque = frame["Torque [Nm]"].to_numpy(dtype=float)
    speed = frame["Rotational speed [rpm]"].to_numpy(dtype=float)
    wear = frame["Tool wear [min]"].to_numpy(dtype=float)
    thermal = frame["Thermal_Stress_K"].to_numpy(dtype=float)
    power_w = frame["Mechanical_Power_kW"].to_numpy(dtype=float) * 1000.0

    threshold = np.full(len(frame), OSF_LIMITS["M"])
    threshold = np.where(frame["Type_L"].to_numpy(dtype=float) == 1.0, OSF_LIMITS["L"], threshold)
    threshold = np.where(frame["Type_H"].to_numpy(dtype=float) == 1.0, OSF_LIMITS["H"], threshold)

    overstrain = np.clip((wear * torque) / threshold, 0.0, 1.5) / 1.5
    heat = np.clip((HDF_THERMAL_LIMIT_K - thermal) / HDF_THERMAL_LIMIT_K, 0.0, 1.0) * np.clip(
        (HDF_SPEED_LIMIT_RPM - speed) / HDF_SPEED_LIMIT_RPM, 0.0, 1.0
    )
    power_risk = np.maximum(
        np.clip((PWF_LOW_W - power_w) / PWF_LOW_W, 0.0, 1.0),
        np.clip((power_w - PWF_HIGH_W) / PWF_HIGH_W, 0.0, 1.0),
    )
    wear_risk = np.clip((wear - 150.0) / (TWF_WEAR_WINDOW[1] - 150.0), 0.0, 1.0)
    return np.clip(np.vstack([overstrain, heat, power_risk, wear_risk]).max(axis=0), 0.0, 1.0)
