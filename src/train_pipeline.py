"""Commission the horizon diagnostic core for the machining-cell console."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.blend import OpeningBlockLogisticRegression

RAW_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
STREAM_PATH = ROOT / "data" / "raw" / "live_telemetry_stream.csv"
MODEL_PATH = ROOT / "models" / "horizon_stacking_pipeline.pkl"

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
EVENT_FLAG_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
POWER_Z_EDGES = np.array([-np.inf, -1.0, 1.0, np.inf], dtype=float)
OPENING_BLOCK = 0.25


def load_cycles(path: Path) -> pd.DataFrame:
    """Load plant cycles in time order and remove identifiers plus stale columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Place ai4i2020.csv in {path.parent} before commissioning the core."
        )

    frame = pd.read_csv(path)
    if "UDI" not in frame.columns:
        raise ValueError("The source archive must include UDI so cycles stay in order.")

    frame = frame.sort_values("UDI").reset_index(drop=True)
    stale = [
        column
        for column in frame.columns
        if "kmeans" in column.lower() or "rolling_avg" in column.lower()
    ]
    frame = frame.drop(columns=stale, errors="ignore")
    frame["Cycle_ID"] = frame["UDI"].astype(int)
    return frame.drop(columns=["UDI", "Product ID"], errors="ignore")


def add_physics(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized first-principles drivers used by the diagnostic core."""
    frame = frame.copy()
    frame["Mechanical_Power_kW"] = (
        frame["Torque [Nm]"] * frame["Rotational speed [rpm]"] / 9550.0
    )
    frame["Thermal_Stress_K"] = (
        frame["Process temperature [K]"] - frame["Air temperature [K]"]
    )
    frame["Tool_Wear_Strain_Index"] = (
        frame["Mechanical_Power_kW"] * frame["Tool wear [min]"]
    )
    return frame


def add_horizon_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Label each cycle if a failure occurs on any of the next five cycles."""
    frame = frame.copy()
    failures = frame["Machine failure"].to_numpy()
    frame["Target_Next_5_Cycles"] = (
        frame["Machine failure"].shift(-5).rolling(window=5, min_periods=1).max()
    )

    lookahead = frame["Target_Next_5_Cycles"].to_numpy()
    for index in range(0, len(frame) - 5, 1000):
        expected = float(failures[index + 1 : index + 6].max())
        if lookahead[index] != expected:
            raise RuntimeError(
                f"Horizon label mismatch at cycle index {index}: "
                f"{lookahead[index]} != {expected}"
            )

    # The last five cycles do not have a complete five-cycle lookahead.
    frame = frame.iloc[:-5].copy()
    frame = frame.dropna(subset=["Target_Next_5_Cycles"]).reset_index(drop=True)
    frame["Target_Next_5_Cycles"] = frame["Target_Next_5_Cycles"].astype(int)
    return frame


def encode_variant(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    grades = pd.get_dummies(frame["Type"].astype(str), prefix="Type", dtype=float)
    frame = pd.concat([frame.drop(columns=["Type"]), grades], axis=1)
    for column in GRADE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80% commission / 15% historical validation / 5% unseen live stream."""
    count = len(frame)
    train_end = int(count * 0.80)
    validation_end = int(count * 0.95)
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def legacy_rule_engine_score(frame: pd.DataFrame) -> np.ndarray:
    """Continuous score from the fixed overstrain, heat, power, and wear limits."""
    torque = frame["Torque [Nm]"].to_numpy(dtype=float)
    speed = frame["Rotational speed [rpm]"].to_numpy(dtype=float)
    wear = frame["Tool wear [min]"].to_numpy(dtype=float)
    thermal = frame["Thermal_Stress_K"].to_numpy(dtype=float)
    power_w = frame["Mechanical_Power_kW"].to_numpy(dtype=float) * 1000.0

    threshold = np.full(len(frame), 12000.0)
    if "Type_L" in frame.columns:
        threshold = np.where(frame["Type_L"].to_numpy(dtype=float) == 1.0, 11000.0, threshold)
    if "Type_H" in frame.columns:
        threshold = np.where(frame["Type_H"].to_numpy(dtype=float) == 1.0, 13000.0, threshold)

    overstrain = np.clip((wear * torque) / threshold, 0.0, 1.5) / 1.5
    heat = np.clip((8.6 - thermal) / 8.6, 0.0, 1.0) * np.clip(
        (1380.0 - speed) / 1380.0, 0.0, 1.0
    )
    power_risk = np.maximum(
        np.clip((3500.0 - power_w) / 3500.0, 0.0, 1.0),
        np.clip((power_w - 9000.0) / 9000.0, 0.0, 1.0),
    )
    wear_risk = np.clip((wear - 150.0) / 90.0, 0.0, 1.0)
    return np.clip(np.vstack([overstrain, heat, power_risk, wear_risk]).max(axis=0), 0.0, 1.0)


def chronological_folds(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition cycles into contiguous blocks.

    Later blocks are scored only from earlier cycles. The opening block is
    scored from the following block so every cycle still receives one
    out-of-fold score, which stacking requires.
    """
    edges = [
        0,
        *(int(n_samples * fraction) for fraction in (OPENING_BLOCK, 0.50, 0.75)),
        n_samples,
    ]
    if len(set(edges)) != len(edges):
        raise RuntimeError("Not enough cycles to build chronological folds.")

    opening_train = np.arange(edges[1], edges[2])
    opening_test = np.arange(edges[0], edges[1])
    folds = [(opening_train, opening_test)]
    for index in range(1, 4):
        train_index = np.arange(0, edges[index])
        test_index = np.arange(edges[index], edges[index + 1])
        folds.append((train_index, test_index))
    return folds


def build_pipeline(
    scale_pos_weight: float,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> Pipeline:
    trees = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        max_delta_step=1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    geometric = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42,
        cache_size=2000,
    )
    deep = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        random_state=42,
    )
    core = StackingClassifier(
        estimators=[
            ("trees", trees),
            ("geometric", geometric),
            ("deep", deep),
        ],
        final_estimator=OpeningBlockLogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        stack_method="predict_proba",
        cv=folds,
        n_jobs=1,
        passthrough=False,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("core", core),
        ]
    )


def main() -> None:
    started = time.perf_counter()
    cycles = load_cycles(RAW_PATH)
    cycles = add_physics(cycles)
    cycles = add_horizon_target(cycles)
    cycles = encode_variant(cycles)

    missing = [column for column in FEATURE_COLUMNS if column not in cycles.columns]
    if missing:
        raise RuntimeError(f"Process parameters missing after engineering: {missing}")

    train, validation, holdout = chronological_split(cycles)
    leaked = [column for column in EVENT_FLAG_COLUMNS if column in FEATURE_COLUMNS]
    if leaked:
        raise RuntimeError(f"Event flags leaked into process parameters: {leaked}")

    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    holdout[["Cycle_ID", *FEATURE_COLUMNS]].to_csv(STREAM_PATH, index=False)

    x_train = train[FEATURE_COLUMNS].astype(float)
    y_train = train["Target_Next_5_Cycles"].astype(int)
    x_validation = validation[FEATURE_COLUMNS].astype(float)
    y_validation = validation["Target_Next_5_Cycles"].astype(int)

    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    if positive == 0:
        raise RuntimeError("The commissioning window has no impending-failure cycles.")
    scale_pos_weight = negative / positive

    print(
        f"Cycles ready | train {len(train)} | validation {len(validation)} | "
        f"holdout {len(holdout)} | scale_pos_weight {scale_pos_weight:.3f}"
    )
    print("Fitting horizon stacking pipeline...")

    pipeline = build_pipeline(scale_pos_weight, chronological_folds(len(x_train)))
    pipeline.fit(x_train, y_train)

    core_scores = pipeline.predict_proba(x_validation)[:, 1]
    legacy_scores = legacy_rule_engine_score(validation)
    core_precision, core_recall, _ = precision_recall_curve(y_validation, core_scores)
    legacy_precision, legacy_recall, _ = precision_recall_curve(y_validation, legacy_scores)
    core_ap = float(average_precision_score(y_validation, core_scores))
    legacy_ap = float(average_precision_score(y_validation, legacy_scores))
    recall_marks = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    def _precision_at(recall: np.ndarray, precision: np.ndarray) -> np.ndarray:
        order = np.argsort(recall)
        return np.interp(recall_marks, recall[order], precision[order])

    core_at = _precision_at(core_recall, core_precision)
    legacy_at = _precision_at(legacy_recall, legacy_precision)
    print("Blend coefficients:", pipeline.named_steps["core"].final_estimator_.coef_)
    print("Precision across the operating range:")
    for mark, core_value, legacy_value in zip(recall_marks, core_at, legacy_at):
        print(f"  recall {mark:.1f} | core {core_value:.3f} | legacy {legacy_value:.3f}")
    if core_ap <= legacy_ap or float(np.mean(core_at - legacy_at)) < 0.03:
        raise RuntimeError(
            "Autonomous diagnostic core did not clearly outperform the legacy rule engine "
            f"on historical validation ({core_ap:.3f} vs {legacy_ap:.3f})."
        )

    power = x_train["Mechanical_Power_kW"].to_numpy(dtype=float)
    pipeline.feature_columns_ = list(FEATURE_COLUMNS)
    pipeline.power_reference_mean_ = float(power.mean())
    pipeline.power_reference_std_ = float(power.std(ddof=0))
    pipeline.power_reference_z_edges_ = POWER_Z_EDGES
    pipeline.power_reference_expected_ = np.diff(norm.cdf(POWER_Z_EDGES))
    pipeline.historical_precision_core_ = np.asarray(core_precision, dtype=float)
    pipeline.historical_recall_core_ = np.asarray(core_recall, dtype=float)
    pipeline.historical_precision_legacy_ = np.asarray(legacy_precision, dtype=float)
    pipeline.historical_recall_legacy_ = np.asarray(legacy_recall, dtype=float)

    influence = np.asarray(
        pipeline.named_steps["core"].named_estimators_["trees"].feature_importances_,
        dtype=float,
    )
    if influence.shape[0] != len(FEATURE_COLUMNS):
        raise RuntimeError(
            f"Influence width {influence.shape[0]} does not match "
            f"{len(FEATURE_COLUMNS)} process parameters."
        )

    holdout_scores = pipeline.predict_proba(holdout[FEATURE_COLUMNS].astype(float))
    if holdout_scores.ndim != 2 or holdout_scores.shape[1] != 2:
        raise RuntimeError(f"Unexpected score shape {holdout_scores.shape}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    restored = joblib.load(MODEL_PATH)
    required = [
        "feature_columns_",
        "power_reference_mean_",
        "power_reference_std_",
        "power_reference_z_edges_",
        "power_reference_expected_",
        "historical_precision_core_",
        "historical_recall_core_",
        "historical_precision_legacy_",
        "historical_recall_legacy_",
    ]
    missing_attrs = [name for name in required if not hasattr(restored, name)]
    if missing_attrs:
        raise RuntimeError(f"Serialized core is missing {missing_attrs}")

    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import joblib, sys; joblib.load(sys.argv[1])",
            str(MODEL_PATH),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Serialized core cannot be reloaded:\n{probe.stderr}")

    ranking = sorted(
        zip(FEATURE_COLUMNS, influence),
        key=lambda item: item[1],
        reverse=True,
    )
    print("Process parameter influence:")
    for name, weight in ranking:
        print(f"  {name}: {weight:.4f}")
    print(
        "Historical average precision | "
        f"core {core_ap:.3f} | "
        f"legacy {legacy_ap:.3f}"
    )
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {STREAM_PATH}")
    print(f"Commissioned in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
