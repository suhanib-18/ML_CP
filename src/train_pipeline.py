from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV

from src.data import (
    FEATURE_COLUMNS,
    HORIZON,
    HORIZON_GRID,
    MODEL_PATH,
    REPORT_PATH,
    STREAM_PATH,
    TARGET,
    chronological_folds,
    chronological_split,
    legacy_rule_engine_score,
    prepare,
)
from src.evaluation import at_threshold, best_threshold, pr_curve, summarize
from src.models import CANDIDATES, build_stack, positive_scores
from src.monitoring import AlertPolicy, event_metrics, quantile_edges, raw_levels, validate_series

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

STACK_NAME = "Stacked Ensemble"
LEGACY_NAME = "Legacy Rule Engine"
BLEND_WINDOWS = [0.25, 0.40, 0.55, 0.65, 0.75]
ALERT_GRID = {
    "alpha": [0.4, 0.6, 0.8],
    "confirm": [2, 3],
    "release": [2, 3, 4],
    "hysteresis": [0.8, 0.9],
}


def plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float), frame[TARGET].to_numpy(dtype=int)


def split_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "positives": int(frame[TARGET].sum()),
        "base_rate": float(frame[TARGET].mean()),
        "failure_events": int(frame["Machine failure"].sum()),
        "first_cycle": int(frame["Cycle_ID"].min()),
        "last_cycle": int(frame["Cycle_ID"].max()),
    }


def tune_base_learners(x_train, y_train, folds, scale_pos_weight) -> tuple[dict, dict]:
    tuned, log = {}, {}
    for name, (factory, grid) in CANDIDATES.items():
        estimator = factory()
        if name == "XGBoost":
            estimator.set_params(scale_pos_weight=scale_pos_weight)
        search = GridSearchCV(estimator, grid, scoring="average_precision", cv=folds[1:], n_jobs=-1, refit=True)
        search.fit(x_train, y_train)
        tuned[name] = search.best_estimator_
        log[name] = {
            "best_params": {key.split("__")[-1]: plain(value) for key, value in search.best_params_.items()},
            "cv_ap": float(search.best_score_),
        }
        print(f"  {name:10s} cv AP {search.best_score_:.3f} | {log[name]['best_params']}")
    return tuned, log


def select_blend_window(trees, geometric, folds, x_train, y_train, x_validation, y_validation):
    best = None
    for fraction in BLEND_WINDOWS:
        stack = build_stack(clone(trees), clone(geometric), folds, ignore_fraction=fraction).fit(x_train, y_train)
        ap = float(average_precision_score(y_validation, positive_scores(stack, x_validation)))
        print(f"  ignore first {fraction:.0%} | validation AP {ap:.3f}")
        if best is None or ap > best[2]:
            best = (stack, fraction, ap)
    return best[0], best[1]


def thresholds_for(y, scores) -> tuple[float, float]:
    warning = best_threshold(y, scores, beta=2.0)
    critical = max(best_threshold(y, scores, beta=1.0), warning)
    return warning, critical


def select_alert_policy(scores, failures, warning, critical, horizon) -> AlertPolicy:
    best, best_key = None, None
    for alpha, confirm, release, hysteresis in itertools.product(*ALERT_GRID.values()):
        policy = AlertPolicy(warning, critical, alpha, confirm, release, hysteresis)
        levels, _ = validate_series(scores, policy)
        metrics = event_metrics(levels, failures, horizon)
        key = (metrics["warned"] - metrics["false_episodes"], -metrics["state_changes"], metrics["mean_lead_cycles"])
        if best_key is None or key > best_key:
            best, best_key = policy, key
    return best


def alert_summary(scores, failures, policy: AlertPolicy, horizon: int) -> dict:
    validated, _ = validate_series(scores, policy)
    return {
        "raw": event_metrics(raw_levels(scores, policy), failures, horizon),
        "validated": event_metrics(validated, failures, horizon),
    }


def horizon_study(trees, geometric, policy: AlertPolicy, shipped_stack, ignore_fraction: float) -> list[dict]:
    rows = []
    for horizon in HORIZON_GRID:
        train, validation, holdout = chronological_split(prepare(horizon))
        x_train, y_train = matrices(train)
        x_validation, y_validation = matrices(validation)
        x_holdout, y_holdout = matrices(holdout)
        if horizon == HORIZON:
            stack = shipped_stack
        else:
            weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            stack = build_stack(
                clone(trees).set_params(scale_pos_weight=weight),
                clone(geometric),
                chronological_folds(len(x_train)),
                ignore_fraction=ignore_fraction,
            ).fit(x_train, y_train)
        validation_scores = positive_scores(stack, x_validation)
        holdout_scores = positive_scores(stack, x_holdout)
        warning, critical = thresholds_for(y_validation, validation_scores)
        study_policy = AlertPolicy(warning, critical, policy.alpha, policy.confirm, policy.release, policy.hysteresis)
        levels, _ = validate_series(holdout_scores, study_policy)
        events = event_metrics(levels, holdout["Machine failure"].to_numpy(), horizon)
        row = {
            "horizon": horizon,
            "holdout_ap": float(average_precision_score(y_holdout, holdout_scores)),
            "warned_rate": events["warned_rate"],
            "lead": events["mean_lead_cycles"],
            "false_episodes": events["false_episodes"],
        }
        rows.append(row)
        print(
            f"  horizon {horizon:2d} | holdout AP {row['holdout_ap']:.3f} "
            f"| warned {row['warned_rate']:.0%} | lead {row['lead']:.1f} | false {row['false_episodes']}"
        )
    return rows


def main() -> None:
    started = time.perf_counter()
    frame = prepare(HORIZON)
    train, validation, holdout = chronological_split(frame)
    x_train, y_train = matrices(train)
    x_validation, y_validation = matrices(validation)
    x_holdout, y_holdout = matrices(holdout)

    positive = int((y_train == 1).sum())
    if positive == 0:
        raise RuntimeError("The training window has no impending-failure cycles.")
    scale_pos_weight = float((y_train == 0).sum() / positive)
    folds = chronological_folds(len(x_train))

    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    stream = holdout[["Cycle_ID", *FEATURE_COLUMNS, "Machine failure", "Failure_Mode"]].rename(
        columns={"Machine failure": "Failure_Event"}
    )
    stream.to_csv(STREAM_PATH, index=False)

    print(
        f"Horizon {HORIZON} | train {len(train)} | validation {len(validation)} | holdout {len(holdout)} "
        f"| positive rate {y_train.mean():.3f} | scale_pos_weight {scale_pos_weight:.3f}"
    )
    print("Tuning base learners with forward-chaining folds...")
    tuned, tuning = tune_base_learners(x_train, y_train, folds, scale_pos_weight)
    trees = clone(tuned["XGBoost"])
    geometric = clone(tuned["SVM (RBF)"])

    print("Selecting the meta-learner window...")
    stack, ignore_fraction = select_blend_window(trees, geometric, folds, x_train, y_train, x_validation, y_validation)

    stack_validation = positive_scores(stack, x_validation)
    stack_holdout = positive_scores(stack, x_holdout)
    legacy_validation = legacy_rule_engine_score(validation)
    legacy_holdout = legacy_rule_engine_score(holdout)

    comparison = {
        STACK_NAME: {
            "validation": summarize(y_validation, stack_validation),
            "holdout": summarize(y_holdout, stack_holdout),
        },
        LEGACY_NAME: {
            "validation": summarize(y_validation, legacy_validation),
            "holdout": summarize(y_holdout, legacy_holdout),
        },
    }
    core_ap = comparison[STACK_NAME]["validation"]["average_precision"]
    legacy_ap = comparison[LEGACY_NAME]["validation"]["average_precision"]
    if core_ap <= legacy_ap:
        raise RuntimeError(
            f"Stacked ensemble did not outperform the legacy rule engine on validation ({core_ap:.3f} vs {legacy_ap:.3f})."
        )

    warning, critical = thresholds_for(y_validation, stack_validation)
    legacy_warning = best_threshold(y_validation, legacy_validation, beta=2.0)
    policy = select_alert_policy(stack_validation, validation["Machine failure"].to_numpy(), warning, critical, HORIZON)
    alerts = {
        "policy": policy.as_dict(),
        "validation": alert_summary(stack_validation, validation["Machine failure"].to_numpy(), policy, HORIZON),
        "holdout": alert_summary(stack_holdout, holdout["Machine failure"].to_numpy(), policy, HORIZON),
    }

    print("Horizon study...")
    horizons = horizon_study(trees, geometric, policy, stack, ignore_fraction)

    power_edges = quantile_edges(train["Mechanical_Power_kW"].to_numpy(dtype=float), bins=4)
    stack.feature_columns_ = list(FEATURE_COLUMNS)
    stack.horizon_ = HORIZON
    stack.alert_policy_ = policy.as_dict()
    stack.power_edges_ = power_edges
    stack.power_expected_ = np.full(4, 0.25)
    stack.trained_at_ = datetime.now().isoformat(timespec="seconds")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stack, MODEL_PATH)
    probe = subprocess.run(
        [sys.executable, "-c", "import joblib, sys; joblib.load(sys.argv[1])", str(MODEL_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Serialized model cannot be reloaded:\n{probe.stderr}")

    report = {
        "generated_at": stack.trained_at_,
        "horizon": HORIZON,
        "features": list(FEATURE_COLUMNS),
        "scale_pos_weight": scale_pos_weight,
        "splits": {
            "train": split_summary(train),
            "validation": split_summary(validation),
            "holdout": split_summary(holdout),
        },
        "tuning": tuning,
        "ignore_fraction": ignore_fraction,
        "comparison": comparison,
        "curves": {STACK_NAME: pr_curve(y_holdout, stack_holdout), LEGACY_NAME: pr_curve(y_holdout, legacy_holdout)},
        "thresholds": {"warning": warning, "critical": critical},
        "confusion": {
            STACK_NAME: at_threshold(y_holdout, stack_holdout, warning),
            LEGACY_NAME: at_threshold(y_holdout, legacy_holdout, legacy_warning),
        },
        "alerts": alerts,
        "horizons": horizons,
    }
    report["duration_s"] = time.perf_counter() - started
    REPORT_PATH.write_text(json.dumps(report, indent=1, default=plain), encoding="utf-8")

    for name, entry in comparison.items():
        print(
            f"{name:20s} | validation AP {entry['validation']['average_precision']:.3f} "
            f"| holdout AP {entry['holdout']['average_precision']:.3f} ROC-AUC {entry['holdout']['roc_auc']:.3f}"
        )
    print(f"Thresholds | warning {warning:.3f} | critical {critical:.3f}")
    print(f"Alert policy | {policy.as_dict()}")
    for split in ("validation", "holdout"):
        raw, checked = alerts[split]["raw"], alerts[split]["validated"]
        print(
            f"  {split:10s} raw: warned {raw['warned']}/{raw['failure_events']} false {raw['false_episodes']} "
            f"changes {raw['state_changes']} | validated: warned {checked['warned']}/{checked['failure_events']} "
            f"false {checked['false_episodes']} changes {checked['state_changes']} lead {checked['mean_lead_cycles']:.1f}"
        )
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {REPORT_PATH}")
    print(f"Saved {STREAM_PATH}")
    print(f"Completed in {report['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
