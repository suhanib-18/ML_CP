# ORION AIS — Asset Integrity Console

Predictive maintenance for a machining cell. The system predicts whether the machine will fail **within the next 10 cycles**, scores a simulated live sensor stream cycle by cycle, confirms alerts through a validation layer before raising them, and monitors sensor drift.

---

## 1. How to run

```powershell
cd ML_CP
pip install -r requirements.txt
python -m src.eda              # prints the exploratory data analysis
python -m src.train_pipeline   # trains, validates and saves the model + report (~75 s)
streamlit run app.py           # http://localhost:8501
```

---

## 2. Architecture

```
ai4i2020.csv
   │
   ├── src/data.py ─────── clean, physics features, 10-cycle label, chronological split
   │
   ├── src/eda.py ──────── exploratory analysis (EDA page)
   │
   └── src/train_pipeline.py
          │  tune XGBoost + SVM (src/models.py)
          │  stack them with a logistic-regression meta-learner
          │  compare against the Legacy Rule Engine (baseline)
          │  pick warning / critical thresholds (src/evaluation.py)
          │  tune the alert validation layer (src/monitoring.py)
          │  horizon study (1, 3, 5, 10, 15 cycles)
          ▼
   models/horizon_stacking_pipeline.pkl   models/evaluation_report.json   data/raw/live_telemetry_stream.csv
          │                                        │                                 │
          └──────────────── app.py (Streamlit UI) + charts.py (Plotly) ──────────────┘
```

| File | Role |
|------|------|
| `src/data.py` | Loading, cleaning, physics features, horizon label, chronological split and folds, legacy rule baseline |
| `src/eda.py` | Exploratory data analysis |
| `src/models.py` | XGBoost, SVM, opening-block meta-learner, stacking builder |
| `src/evaluation.py` | AP, ROC-AUC, F-beta thresholds, confusion matrix, PR curve |
| `src/monitoring.py` | Alert validation layer, event metrics, PSI drift |
| `src/train_pipeline.py` | End-to-end training and report |
| `charts.py` | Plotly figures |
| `app.py` | UI: Exploratory Data Analysis, Model Evaluation, Active Asset Telemetry |

---

## 3. Dataset and cleaning

AI4I 2020 Predictive Maintenance (UCI): 10,000 consecutive machining cycles.

| Issue | Finding | Action |
|---|---|---|
| Missing values | 0 | None needed |
| Duplicate rows | 0 | None needed |
| `UDI`, `Product ID` | Identifiers | `UDI` sorts rows into time order (kept as `Cycle_ID`). Both are excluded from the model. |
| `Type` (L / M / H) | Categorical | One-hot encoded |
| Failure-mode flags (TWF, HDF, PWF, OSF, RNF) | Outcomes, not sensor readings | Excluded from the inputs (leakage). `prepare()` raises an error if any leak in. |
| Label noise | 9 failures without a mode, 18 mode flags without a failure, 24 failures with multiple modes | Kept as-is. Only `Machine failure` is the target. |
| Outliers (IQR) | Torque outliers fail 90% of the time, power outliers 100% | Kept: outliers are the failures |
| Skewed speed (skew ≈ 2.0) | | Trees don't need scaling. The SVM scales inside its pipeline. |
| Time dependence | Temperature lag-1 autocorrelation 0.999 | Chronological split, no shuffling |
| Last 10 rows | No complete future window | Dropped (9,990 rows remain) |

Key EDA facts:

- **Imbalance:** 339 failures out of 10,000 rows (3.39%), about 1 : 28.
- **Failure modes:** HDF 115, OSF 98, PWF 95, TWF 46, RNF 19.
- **HDF is concentrated in time:** 108 of the 115 HDF failures fall in cycles 4,001–5,000.

---

## 4. Features

11 inputs:

- **5 sensors:** air temperature, process temperature, speed, torque, tool wear
- **3 one-hot grade columns**
- **3 physics features:**
  - `Thermal_Stress_K` = process − air temperature (heat-dissipation failures)
  - `Mechanical_Power_kW` = torque × speed / 9550 (power failures)
  - `Tool_Wear_Strain_Index` = mechanical power × tool wear (overstrain failures)

---

## 5. Label and split

- **Label:** a cycle is positive if a failure occurs in the next 10 cycles. This raises the positive rate from 3.4% to 24.7% and gives the operator time to act.
- **Split:**
  - Train: cycles 1–7,992
  - Validation: 7,993–9,490
  - Holdout: 9,491–9,990
- **Validation** is used for every choice: hyperparameters, meta-learner window, thresholds and alert policy.
- The **holdout** is only for the final report and the live stream.

---

## 6. Models

| Model | Role |
|---|---|
| **Legacy Rule Engine (baseline)** | Hand-coded physics limits (heat, power, overstrain, tool wear). No learning. |
| **Stacked Ensemble (shipped)** | XGBoost + RBF-SVM, combined by a logistic-regression meta-learner |

How the stack works:

1. **XGBoost** (boosted decision trees) learns hard threshold rules such as "tool wear > 200 min" or "power outside 3.5–9 kW".
   - Tuned: learning rate 0.03, depth 4, min child weight 5, 200 trees.
2. **SVM with RBF kernel** learns smooth, curved boundaries between torque, speed and temperature.
   - Tuned: C = 10, gamma = 0.03.
   - It is scaled with `StandardScaler` inside its pipeline.
3. **Meta-learner** (logistic regression) takes XGBoost's probability and the SVM's decision score and outputs one calibrated failure probability.
   - It is trained on out-of-fold predictions from forward-chaining time folds, so it never sees a base model's prediction on data that model was trained on.
   - **Opening-block weighting:** the first 55% of out-of-fold rows are ignored when fitting the meta-learner, because they contain the HDF burst that never recurs. The fraction is chosen on validation and raises validation AP from 0.414 to 0.444.

| | Validation AP | Holdout AP | Validation ROC-AUC | Holdout ROC-AUC |
|---|---|---|---|---|
| Legacy Rule Engine | 0.400 | 0.447 | 0.818 | 0.789 |
| **Stacked Ensemble** | **0.444** | **0.477** | 0.799 | **0.828** |

At the warning threshold on the holdout:
- The stack catches **73 of 91** at-risk cycles, versus 69 for the baseline.
- It raises **95** false positives, versus 107.

Training refuses to save the model if it does not beat the baseline's validation AP.

---

## 7. Class imbalance

| Where | Technique |
|---|---|
| XGBoost | `scale_pos_weight` = negatives / positives in training (2.68) |
| SVM | `class_weight="balanced"` |
| Meta-learner | Unweighted, so the output remains a true probability (it drives the health index) |
| Metric | Average precision (PR-AUC), not accuracy (accuracy is 96.6% for "never fails") |
| Thresholds | Tuned on validation, not 0.5: warning = F2-optimal (0.184), critical = F1-optimal (0.294) |
| Label | The 10-cycle horizon raises the positive rate from 3.4% to 24.7% |

SMOTE was not used. It would interpolate synthetic sensor states between time-ordered cycles.

---

## 8. Validation layer

`AlertValidator` in `src/monitoring.py`:

1. **EWMA smoothing:** smoothed = 0.6 × risk + 0.4 × previous.
2. **Confirm:** the level rises only after 3 consecutive cycles above the threshold (shown as "VERIFYING n/3").
3. **Release:** the level drops only after 2 consecutive cycles below it.
4. **Hysteresis:** to release, the risk must fall below 0.8 × the threshold.

The policy is grid-searched on validation to maximise failures warned − false alarms, then to minimise status changes.

| Holdout | Raw model | With validation layer |
|---|---|---|
| False alarms | 10 | **1** |
| Status changes | 43 | **22** |
| Failures warned | 13 / 14 | **13 / 14** |
| Mean warning lead | | 8.8 cycles |
| Share of cycles in alert | 34% | 39% |

---

## 9. Prediction horizon (holdout)

| Horizon | 1 | 3 | 5 | **10** | 15 |
|---|---|---|---|---|---|
| Failures warned | 31% | 46% | 77% | **93%** | 93% |
| Mean lead (cycles) | 1.0 | 3.0 | 4.7 | **8.8** | 14.0 |

H = 10 was chosen as a balance: it warns as many holdout failures as H = 15, while a 15-cycle warning is vaguer for the operator. This is a judgment call, not a measured optimum.

---

## 10. UI

Every chart is an interactive Plotly figure.

**Exploratory Data Analysis.** Tiles show cycles, failures, imbalance, missing values and duplicates, followed by five charts:

| Chart | What it proves |
|---|---|
| Class Balance | 3.4% failures, so imbalance handling is needed |
| Effect Size (Cohen d) | Torque, wear strain, power and tool wear separate failures most, so the physics features are useful |
| Torque vs Speed by Failure Mode | Failures sit at physical limits (power envelope, low-speed heat) |
| Failures per 1,000 Cycles | Failures are not uniform in time (HDF burst), so the split must be chronological |
| Chronological Split | Train / validation / holdout ranges |

**Model Evaluation.** Tiles show the horizon, the two thresholds and failures warned, followed by five charts:

| Chart | What it proves |
|---|---|
| Baseline vs Stacked Ensemble | The stack beats the rule engine on AP (and on holdout ROC-AUC) |
| Precision–Recall (Holdout) | From about 40% recall upward, where an alarm system operates, the stack is clearly more precise |
| Confusion Matrix | More failures caught and fewer false positives than the baseline |
| Failures Warned by Horizon | Prediction works 5–10 cycles ahead |
| Raw vs Validated Alerts | The validation layer removes false alarms without losing warnings |

**Active Asset Telemetry.** Replays the 500 holdout cycles as a live stream, one cycle per interval. For each cycle:

1. The model outputs a risk.
2. The validation layer decides the level.
3. Health = 100 × (1 − smoothed risk).
4. The PSI drift check runs on mechanical power over the last 40 cycles.
5. The UI updates the gauge, status banner, counters (raw flags, validated alerts, failures, warned in advance) and the two trend charts.

---

## 11. Limitations

- **The Legacy Rule Engine is a baseline we wrote** from the dataset's documented failure rules. It is not an existing plant system.
- **The holdout is small:** 500 cycles and 14 failures. Bootstrap 95% intervals for holdout AP overlap (stack 0.40–0.59, baseline 0.37–0.57), so the improvement is consistent but not statistically proven.
- **Alerts are on a lot of the time.** The validated system is in Warning or Critical on 39% of holdout cycles (raw: 34%). The validation layer reduces the *number* of separate false alarms and the flicker, but alerts last longer.
- **"Warned" is a generous measure:** a failure counts as warned if any alert was active in the 10 cycles before it.
- **The model is tuned on validation data:** thresholds, the meta-learner window and the alert policy are all chosen there, so validation numbers are optimistic. Holdout numbers are the honest ones.
- **Small label overlap at split boundaries:** the last 10 training labels look at the first validation cycles. This affects 10 of 7,992 rows.
- HDF failures occur almost only in cycles 4,001–5,000, so validation and holdout contain no HDF events.
- The data is synthetic (AI4I), and RNF random failures are unpredictable by design.
- On validation ROC-AUC the rule engine is slightly higher (0.818 vs 0.799). The stack wins on AP, the primary metric for imbalanced data, and on both holdout metrics.
