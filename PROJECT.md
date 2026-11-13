# ORION AIS — Asset Integrity Console

**Predictive maintenance system for machining cell MC-01**

ORION AIS (Asset Integrity System) is an end-to-end machine learning application that predicts impending machine failure **up to five cycles ahead**, scores live telemetry in real time, detects sensor drift, and presents operators with an industrial control-room console built in Streamlit.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Solution Workflow](#3-solution-workflow)
4. [System Architecture](#4-system-architecture)
5. [Tech Stack](#5-tech-stack)
6. [Dataset](#6-dataset)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Design](#8-model-design)
9. [Training & Commissioning Pipeline](#9-training--commissioning-pipeline)
10. [Live Inference & Drift Detection](#10-live-inference--drift-detection)
11. [Operator Console (UI)](#11-operator-console-ui)
12. [File Structure](#12-file-structure)
13. [Key Modules Reference](#13-key-modules-reference)
14. [How to Run](#14-how-to-run)
15. [Risk Thresholds & Status Logic](#15-risk-thresholds--status-logic)
16. [Validation & Acceptance Criteria](#16-validation--acceptance-criteria)
17. [Glossary](#17-glossary)

---

## 1. Project Overview

| Attribute | Detail |
|-----------|--------|
| **Product name** | ORION AIS — Asset Integrity Console |
| **Target asset** | Cell MC-01 · Turning Module · Rotating Assembly |
| **Domain** | Predictive maintenance / industrial IoT telemetry |
| **Prediction horizon** | Next 5 machining cycles |
| **Primary deliverable** | Commissioned stacking pipeline + live Streamlit console |
| **Source data** | AI4I 2020 Predictive Maintenance Dataset (10,000 cycles) |

The system has two commissioned modes:

1. **Training / commissioning** — builds physics features, trains a stacked ensemble, validates against a legacy rule engine, and serializes the diagnostic core.
2. **Live console** — replays holdout telemetry cycle-by-cycle, scores failure risk, monitors mechanical-power drift (PSI), and visualizes health KPIs.

---

## 2. Problem Statement

CNC / turning cells fail infrequently (~3.4% of cycles in the source archive), but failures are costly. Fixed threshold “rule engines” (overstrain, heat, power, wear limits) catch some events late and produce noisy alerts.

**Goals:**

- Predict failure **before** it occurs (5-cycle lookahead), not only at the failure cycle.
- Outperform a transparent **legacy rule engine** on precision–recall in historical validation.
- Surface a continuous **Asset Health Index** to operators with clear optimal / warning / critical bands.
- Detect **sensor drift** in mechanical power so recalibration can be recommended.
- Keep training leakage-free: chronological splits, no event-flag features, physics derived only from sensors.

---

## 3. Solution Workflow

```
┌─────────────────┐
│  Raw archive    │  ai4i2020.csv (UDI-ordered plant cycles)
│  10,000 cycles  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load & clean   │  Sort by UDI · drop IDs / stale columns · assign Cycle_ID
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Physics FE     │  Mechanical Power · Thermal Stress · Tool Wear Strain Index
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Horizon label  │  Target_Next_5_Cycles = any failure in cycles t+1 … t+5
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Encode type    │  One-hot Type_L / Type_M / Type_H
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chrono split   │  80% train · 15% validation · 5% holdout (live stream)
└────────┬────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│  Fit stacking   │            │  Export holdout │
│  pipeline       │            │  as live stream │
│  (scaler+core)  │            │  CSV            │
└────────┬────────┘            └────────┬────────┘
         │                              │
         ▼                              │
┌─────────────────┐                     │
│  Validate vs    │                     │
│  legacy rules   │                     │
│  (AP / PR)      │                     │
└────────┬────────┘                     │
         │                              │
         ▼                              │
┌─────────────────┐                     │
│  Attach refs    │  power mean/std, PR curves, feature list
│  Serialize .pkl │                     │
└────────┬────────┘                     │
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌─────────────────┐
              │  Streamlit app  │
              │  ORION console  │
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Active Asset   │         │  Diagnostic     │
│  Telemetry      │         │  Architecture   │
│  (live score +  │         │  (PR curves +   │
│   PSI drift)    │         │   influence)    │
└─────────────────┘         └─────────────────┘
```

### End-to-end steps (narrative)

1. **Ingest** — Load `ai4i2020.csv`, enforce UDI order so time causality is preserved.
2. **Engineer** — Derive thermodynamic / mechanical drivers from torque, speed, temperatures, and tool wear.
3. **Label** — Mark each cycle if a machine failure occurs within the next five cycles; drop the incomplete tail.
4. **Encode** — Expand product quality grade (`Type`) to one-hot columns.
5. **Split chronologically** — Never shuffle; later cycles are held out as “live” stream.
6. **Train** — Standardize features; stack XGBoost + RBF-SVM + MLP with a custom logistic blend and chronological CV.
7. **Gate** — Require the core’s average precision (and precision-at-recall) to beat the legacy rule engine.
8. **Commission** — Persist the pipeline plus reference statistics and historical PR curves to `models/horizon_stacking_pipeline.pkl`.
9. **Operate** — Console loads the core, streams holdout rows once per second, scores risk, and computes Population Stability Index on mechanical power.

---

## 4. System Architecture

### 4.1 Logical layers

| Layer | Responsibility | Components |
|-------|----------------|------------|
| **Data** | Raw cycles + live stream CSV | `data/raw/` |
| **Feature / training** | Physics, labels, stacking fit, acceptance tests | `src/train_pipeline.py`, `src/blend.py` |
| **Model store** | Serialized diagnostic core | `models/horizon_stacking_pipeline.pkl` |
| **Presentation** | Operator UI, gauges, trends, architecture charts | `app.py`, `.streamlit/config.toml` |

### 4.2 Diagnostic core (inference object)

The commissioned artifact is a scikit-learn `Pipeline`:

```
Pipeline
├── scaler          → StandardScaler
└── core            → StackingClassifier
    ├── trees       → XGBClassifier
    ├── geometric   → SVC(RBF, probability=True)
    ├── deep        → MLPClassifier(64, 32)
    └── final_estimator → OpeningBlockLogisticRegression
```

**Attached attributes** (written after fit, read by the console):

| Attribute | Purpose |
|-----------|---------|
| `feature_columns_` | Ordered process-parameter list for scoring |
| `power_reference_mean_` / `power_reference_std_` | Commissioned mechanical-power window |
| `power_reference_z_edges_` | PSI histogram edges in z-score space |
| `power_reference_expected_` | Expected bin probabilities under Normal |
| `historical_precision_core_` / `historical_recall_core_` | PR curve for architecture page |
| `historical_precision_legacy_` / `historical_recall_legacy_` | Legacy rule-engine PR curve |

### 4.3 Console architecture

```
app.py
├── Theme / CSS (industrial dark UI)
├── load_core()          @st.cache_resource → joblib load
├── Navigation
│   ├── Active Asset Telemetry
│   │   ├── Initiate Sensor Stream
│   │   ├── score_cycle() per row
│   │   ├── population_stability_index()
│   │   ├── KPI cards + status banner
│   │   ├── Asset Health Index gauge
│   │   └── Torque / Power trend (last 40)
│   └── Diagnostic Architecture
│       ├── Precision–Recall: Core vs Legacy
│       └── Process Parameter Influence (XGBoost importances)
└── Sidebar: brand, cell MC-01, core ONLINE status
```

### 4.4 Data flow at runtime

```
live_telemetry_stream.csv
        │
        ▼
   for each cycle row  ──►  reindex to feature_columns_
        │                          │
        │                          ▼
        │                   core.predict_proba → P(failure in ≤5 cycles)
        │                          │
        │                          ▼
        │                   Asset Health Index % + status band
        │
        └── Mechanical_Power_kW window (40)
                    │
                    ▼
              PSI vs commissioned Normal
                    │
                    ▼
              Drift banner / toast if PSI > 0.25
```

---

## 5. Tech Stack

### 5.1 Languages & frameworks

| Technology | Role |
|------------|------|
| **Python 3** | Entire codebase |
| **Streamlit** | Operator console / web UI |
| **Plotly** | Gauges, dual-axis trends, PR curves, influence bars |
| **scikit-learn** | Pipeline, stacking, SVM, MLP, scaler, metrics |
| **XGBoost** | Tree base learner in the stack |
| **pandas / NumPy** | Data loading, vectorized physics, arrays |
| **SciPy** | Normal CDF for PSI expected bin masses |
| **joblib** | Model serialization / deserialization |

### 5.2 Dependencies (`requirements.txt`)

```
pandas
numpy
scikit-learn
xgboost
streamlit
plotly
scipy
```

> Note: `joblib` is typically available via scikit-learn; install it explicitly if your environment does not pull it in.

### 5.3 UI / theming

| Item | Value |
|------|-------|
| Theme base | Dark (`\.streamlit\config.toml`) |
| Primary accent | `#2EC4B6` (teal) |
| Background | `#0B1117` |
| Secondary panels | `#101820` |
| Text | `#E6EDF3` |
| Custom CSS | Inline industrial masthead, KPI cards, status lamps in `app.py` |

---

## 6. Dataset

### 6.1 Source

**AI4I 2020 Predictive Maintenance Dataset** — synthetic but physics-informed industrial process data commonly used for failure prediction benchmarks.

| Property | Value |
|----------|-------|
| Location | `data/raw/ai4i2020.csv` (copy also at repo root) |
| Rows | 10,000 cycles |
| Ordering key | `UDI` (Unique Data Index) — treated as time |
| Failure rate | ~3.39% (`Machine failure == 1`) |

### 6.2 Raw columns

| Column | Description |
|--------|-------------|
| `UDI` | Sequential cycle identifier |
| `Product ID` | Product serial (dropped after load) |
| `Type` | Quality grade: L / M / H |
| `Air temperature [K]` | Ambient temperature |
| `Process temperature [K]` | Process / tool zone temperature |
| `Rotational speed [rpm]` | Spindle speed |
| `Torque [Nm]` | Cutting / spindle torque |
| `Tool wear [min]` | Accumulated tool wear time |
| `Machine failure` | Binary failure flag at this cycle |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Failure mode flags (tool wear, heat dissipation, power, overstrain, random) — **never used as model features** |

### 6.3 Chronological split

| Split | Fraction | Role |
|-------|----------|------|
| Train | 80% | Fit stacking pipeline |
| Validation | 15% | PR / AP vs legacy; stored for architecture charts |
| Holdout | 5% | Written to `live_telemetry_stream.csv` for console replay |

Splits are **contiguous in UDI order** (no random shuffle).

---

## 7. Feature Engineering

### 7.1 Process parameters (model inputs)

**Sensors (5)**

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

**Physics drivers (3)**

| Feature | Formula |
|---------|---------|
| `Mechanical_Power_kW` | `Torque [Nm] × Rotational speed [rpm] / 9550` |
| `Thermal_Stress_K` | `Process temperature [K] − Air temperature [K]` |
| `Tool_Wear_Strain_Index` | `Mechanical_Power_kW × Tool wear [min]` |

**Product grade one-hots (3)**

- `Type_L`, `Type_M`, `Type_H`

**Total feature width:** 11 columns (`FEATURE_COLUMNS`).

### 7.2 Target

```
Target_Next_5_Cycles =
  1 if Machine failure occurs on any of cycles t+1 … t+5
  0 otherwise
```

Implementation: `shift(-5).rolling(window=5).max()` on `Machine failure`, with an integrity check every 1,000 cycles. The last five rows (incomplete horizon) are dropped.

### 7.3 Leakage controls

- Failure mode columns (`TWF`, `HDF`, …) are excluded from `FEATURE_COLUMNS`.
- Training asserts no event-flag leakage into features.
- Chronological folds prevent future-to-past contamination inside stacking CV (with a documented opening-block exception — see Model Design).

---

## 8. Model Design

### 8.1 Base learners

| Name | Algorithm | Role |
|------|-----------|------|
| `trees` | `XGBClassifier` | Nonlinear interactions; provides `feature_importances_` for UI |
| `geometric` | `SVC(kernel="rbf", probability=True)` | Margin / geometric decision surface |
| `deep` | `MLPClassifier((64, 32))` | Dense nonlinear capacity |

**XGBoost highlights:** `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `scale_pos_weight = n_neg / n_pos`, regularization via `reg_lambda`, `min_child_weight=5`.

### 8.2 Stacking & chronological CV

`StackingClassifier` uses `stack_method="predict_proba"` and **custom chronological folds** (`chronological_folds`):

- Partition train into contiguous blocks at 25% / 50% / 75%.
- Later blocks are scored only from earlier cycles (past → future).
- The **opening block** has no past-only OOF scores; it is scored from the *following* block so every sample still gets one OOF vector (required by stacking).

### 8.3 Meta-learner: `OpeningBlockLogisticRegression`

Defined in `src/blend.py`. Extends `LogisticRegression` with `class_weight="balanced"`.

- Sets sample weights to **0** for the first **55%** of stacking meta-features (`IGNORE_FRACTION = 0.55`).
- Rationale: early cycles are a different operating regime, and the opening CV block lacks past-only OOF scores — both are excluded from the blend fit.

### 8.4 Legacy rule engine (baseline)

Continuous risk score = max of clipped sub-risks:

| Component | Idea |
|-----------|------|
| Overstrain | `(wear × torque) / grade_threshold` (L: 11k, default: 12k, H: 13k) |
| Heat | Low thermal stress × low speed relative to fixed limits |
| Power | Distance outside [3500 W, 9000 W] band |
| Wear | Tool wear above 150 min toward 240 min |

Used only for **historical validation charts** and acceptance gating — not for live scoring.

---

## 9. Training & Commissioning Pipeline

**Entry point:** `python -m src.train_pipeline` (or `python src/train_pipeline.py` from repo root with path bootstrap).

### Pipeline stages

1. `load_cycles` — CSV → sorted frame, `Cycle_ID`, drop `UDI` / `Product ID` / stale columns.
2. `add_physics` — vectorized power, thermal stress, wear–strain.
3. `add_horizon_target` — 5-cycle lookahead label + integrity checks.
4. `encode_variant` — one-hot product type.
5. `chronological_split` — 80 / 15 / 5.
6. Export holdout features (+ `Cycle_ID`) → `data/raw/live_telemetry_stream.csv`.
7. Compute `scale_pos_weight`; build pipeline; `fit`.
8. Score validation with core and legacy; compute PR curves and average precision.
9. **Acceptance gate:** core AP must exceed legacy AP, and mean precision advantage across recall marks `{0.2…0.8}` must be ≥ 0.03.
10. Attach reference / history attributes; `joblib.dump` → `models/horizon_stacking_pipeline.pkl`.
11. Reload probe (in-process + subprocess) to ensure the console can deserialize the core.
12. Print influence ranking and commissioning timing.

### Outputs

| Artifact | Path |
|----------|------|
| Commissioned core | `models/horizon_stacking_pipeline.pkl` |
| Live telemetry stream | `data/raw/live_telemetry_stream.csv` |

---

## 10. Live Inference & Drift Detection

### 10.1 Cycle scoring

For each streamed row:

1. Build a one-row `DataFrame` aligned to `core.feature_columns_`.
2. `predict_proba` → probability of class 1 (impending failure).
3. `Asset Health Index (%) = probability × 100`.

### 10.2 Population Stability Index (PSI)

Compares the **last 40** mechanical-power values to the commissioned Normal(`mean`, `std`) using z-score bins `[-∞, -1), [-1, 1], (1, ∞]`.

```
PSI = Σ (actual − expected) × ln(actual / expected)
```

| Condition | Behavior |
|-----------|----------|
| Window &lt; 40 cycles or `std ≈ 0` | PSI = 0 |
| PSI &gt; 0.25 | Drift latched → amber banner + toast: *“Data Drift Detected: Sensor Recalibration Recommended.”* |
| PSI ≤ 0.20 | Drift latch cleared |

### 10.3 Stream cadence

`run_stream` iterates holdout rows with `time.sleep(1.0)` — one cycle per second for operator-readable playback.

---

## 11. Operator Console (UI)

### 11.1 Pages

| Navigation | Content |
|------------|---------|
| **Active Asset Telemetry** | Start stream, KPIs (Cycle ID, Mechanical Power, Thermal Stress), status banner, health gauge, sensor alignment, torque/power trend |
| **Diagnostic Architecture** | Historical PR: Autonomous Diagnostic Core vs Legacy Rule Engine; horizontal bar chart of XGBoost feature influence (amber = thermodynamic drivers) |

### 11.2 Status bands (Asset Health Index)

| Index % | Banner |
|---------|--------|
| &lt; 40 | `STATUS: SYSTEM OPTIMAL` (green) |
| 40–75 | `WARNING: DEGRADATION DETECTED. Inspect within 5 cycles` (amber) |
| &gt; 75 | `CRITICAL SHUTDOWN RISK` (red) |

### 11.3 Standby state

Before “Initiate Sensor Stream”, the console shows empty KPIs, zeroed gauge, and *Awaiting sensor stream* on the trend chart.

---

## 12. File Structure

```
CP/
├── app.py                          # ORION Streamlit console (telemetry + architecture)
├── requirements.txt                # Python dependencies
├── PROJECT.md                      # This documentation
├── ai4i2020.csv                    # Dataset copy at repo root (optional convenience)
├── holdout_stream.csv              # Legacy / alternate holdout export (if present)
│
├── .streamlit/
│   └── config.toml                 # Dark theme, primary teal, usage stats off
│
├── data/
│   └── raw/
│       ├── ai4i2020.csv            # Canonical training archive
│       └── live_telemetry_stream.csv  # Holdout cycles for live replay (written by train)
│
├── models/
│   └── horizon_stacking_pipeline.pkl  # Commissioned diagnostic core
│
└── src/
    ├── __init__.py                 # Package marker
    ├── blend.py                    # OpeningBlockLogisticRegression meta-learner
    └── train_pipeline.py           # Full commissioning / training script
```

### File roles (quick map)

| Path | Role |
|------|------|
| `app.py` | Presentation + live scoring + PSI |
| `src/train_pipeline.py` | Offline ML pipeline & model export |
| `src/blend.py` | Reload-safe custom estimator for stacking |
| `models/*.pkl` | Runtime artifact for `load_core()` |
| `data/raw/live_telemetry_stream.csv` | Simulated plant network feed |
| `.streamlit/config.toml` | Streamlit theming |

---

## 13. Key Modules Reference

### `src/train_pipeline.py`

| Symbol | Description |
|--------|-------------|
| `SENSOR_COLUMNS` / `PHYSICS_COLUMNS` / `GRADE_COLUMNS` | Feature taxonomy |
| `load_cycles` | Ordered ingest |
| `add_physics` | First-principles drivers |
| `add_horizon_target` | 5-cycle failure horizon |
| `encode_variant` | Type one-hots |
| `chronological_split` | 80/15/5 |
| `legacy_rule_engine_score` | Baseline continuous score |
| `chronological_folds` | Stacking CV splits |
| `build_pipeline` | Scaler + StackingClassifier |
| `main` | Commission end-to-end |

### `src/blend.py`

| Symbol | Description |
|--------|-------------|
| `IGNORE_FRACTION` | 0.55 — fraction of early OOF rows zero-weighted |
| `OpeningBlockLogisticRegression` | Balanced logistic blend on recent cycles only |

### `app.py`

| Symbol | Description |
|--------|-------------|
| `population_stability_index` | Live PSI on mechanical power |
| `risk_status` | Optimal / warning / critical mapping |
| `load_core` | Cached joblib load |
| `score_cycle` | Single-cycle `predict_proba` |
| `build_gauge` / `build_trend` / `build_validation` / `build_influence` | Plotly figures |
| `run_stream` | 1 Hz telemetry loop |
| `page_telemetry` / `page_architecture` | UI pages |
| `main` | App entry |

---

## 14. How to Run

### Prerequisites

- Python 3.10+ recommended (project has been run on 3.13)
- Place `ai4i2020.csv` under `data/raw/` before first commission

### Install

```bash
pip install -r requirements.txt
```

### Commission the diagnostic core

From the project root (`CP/`):

```bash
python -m src.train_pipeline
```

Expected console output includes train/validation/holdout sizes, precision-at-recall vs legacy, process-parameter influence, and paths to the saved `.pkl` and live stream CSV.

### Launch the operator console

```bash
streamlit run app.py
```

1. Confirm sidebar shows **MC-01** and **Diagnostic Core ONLINE**.
2. Open **Active Asset Telemetry** → **Initiate Sensor Stream**.
3. Open **Diagnostic Architecture** for PR comparison and influence ranking.

---

## 15. Risk Thresholds & Status Logic

| Signal | Threshold / rule | Operator meaning |
|--------|------------------|------------------|
| Health index | &lt; 40% | System optimal |
| Health index | 40–75% | Inspect within ~5 cycles |
| Health index | &gt; 75% | Critical shutdown risk |
| PSI | &gt; 0.25 | Sensor drift / recalibration |
| PSI clear | ≤ 0.20 | Return to nominal alignment |
| PSI window | 40 cycles | Minimum history for PSI |

---

## 16. Validation & Acceptance Criteria

During commissioning, the Autonomous Diagnostic Core must:

1. Produce higher **average precision** than the legacy rule engine on the chronological validation set.
2. Show a mean precision advantage of at least **0.03** across recall marks `0.2, 0.3, …, 0.8`.
3. Serialize with all required attributes and successfully reload in a fresh subprocess (console compatibility check).
4. Keep XGBoost influence width equal to `len(FEATURE_COLUMNS)`.

Failure of any gate raises `RuntimeError` and aborts model save.

---

## 17. Glossary

| Term | Meaning |
|------|---------|
| **ORION AIS** | Product brand for the Asset Integrity System |
| **MC-01** | Machining cell identifier (turning / rotating assembly) |
| **Diagnostic Core** | Commissioned stacking pipeline used for live scoring |
| **Horizon** | Five-cycle lookahead failure window |
| **Legacy Rule Engine** | Fixed physics/threshold baseline for comparison |
| **PSI** | Population Stability Index — distribution drift metric |
| **Asset Health Index** | Displayed failure probability as a 0–100% gauge |
| **Commissioning** | Offline train + validate + serialize workflow |

---

## Appendix A — Process Parameter Display Names (UI)

| Internal column | Console label |
|-----------------|---------------|
| Air temperature [K] | Air Temperature |
| Process temperature [K] | Process Temperature |
| Rotational speed [rpm] | Rotational Speed |
| Torque [Nm] | Torque |
| Tool wear [min] | Tool Wear |
| Mechanical_Power_kW | Mechanical Power |
| Thermal_Stress_K | Thermal Stress |
| Tool_Wear_Strain_Index | Tool Wear Strain Index |
| Type_L / Type_M / Type_H | Variant L / M / H |

---

## Appendix B — Design Principles

1. **Time causality first** — chronological splits and folds; no random shuffle of plant cycles.
2. **Physics before black-box** — engineered power, thermal stress, and wear–strain as first-class features.
3. **Beatable baseline** — model ships only if it clearly beats the legacy rule engine.
4. **Operator-grade UX** — dark industrial theme, sparse KPIs, explicit status language, one-second stream pace.
5. **Reload safety** — custom estimators live in `src.blend` so joblib can import them when the Streamlit process starts.

---

*Document generated for the ORION AIS / CP predictive maintenance project. Reflects the codebase as of the commissioned stacking console layout.*
