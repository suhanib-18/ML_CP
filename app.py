from __future__ import annotations

import json
import time

import joblib
import pandas as pd
import streamlit as st

import charts
from src import eda
from src.data import MODEL_PATH, RAW_PATH, REPORT_PATH, STREAM_PATH, load_raw
from src.monitoring import PSI_ALERT, PSI_CLEAR, AlertPolicy, AlertValidator, population_stability_index

NAVIGATION = ["📊 Exploratory Data Analysis", "🧪 Model Evaluation", "📡 Active Asset Telemetry"]
DRIFT_MESSAGE = "Data Drift Detected: Sensor Recalibration Recommended."
INTERVALS = [0.25, 0.5, 1.0, 2.0]
LEVEL_COLORS = ["#2FCE8B", "#E6A23C", "#EF5B4C"]
VERIFY_COLOR = "#4CC9F0"
STANDBY_COLOR = "#8EA0B3"
TRAIL = 40

THEME = """
<style>
    header[data-testid="stHeader"] { background: rgba(11, 17, 23, 0.92); }
    #MainMenu, footer { visibility: hidden; }
    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, rgba(46, 196, 182, 0.08), transparent 55%),
            linear-gradient(180deg, #0B1117 0%, #0A1016 100%);
        color: #E6EDF3;
    }
    section[data-testid="stSidebar"] { background: #0E1620; border-right: 1px solid #243140; }
    .block-container { padding-top: 1.1rem; padding-bottom: 1.4rem; max-width: 1480px; }
    .brand-block { padding: 0.2rem 0 0.8rem 0; border-bottom: 1px solid #243140; margin-bottom: 0.8rem; }
    .brand-kicker { color: #2EC4B6; letter-spacing: 0.22em; font-size: 0.72rem; font-weight: 700; }
    .brand-title { color: #E6EDF3; font-size: 1.15rem; font-weight: 650; margin-top: 0.15rem; }
    .nav-label, .side-label {
        color: #8EA0B3; letter-spacing: 0.16em; font-size: 0.68rem; font-weight: 700; margin: 0.9rem 0 0.35rem 0;
    }
    .side-card { border: 1px solid #243140; background: #101820; padding: 0.75rem 0.8rem; margin-top: 0.45rem; }
    .side-card b { display: block; color: #E6EDF3; font-size: 0.92rem; }
    .side-card span { color: #8EA0B3; font-size: 0.78rem; }
    .masthead {
        display: flex; justify-content: space-between; align-items: flex-end;
        border-bottom: 1px solid #243140; padding-bottom: 0.75rem; margin-bottom: 0.9rem;
    }
    .mast-kicker { color: #2EC4B6; letter-spacing: 0.2em; font-size: 0.72rem; font-weight: 700; }
    .mast-title { font-size: 1.55rem; font-weight: 650; line-height: 1.15; color: #F4F7FA; }
    .mast-meta { text-align: right; color: #8EA0B3; font-size: 0.82rem; letter-spacing: 0.04em; }
    .kpi-card { background: #101820; border: 1px solid #243140; padding: 0.8rem 1rem 0.7rem 1rem; min-height: 96px; margin-bottom: 0.6rem; }
    .kpi-label { color: #8EA0B3; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .kpi-value { color: #F4F7FA; font-family: Consolas, "Segoe UI", monospace; font-size: clamp(1.1rem, 1.9vw, 1.75rem); font-weight: 650; line-height: 1.2; margin-top: 0.25rem; white-space: nowrap; }
    .status-banner, .drift-banner {
        display: flex; align-items: center; gap: 0.7rem;
        background: #101820; border: 1px solid #243140; border-left: 4px solid #5C6B7A;
        padding: 0.75rem 0.9rem; margin: 0.35rem 0 0.8rem 0; font-weight: 650; letter-spacing: 0.02em;
    }
    .status-lamp { width: 10px; height: 10px; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px currentColor; }
    .section-kicker { color: #8EA0B3; letter-spacing: 0.16em; font-size: 0.72rem; font-weight: 700; margin: 0.6rem 0 0.3rem 0; text-transform: uppercase; }
    .side-value { font-family: Consolas, "Segoe UI", monospace; font-size: clamp(0.95rem, 1.4vw, 1.35rem); font-weight: 650; margin-bottom: 0.9rem; white-space: nowrap; }
    div[data-testid="stPlotlyChart"] { background: #101820; border: 1px solid #243140; padding: 0.15rem 0.15rem 0 0.15rem; }
    .stButton > button {
        background: #123F3B; color: #E8FFF9; border: 1px solid #2EC4B6; border-radius: 2px;
        font-weight: 700; letter-spacing: 0.04em; height: 2.7rem; width: 100%;
    }
    .stButton > button:hover { background: #176159; border-color: #7EE0D6; color: white; }
    div[data-testid="stSidebar"] .stRadio label { color: #E6EDF3; }
</style>
"""


def stamp(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource
def load_core(version: float):
    if not MODEL_PATH.exists():
        return None
    core = joblib.load(MODEL_PATH)
    return core if hasattr(core, "alert_policy_") else None


@st.cache_data
def load_report(version: float):
    if not REPORT_PATH.exists():
        return None
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def eda_bundle(version: float) -> dict:
    frame = eda.load_frame()
    return {
        "frame": frame,
        "overview": eda.overview(load_raw()),
        "balance": eda.class_balance(frame),
        "separation": eda.class_separation(frame),
        "segments": eda.segment_modes(frame),
        "splits": eda.split_profile(),
    }


def chart(figure, key: str) -> None:
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False}, key=key)


def kpi_card(label: str, value) -> str:
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'


def tiles(items: list[tuple[str, object]], per_row: int = 4) -> None:
    for start in range(0, len(items), per_row):
        row = items[start:start + per_row]
        for column, (label, value) in zip(st.columns(per_row), row):
            column.markdown(kpi_card(label, value), unsafe_allow_html=True)


def banner(text: str, color: str, drifted: bool = False) -> str:
    kind = "drift-banner" if drifted else "status-banner"
    return (
        f'<div class="{kind}" style="border-left-color:{color}; color:{color};">'
        f'<span class="status-lamp" style="background:{color}; color:{color};"></span>'
        f"<span>{text}</span></div>"
    )


def masthead(title: str) -> None:
    st.markdown(
        f"""
        <div class="masthead">
            <div>
                <div class="mast-kicker">ORION AIS</div>
                <div class="mast-title">{title}</div>
            </div>
            <div class="mast-meta">CELL MC-01<br>MACHINING LINE · ROTATING ASSEMBLY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_eda() -> None:
    masthead("Exploratory Data Analysis")
    data = eda_bundle(stamp(RAW_PATH))
    summary = data["overview"]
    tiles(
        [
            ("Cycles", f"{summary['rows']:,}"),
            ("Failures", summary["failures"]),
            ("Imbalance", f"1 : {summary['imbalance_ratio']:.0f}"),
            ("Missing", summary["missing_values"]),
            ("Duplicates", summary["duplicate_rows"]),
        ],
        per_row=5,
    )
    left, right = st.columns([1, 1.4])
    with left:
        chart(charts.class_donut(data["balance"]), "eda-balance")
    with right:
        chart(charts.separation_bar(data["separation"]), "eda-separation")
    chart(charts.power_map(data["frame"]), "eda-power")
    left, right = st.columns(2)
    with left:
        chart(charts.segment_stack(data["segments"]), "eda-segments")
    with right:
        chart(charts.split_chart(data["splits"]), "eda-splits")


def page_models(report) -> None:
    masthead("Model Evaluation")
    if report is None:
        st.markdown(banner("Evaluation report not found. Run python -m src.train_pipeline.", "#EF5B4C"), unsafe_allow_html=True)
        return
    thresholds = report["thresholds"]
    validated = report["alerts"]["holdout"]["validated"]
    tiles(
        [
            ("Prediction Horizon", f"{report['horizon']} cycles"),
            ("Warning Threshold", f"{thresholds['warning']:.3f}"),
            ("Critical Threshold", f"{thresholds['critical']:.3f}"),
            ("Failures Warned", f"{validated['warned']} / {validated['failure_events']}"),
        ]
    )
    left, right = st.columns(2)
    with left:
        chart(charts.baseline_bars(report["comparison"]), "model-baseline")
    with right:
        chart(charts.pr_curves(report["curves"], report["comparison"]), "model-pr")
    chart(charts.confusion_pair(report["confusion"]), "model-confusion")
    left, right = st.columns(2)
    with left:
        chart(charts.horizon_chart(report["horizons"], report["horizon"]), "model-horizon")
    with right:
        chart(charts.alert_chart(report["alerts"]), "model-alerts")


def status_text(state: dict, policy: AlertPolicy, horizon: int) -> tuple[str, str]:
    level = state["level"]
    if level == 2:
        text = "CRITICAL: FAILURE RISK CONFIRMED"
    elif level == 1:
        text = f"WARNING: DEGRADATION CONFIRMED · INSPECT WITHIN {horizon} CYCLES"
    else:
        text = "STATUS: SYSTEM OPTIMAL"
    if state["rising"]:
        text += f" · VERIFYING FLAG {state['rising']}/{policy.confirm}"
    if state["falling"]:
        text += f" · RECOVERY CHECK {state['falling']}/{policy.release}"
    return text, VERIFY_COLOR if state["rising"] and level == 0 else LEVEL_COLORS[level]


def validation_label(state: dict, policy: AlertPolicy) -> tuple[str, str]:
    if state["rising"]:
        return f"VERIFYING {state['rising']}/{policy.confirm}", VERIFY_COLOR
    if state["falling"]:
        return f"RELEASING {state['falling']}/{policy.release}", LEVEL_COLORS[1]
    if state["level"]:
        return "CONFIRMED", LEVEL_COLORS[state["level"]]
    return "CLEAR", LEVEL_COLORS[0]


def side_value(label: str, value: str, color: str) -> str:
    return f'<div class="section-kicker">{label}</div><div class="side-value" style="color:{color};">{value}</div>'


def render_frame(stage, view: dict) -> None:
    standby = view["standby"]
    key = view["key"]
    with stage.container():
        tiles(
            [("Cycle ID", view["cycle"]), ("Mechanical Power (kW)", view["power"]), ("Thermal Stress (K)", view["thermal"])],
            per_row=3,
        )
        st.markdown(banner(view["status"], STANDBY_COLOR if standby else view["color"]), unsafe_allow_html=True)
        if view["drifted"]:
            st.markdown(banner(DRIFT_MESSAGE, LEVEL_COLORS[1], drifted=True), unsafe_allow_html=True)

        context, gauge_column, alignment = st.columns([1, 2.1, 1])
        with context:
            st.markdown(
                side_value("Model Risk", view["risk"], STANDBY_COLOR)
                + side_value("Validation", view["validation"], view["validation_color"]),
                unsafe_allow_html=True,
            )
        with gauge_column:
            chart(
                charts.gauge(view["health"], view["color"], standby, view["warning_health"], view["critical_health"]),
                f"gauge-{key}",
            )
        with alignment:
            alignment_text, alignment_color = ("RECALIBRATION", LEVEL_COLORS[1]) if view["drifted"] else ("NOMINAL", LEVEL_COLORS[0])
            st.markdown(
                side_value("Sensor Alignment", alignment_text, alignment_color) + side_value("PSI", view["psi"], STANDBY_COLOR),
                unsafe_allow_html=True,
            )

        tiles(
            [
                ("Raw Flags", view["raw_flags"]),
                ("Validated Alerts", view["validated_alerts"]),
                ("Failure Events", view["failures"]),
                ("Warned In Advance", view["warned"]),
            ]
        )
        left, right = st.columns(2)
        with left:
            chart(
                charts.health_trend(
                    view["cycles"],
                    view["raw_health"],
                    view["health_series"],
                    view["failure_marks"],
                    view["warning_health"],
                    view["critical_health"],
                    standby,
                ),
                f"health-{key}",
            )
        with right:
            chart(charts.torque_power_trend(view["cycles"], view["torque"], view["power_series"], standby), f"trend-{key}")


def standby_view(warning_health: float, critical_health: float) -> dict:
    return {
        "key": "standby",
        "standby": True,
        "cycle": "—",
        "power": "—",
        "thermal": "—",
        "status": "STATUS: AWAITING TELEMETRY",
        "color": STANDBY_COLOR,
        "drifted": False,
        "risk": "—",
        "validation": "—",
        "validation_color": STANDBY_COLOR,
        "health": 0.0,
        "psi": "—",
        "raw_flags": 0,
        "validated_alerts": 0,
        "failures": 0,
        "warned": 0,
        "cycles": [],
        "raw_health": [],
        "health_series": [],
        "failure_marks": [],
        "torque": [],
        "power_series": [],
        "warning_health": warning_health,
        "critical_health": critical_health,
    }


def run_stream(core, stage, policy: AlertPolicy, interval: float, warning_health: float, critical_health: float) -> None:
    telemetry = pd.read_csv(STREAM_PATH)
    columns = list(core.feature_columns_)
    horizon = int(core.horizon_)
    validator = AlertValidator(policy)
    history = {key: [] for key in ("cycle", "torque", "power", "raw_health", "health", "level", "failure")}
    counts = {"raw_flags": 0, "validated_alerts": 0, "failures": 0, "warned": 0}
    previous_raw = previous_level = 0
    drift_latched = False

    for _, row in telemetry.iterrows():
        risk = float(core.predict_proba(row[columns].to_numpy(dtype=float).reshape(1, -1))[0, 1])
        state = validator.step(risk)
        failure = int(row.get("Failure_Event", 0))
        if failure:
            counts["failures"] += 1
            counts["warned"] += int(any(level >= 1 for level in history["level"][-horizon:]))
        counts["raw_flags"] += int(state["raw_level"] >= 1 and previous_raw == 0)
        counts["validated_alerts"] += int(state["level"] >= 1 and previous_level == 0)
        previous_raw, previous_level = state["raw_level"], state["level"]

        cycle = int(row["Cycle_ID"])
        health = 100.0 * (1.0 - state["smoothed"])
        history["cycle"].append(cycle)
        history["torque"].append(float(row["Torque [Nm]"]))
        history["power"].append(float(row["Mechanical_Power_kW"]))
        history["raw_health"].append(100.0 * (1.0 - risk))
        history["health"].append(health)
        history["level"].append(state["level"])
        history["failure"].append(failure)

        psi = population_stability_index(history["power"], core.power_edges_, core.power_expected_)
        drifted = psi > PSI_ALERT
        status, color = status_text(state, policy, horizon)
        validation, validation_color = validation_label(state, policy)
        recent = slice(-TRAIL, None)
        render_frame(
            stage,
            {
                "key": str(cycle),
                "standby": False,
                "cycle": f"{cycle:,}",
                "power": f"{float(row['Mechanical_Power_kW']):.2f}",
                "thermal": f"{float(row['Thermal_Stress_K']):.2f}",
                "status": status,
                "color": color,
                "drifted": drifted,
                "risk": f"{risk:.1%}",
                "validation": validation,
                "validation_color": validation_color,
                "health": health,
                "psi": f"{psi:.3f}",
                **counts,
                "cycles": history["cycle"][recent],
                "raw_health": history["raw_health"][recent],
                "health_series": history["health"][recent],
                "failure_marks": [
                    (mark_cycle, mark_health)
                    for mark_cycle, mark_health, mark in zip(history["cycle"][recent], history["health"][recent], history["failure"][recent])
                    if mark
                ],
                "torque": history["torque"][recent],
                "power_series": history["power"][recent],
                "warning_health": warning_health,
                "critical_health": critical_health,
            },
        )
        if drifted and not drift_latched:
            st.toast(DRIFT_MESSAGE)
            drift_latched = True
        elif psi <= PSI_CLEAR:
            drift_latched = False
        time.sleep(interval)


def page_telemetry(core) -> None:
    masthead("Asset Integrity Console")
    if core is None:
        st.markdown(banner("Diagnostic core not commissioned. Run python -m src.train_pipeline.", "#EF5B4C"), unsafe_allow_html=True)
        return
    policy = AlertPolicy(**core.alert_policy_)
    warning_health = 100.0 * (1.0 - policy.warning)
    critical_health = 100.0 * (1.0 - policy.critical)
    command, pace, _spacer = st.columns([1.15, 1.0, 1.6], vertical_alignment="bottom")
    with command:
        clicked = st.button("Initiate Sensor Stream", type="primary")
    with pace:
        interval = st.select_slider("Cycle Interval (s)", options=INTERVALS, value=1.0)
    stage = st.empty()
    if not STREAM_PATH.exists():
        stage.markdown(banner("Telemetry archive is not commissioned for this cell.", "#EF5B4C"), unsafe_allow_html=True)
        return
    if clicked:
        run_stream(core, stage, policy, interval, warning_health, critical_health)
        return
    render_frame(stage, standby_view(warning_health, critical_health))


def main() -> None:
    st.set_page_config(page_title="ORION Asset Integrity", layout="wide", initial_sidebar_state="expanded")
    st.markdown(THEME, unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="brand-block">
            <div class="brand-kicker">ORION AIS</div>
            <div class="brand-title">Asset Integrity</div>
        </div>
        <div class="nav-label">NAVIGATION</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("Navigation", NAVIGATION, label_visibility="collapsed")
    core = load_core(stamp(MODEL_PATH))
    report = load_report(stamp(REPORT_PATH))
    core_status = (
        f"<b>ONLINE</b><span>Stacked Ensemble · {core.horizon_}-cycle horizon</span>"
        if core is not None
        else "<b>OFFLINE</b><span>Not commissioned</span>"
    )
    st.sidebar.markdown(
        f"""
        <div class="side-label">CELL</div>
        <div class="side-card"><b>MC-01 Turning Module</b><span>Rotating assembly</span></div>
        <div class="side-label">DIAGNOSTIC CORE</div>
        <div class="side-card">{core_status}</div>
        """,
        unsafe_allow_html=True,
    )

    if page == NAVIGATION[0]:
        page_eda()
    elif page == NAVIGATION[1]:
        page_models(report)
    else:
        page_telemetry(core)


if __name__ == "__main__":
    main()
