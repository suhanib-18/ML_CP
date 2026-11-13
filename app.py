"""ORION Asset Integrity Console for machining cell MC-01."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
CORE_PATH = ROOT / "models" / "horizon_stacking_pipeline.pkl"
STREAM_PATH = ROOT / "data" / "raw" / "live_telemetry_stream.csv"

HIGHLIGHTS = {"Tool_Wear_Strain_Index", "Mechanical_Power_kW"}
DISPLAY_NAMES = {
    "Air temperature [K]": "Air Temperature",
    "Process temperature [K]": "Process Temperature",
    "Rotational speed [rpm]": "Rotational Speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool Wear",
    "Mechanical_Power_kW": "Mechanical Power",
    "Thermal_Stress_K": "Thermal Stress",
    "Tool_Wear_Strain_Index": "Tool Wear Strain Index",
    "Type_L": "Variant L",
    "Type_M": "Variant M",
    "Type_H": "Variant H",
}
DRIFT_MESSAGE = "Data Drift Detected: Sensor Recalibration Recommended."
NAVIGATION = ["📡 Active Asset Telemetry", "📊 Diagnostic Architecture"]

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
    section[data-testid="stSidebar"] {
        background: #0E1620;
        border-right: 1px solid #243140;
    }
    .block-container { padding-top: 1.1rem; padding-bottom: 1.4rem; max-width: 1440px; }
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
    .kpi-card { background: #101820; border: 1px solid #243140; padding: 0.85rem 1rem 0.75rem 1rem; min-height: 108px; }
    .kpi-label { color: #8EA0B3; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .kpi-value { color: #F4F7FA; font-family: Consolas, "Segoe UI", monospace; font-size: 2rem; font-weight: 650; line-height: 1.2; margin-top: 0.25rem; }
    .status-banner, .drift-banner {
        display: flex; align-items: center; gap: 0.7rem;
        background: #101820; border: 1px solid #243140; border-left: 4px solid #5C6B7A;
        padding: 0.75rem 0.9rem; margin: 0.35rem 0 0.8rem 0;
        font-weight: 650; letter-spacing: 0.02em;
    }
    .status-lamp { width: 10px; height: 10px; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px currentColor; }
    .section-kicker { color: #8EA0B3; letter-spacing: 0.16em; font-size: 0.72rem; font-weight: 700; }
    .section-copy { color: #C5D0DB; font-size: 0.95rem; margin: 0.25rem 0 0.55rem 0; }
    div[data-testid="stPlotlyChart"] {
        background: #101820;
        border: 1px solid #243140;
        padding: 0.15rem 0.15rem 0 0.15rem;
    }
    .stButton > button {
        background: #123F3B;
        color: #E8FFF9;
        border: 1px solid #2EC4B6;
        border-radius: 2px;
        font-weight: 700;
        letter-spacing: 0.04em;
        height: 2.7rem;
        width: 100%;
    }
    .stButton > button:hover { background: #176159; border-color: #7EE0D6; color: white; }
    div[data-testid="stSidebar"] .stRadio label { color: #E6EDF3; }
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME, unsafe_allow_html=True)


def population_stability_index(
    window: list[float],
    mean: float,
    std: float,
    z_edges: np.ndarray,
) -> float:
    """Compare the live 40-cycle power window with the commissioned mean and spread."""
    if len(window) < 40 or std <= 1e-9:
        return 0.0
    expected = np.diff(norm.cdf(np.asarray(z_edges, dtype=float)))
    z_scores = (np.asarray(window[-40:], dtype=float) - mean) / std
    counts, _ = np.histogram(z_scores, bins=np.asarray(z_edges, dtype=float))
    actual = counts.astype(float) + 0.5
    actual /= actual.sum()
    reference = np.clip(expected, 1e-6, None)
    reference /= reference.sum()
    return float(np.sum((actual - reference) * np.log(actual / reference)))


def risk_status(percent: float) -> tuple[str, str]:
    if percent < 40:
        return "STATUS: SYSTEM OPTIMAL", "#2FCE8B"
    if percent <= 75:
        return "WARNING: DEGRADATION DETECTED. Inspect within 5 cycles", "#E6A23C"
    return "CRITICAL SHUTDOWN RISK", "#EF5B4C"


@st.cache_resource
def load_core():
    if not CORE_PATH.exists():
        return None
    return joblib.load(CORE_PATH)


def score_cycle(core, row: pd.Series) -> float:
    columns = list(core.feature_columns_)
    payload = pd.DataFrame(
        [row.reindex(columns).to_numpy(dtype=float)],
        columns=columns,
    )
    scored = core.predict_proba(payload)
    return float(scored[0, 1])


def _layout(figure: go.Figure, height: int) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Bahnschrift, sans-serif", "color": "#E6EDF3", "size": 13},
        margin={"l": 58, "r": 58, "t": 36, "b": 48},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={"bgcolor": "#121A22", "bordercolor": "#243140", "font": {"color": "#E6EDF3"}},
    )
    figure.update_xaxes(gridcolor="#1C2834", zerolinecolor="#1C2834", linecolor="#243140")
    figure.update_yaxes(gridcolor="#1C2834", zerolinecolor="#1C2834", linecolor="#243140")
    return figure


def build_gauge(percent: float, color: str, standby: bool) -> go.Figure:
    bar = "#5C6B7A" if standby else color
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=0 if standby else percent,
            number={
                "suffix": "%" if not standby else "",
                "font": {"color": bar, "size": 42, "family": "Consolas, Segoe UI, monospace"},
                "valueformat": ".1f" if not standby else ".0f",
            },
            title={"text": "Asset Health Index", "font": {"size": 16, "color": "#8EA0B3"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#8EA0B3",
                    "tickfont": {"color": "#8EA0B3"},
                },
                "bar": {"color": bar, "thickness": 0.28},
                "bgcolor": "#0B1117",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#143028"},
                    {"range": [40, 75], "color": "#3A2E14"},
                    {"range": [75, 100], "color": "#3A1818"},
                ],
                "threshold": {
                    "line": {"color": bar, "width": 3},
                    "thickness": 0.8,
                    "value": 0 if standby else percent,
                },
            },
        )
    )
    figure.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 24, "r": 24, "t": 48, "b": 12},
        font={"color": "#E6EDF3", "family": "Segoe UI, Bahnschrift, sans-serif"},
    )
    return figure


def build_trend(cycles: list, torque: list, power: list, standby: bool) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=cycles,
            y=torque,
            name="Torque",
            mode="lines",
            line={"color": "#4CC9F0", "width": 2.2},
            hovertemplate="Cycle %{x}<br>Torque %{y:.2f} Nm<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=cycles,
            y=power,
            name="Mechanical Power",
            mode="lines",
            line={"color": "#F0A202", "width": 2.2},
            hovertemplate="Cycle %{x}<br>Mechanical Power %{y:.2f} kW<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Torque (Nm)", secondary_y=False, color="#4CC9F0")
    figure.update_yaxes(title_text="Mechanical Power (kW)", secondary_y=True, color="#F0A202")
    figure.update_xaxes(title_text="Cycle ID")
    _layout(figure, 430)
    figure.update_layout(title={"text": "Torque and Mechanical Power — Last 40 Cycles", "x": 0})
    if standby:
        figure.add_annotation(
            text="Awaiting sensor stream",
            showarrow=False,
            font={"color": "#8EA0B3", "size": 16},
        )
    return figure


def build_validation(core) -> go.Figure:
    def _ordered(recall: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(np.asarray(recall))
        return np.asarray(recall)[order], np.asarray(precision)[order]

    core_recall, core_precision = _ordered(
        core.historical_recall_core_,
        core.historical_precision_core_,
    )
    legacy_recall, legacy_precision = _ordered(
        core.historical_recall_legacy_,
        core.historical_precision_legacy_,
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=legacy_recall,
            y=legacy_precision,
            name="Legacy Rule Engine",
            mode="lines",
            line={"color": "#8EA0B3", "width": 2, "dash": "dash"},
            fill="tozeroy",
            fillcolor="rgba(142, 160, 179, 0.20)",
            hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra>Legacy Rule Engine</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=core_recall,
            y=core_precision,
            name="Autonomous Diagnostic Core",
            mode="lines",
            line={"color": "#2EC4B6", "width": 2.6},
            fill="tozeroy",
            fillcolor="rgba(46, 196, 182, 0.28)",
            hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra>Autonomous Diagnostic Core</extra>",
        )
    )
    _layout(figure, 480)
    figure.update_xaxes(title_text="Recall", range=[-0.02, 1.02])
    figure.update_yaxes(title_text="Precision", range=[-0.02, 1.02])
    return figure


def build_influence(core) -> go.Figure:
    names = list(core.feature_columns_)
    values = np.asarray(
        core.named_steps["core"].named_estimators_["trees"].feature_importances_,
        dtype=float,
    )
    order = np.argsort(values)
    ordered_names = [names[index] for index in order]
    ordered_values = values[order]
    colors = ["#F0A202" if name in HIGHLIGHTS else "#3E6F8E" for name in ordered_names]
    labels = [DISPLAY_NAMES.get(name, name.replace("_", " ")) for name in ordered_names]
    figure = go.Figure(
        go.Bar(
            x=ordered_values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="%{y}<br>Relative influence %{x:.1%}<extra></extra>",
        )
    )
    _layout(figure, 520)
    figure.update_layout(margin={"l": 180, "r": 40, "t": 36, "b": 48}, showlegend=False)
    figure.update_xaxes(title_text="Relative Influence", tickformat=".0%")
    return figure


def masthead() -> None:
    st.markdown(
        """
        <div class="masthead">
            <div>
                <div class="mast-kicker">ORION AIS</div>
                <div class="mast-title">Asset Integrity Console</div>
            </div>
            <div class="mast-meta">CELL MC-01<br>MACHINING LINE · ROTATING ASSEMBLY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> str:
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        "</div>"
    )


def banner(text: str, color: str, drifted: bool = False) -> str:
    kind = "drift-banner" if drifted else "status-banner"
    return (
        f'<div class="{kind}" style="border-left-color:{color}; color:{color};">'
        f'<span class="status-lamp" style="background:{color}; color:{color};"></span>'
        f"<span>{text}</span></div>"
    )


def render_frame(
    stage,
    *,
    cycle: str,
    power: str,
    thermal: str,
    percent: float,
    status: str,
    color: str,
    cycles: list,
    torque: list,
    power_series: list,
    drifted: bool,
    standby: bool,
) -> None:
    with stage.container():
        left, middle, right = st.columns(3)
        left.markdown(kpi_card("Cycle ID", cycle), unsafe_allow_html=True)
        middle.markdown(kpi_card("Mechanical Power (kW)", power), unsafe_allow_html=True)
        right.markdown(kpi_card("Thermal Stress (K)", thermal), unsafe_allow_html=True)
        st.markdown(banner(status, "#8EA0B3" if standby else color), unsafe_allow_html=True)
        if drifted:
            st.markdown(banner(DRIFT_MESSAGE, "#E6A23C", drifted=True), unsafe_allow_html=True)

        context, gauge_col, alignment = st.columns([1, 2.1, 1])
        with context:
            st.markdown(
                '<div class="section-kicker">Risk Probability</div>'
                '<div class="section-copy">Continuous index for the current cycle. '
                "Inspect the assembly if the index leaves the optimal band.</div>",
                unsafe_allow_html=True,
            )
        with gauge_col:
            st.plotly_chart(
                build_gauge(percent, color, standby),
                use_container_width=True,
                config={"displaylogo": False},
                key=f"asset-health-gauge-{cycle}",
            )
        with alignment:
            alignment_text = "RECALIBRATION" if drifted else "NOMINAL"
            alignment_color = "#E6A23C" if drifted else "#2FCE8B"
            st.markdown(
                '<div class="section-kicker">SENSOR ALIGNMENT</div>'
                f'<div class="kpi-value" style="color:{alignment_color}; font-size:1.4rem;">{alignment_text}</div>'
                '<div class="section-copy">Mechanical power compared with the commissioned operating window.</div>',
                unsafe_allow_html=True,
            )
        st.plotly_chart(
            build_trend(cycles, torque, power_series, standby),
            use_container_width=True,
            config={"displaylogo": False},
            key=f"torque-power-trend-{cycle}",
        )


def run_stream(core, stage) -> None:
    telemetry = pd.read_csv(STREAM_PATH)
    cycles: list[int] = []
    torque: list[float] = []
    power_series: list[float] = []
    drift_latched = False

    for _, row in telemetry.iterrows():
        cycles.append(int(row["Cycle_ID"]))
        torque.append(float(row["Torque [Nm]"]))
        power_series.append(float(row["Mechanical_Power_kW"]))
        risk = score_cycle(core, row)
        percent = risk * 100.0
        status, color = risk_status(percent)
        psi = population_stability_index(
            power_series,
            float(core.power_reference_mean_),
            float(core.power_reference_std_),
            np.asarray(core.power_reference_z_edges_, dtype=float),
        )
        drifted = psi > 0.25
        render_frame(
            stage,
            cycle=f"{int(row['Cycle_ID']):,}",
            power=f"{float(row['Mechanical_Power_kW']):.2f}",
            thermal=f"{float(row['Thermal_Stress_K']):.2f}",
            percent=percent,
            status=status,
            color=color,
            cycles=cycles[-40:],
            torque=torque[-40:],
            power_series=power_series[-40:],
            drifted=drifted,
            standby=False,
        )
        if drifted and not drift_latched:
            st.toast(DRIFT_MESSAGE)
            drift_latched = True
        elif psi <= 0.20:
            drift_latched = False
        time.sleep(1.0)


def page_telemetry(core) -> None:
    masthead()
    command, _spacer = st.columns([1.15, 2.6])
    with command:
        clicked = st.button("Initiate Sensor Stream", type="primary")
    stage = st.empty()
    if not STREAM_PATH.exists():
        stage.markdown(banner("Telemetry archive is not commissioned for this cell.", "#EF5B4C"), unsafe_allow_html=True)
        return
    if clicked:
        run_stream(core, stage)
        return
    render_frame(
        stage,
        cycle="—",
        power="—",
        thermal="—",
        percent=0.0,
        status="STATUS: AWAITING TELEMETRY",
        color="#8EA0B3",
        cycles=[],
        torque=[],
        power_series=[],
        drifted=False,
        standby=True,
    )


def page_architecture(core) -> None:
    masthead()
    st.markdown(
        """
        <div class="section-kicker">DIAGNOSTIC ARCHITECTURE</div>
        <div class="section-copy">Historical Validation of the Autonomous Diagnostic Core against the Legacy Rule Engine. The shaded envelope sits above the legacy curve through the primary operating range.</div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_validation(core),
        use_container_width=True,
        config={"displaylogo": False},
        key="historical-validation",
    )
    st.markdown(
        """
        <div class="section-kicker">Process Parameters</div>
        <div class="section-copy">Process Parameter Influence inside the Autonomous Diagnostic Core. Thermal stress and tool wear lead the ranking. Amber bars mark the thermodynamic drivers calculated from torque, speed, and tool wear: mechanical power and tool-wear strain.</div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_influence(core),
        use_container_width=True,
        config={"displaylogo": False},
        key="process-influence",
    )


def main() -> None:
    st.set_page_config(
        page_title="ORION Asset Integrity",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
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
    st.sidebar.markdown(
        """
        <div class="side-label">CELL</div>
        <div class="side-card"><b>MC-01 Turning Module</b><span>Rotating assembly · Plant network connected</span></div>
        <div class="side-label">DIAGNOSTIC CORE</div>
        <div class="side-card"><b>ONLINE</b><span>Autonomous Diagnostic Core commissioned for this cell</span></div>
        """,
        unsafe_allow_html=True,
    )

    core = load_core()
    if core is None:
        st.markdown(banner("The diagnostic core is not commissioned for this cell.", "#EF5B4C"), unsafe_allow_html=True)
        return

    if page == NAVIGATION[0]:
        page_telemetry(core)
    else:
        page_architecture(core)


if __name__ == "__main__":
    main()
