from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import PWF_HIGH_W, PWF_LOW_W

TEAL = "#2EC4B6"
AMBER = "#F0A202"
RED = "#EF5B4C"
BLUE = "#4CC9F0"
GREY = "#8EA0B3"
SLATE = "#3E6F8E"
PURPLE = "#B388EB"
INK = "#E6EDF3"
GRID = "#1C2834"
EDGE = "#243140"
MODE_COLORS = {
    "None": "#2B4A5E",
    "TWF": AMBER,
    "HDF": RED,
    "PWF": BLUE,
    "OSF": PURPLE,
    "RNF": GREY,
    "Unlabelled": "#5C6B7A",
}
STACK = "Stacked Ensemble"
LEGACY = "Legacy Rule Engine"
MODEL_COLORS = {STACK: TEAL, LEGACY: GREY}
SHORT_NAMES = {
    "Air temperature [K]": "Air Temp",
    "Process temperature [K]": "Process Temp",
    "Rotational speed [rpm]": "Speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool Wear",
    "Mechanical_Power_kW": "Power",
    "Thermal_Stress_K": "Thermal Stress",
    "Tool_Wear_Strain_Index": "Wear Strain",
}


def short(name: str) -> str:
    return SHORT_NAMES.get(name, name)


def layout(figure: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Bahnschrift, sans-serif", "color": INK, "size": 13},
        margin={"l": 60, "r": 30, "t": 64 if title else 36, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": "#121A22", "bordercolor": EDGE, "font": {"color": INK}},
    )
    if title:
        figure.update_layout(title={"text": title, "x": 0, "y": 0.98, "yanchor": "top", "font": {"size": 15}})
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=EDGE)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=EDGE)
    return figure


def gauge(health: float, color: str, standby: bool, warning_health: float, critical_health: float) -> go.Figure:
    bar = "#5C6B7A" if standby else color
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=0 if standby else health,
            number={
                "suffix": "" if standby else "%",
                "font": {"color": bar, "size": 42, "family": "Consolas, Segoe UI, monospace"},
                "valueformat": ".0f" if standby else ".1f",
            },
            title={"text": "Asset Health Index", "font": {"size": 16, "color": GREY}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": GREY, "tickfont": {"color": GREY}},
                "bar": {"color": bar, "thickness": 0.28},
                "bgcolor": "#0B1117",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, critical_health], "color": "#3A1818"},
                    {"range": [critical_health, warning_health], "color": "#3A2E14"},
                    {"range": [warning_health, 100], "color": "#143028"},
                ],
                "threshold": {"line": {"color": bar, "width": 3}, "thickness": 0.8, "value": 0 if standby else health},
            },
        )
    )
    figure.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 24, "r": 24, "t": 48, "b": 12},
        font={"color": INK, "family": "Segoe UI, Bahnschrift, sans-serif"},
    )
    return figure


def health_trend(
    cycles: list,
    raw_health: list,
    health: list,
    failures: list,
    warning_health: float,
    critical_health: float,
    standby: bool,
) -> go.Figure:
    figure = go.Figure()
    figure.add_hrect(y0=0, y1=critical_health, fillcolor=RED, opacity=0.08, line_width=0)
    figure.add_hrect(y0=critical_health, y1=warning_health, fillcolor=AMBER, opacity=0.08, line_width=0)
    figure.add_trace(
        go.Scatter(
            x=cycles,
            y=raw_health,
            name="Raw",
            mode="lines+markers",
            line={"color": GREY, "width": 1.2, "dash": "dot"},
            marker={"size": 4},
            hovertemplate="Cycle %{x}<br>Raw %{y:.1f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=cycles,
            y=health,
            name="Validated",
            mode="lines",
            line={"color": TEAL, "width": 2.6},
            hovertemplate="Cycle %{x}<br>Validated %{y:.1f}%<extra></extra>",
        )
    )
    if failures:
        figure.add_trace(
            go.Scatter(
                x=[cycle for cycle, _ in failures],
                y=[value for _, value in failures],
                name="Failure Event",
                mode="markers",
                marker={"symbol": "x", "size": 13, "color": RED, "line": {"width": 2}},
                hovertemplate="Cycle %{x}<br>Failure event<extra></extra>",
            )
        )
    layout(figure, 400, "Asset Health Index — Last 40 Cycles")
    figure.update_yaxes(title_text="Health (%)", range=[0, 101])
    figure.update_xaxes(title_text="Cycle ID")
    if standby:
        figure.add_annotation(text="Awaiting sensor stream", showarrow=False, font={"color": GREY, "size": 16})
    return figure


def torque_power_trend(cycles: list, torque: list, power: list, standby: bool) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=cycles,
            y=torque,
            name="Torque",
            mode="lines",
            line={"color": BLUE, "width": 2.2},
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
            line={"color": AMBER, "width": 2.2},
            hovertemplate="Cycle %{x}<br>Power %{y:.2f} kW<extra></extra>",
        ),
        secondary_y=True,
    )
    layout(figure, 400, "Torque and Mechanical Power — Last 40 Cycles")
    figure.update_yaxes(title_text="Torque (Nm)", secondary_y=False, color=BLUE)
    figure.update_yaxes(title_text="Power (kW)", secondary_y=True, color=AMBER, showgrid=False)
    figure.update_xaxes(title_text="Cycle ID")
    if standby:
        figure.add_annotation(text="Awaiting sensor stream", showarrow=False, font={"color": GREY, "size": 16})
    return figure


def class_donut(balance: pd.Series) -> go.Figure:
    figure = go.Figure(
        go.Pie(
            labels=balance.index,
            values=balance.values,
            hole=0.62,
            marker={"colors": [SLATE if label == "No Failure" else RED for label in balance.index]},
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value} cycles<extra></extra>",
        )
    )
    layout(figure, 420, "Class Balance")
    figure.update_layout(showlegend=False)
    return figure


def separation_bar(separation: pd.DataFrame) -> go.Figure:
    ordered = separation.assign(size=separation["Cohen d"].abs()).sort_values("size")
    figure = go.Figure(
        go.Bar(
            x=ordered["Cohen d"],
            y=[short(name) for name in ordered["Feature"]],
            orientation="h",
            marker={"color": [RED if value > 0 else BLUE for value in ordered["Cohen d"]]},
            text=[f"{value:+.2f}" for value in ordered["Cohen d"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Cohen d %{x:.2f}<extra></extra>",
        )
    )
    reach = float(ordered["Cohen d"].abs().max()) * 1.4
    layout(figure, 420, "Effect Size (Cohen d)")
    figure.update_layout(margin={"l": 110, "r": 40})
    figure.update_xaxes(range=[-reach, reach])
    return figure


def power_map(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for mode in ["None", "TWF", "HDF", "PWF", "OSF", "RNF", "Unlabelled"]:
        part = frame[frame["Failure_Mode"] == mode]
        if part.empty:
            continue
        figure.add_trace(
            go.Scattergl(
                x=part["Rotational speed [rpm]"],
                y=part["Torque [Nm]"],
                mode="markers",
                name="No Failure" if mode == "None" else mode,
                marker={
                    "color": MODE_COLORS[mode],
                    "size": 4 if mode == "None" else 8,
                    "opacity": 0.35 if mode == "None" else 0.95,
                },
                hovertemplate=f"{mode}<br>Speed %{{x:.0f}} rpm<br>Torque %{{y:.1f}} Nm<extra></extra>",
            )
        )
    speed = np.linspace(frame["Rotational speed [rpm]"].min(), frame["Rotational speed [rpm]"].max(), 200)
    for watts, label in ((PWF_LOW_W, "3.5 kW"), (PWF_HIGH_W, "9 kW")):
        figure.add_trace(
            go.Scatter(x=speed, y=watts * 9.55 / speed, mode="lines", name=label, line={"color": INK, "dash": "dash", "width": 1.4})
        )
    layout(figure, 500, "Torque vs Speed by Failure Mode")
    figure.update_xaxes(title_text="Rotational speed [rpm]")
    figure.update_yaxes(title_text="Torque [Nm]", range=[0, frame["Torque [Nm]"].max() * 1.05])
    return figure


def segment_stack(segments: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for mode in [column for column in ["TWF", "HDF", "PWF", "OSF", "RNF", "Unlabelled"] if column in segments.columns]:
        figure.add_trace(go.Bar(x=segments.index, y=segments[mode], name=mode, marker={"color": MODE_COLORS[mode]}))
    layout(figure, 420, "Failures per 1,000 Cycles")
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title_text="Cycle Range")
    figure.update_yaxes(title_text="Failures")
    return figure


def split_chart(profile: pd.DataFrame) -> go.Figure:
    colors = {"Train": SLATE, "Validation": AMBER, "Holdout": TEAL}
    figure = go.Figure()
    for _, row in profile.iterrows():
        figure.add_trace(
            go.Bar(
                y=[f"{row['Split']}<br>{row['Rows']:,} cycles<br>{row['Failure Events']} failures"],
                x=[row["Rows"]],
                base=[row["First Cycle"]],
                orientation="h",
                name=row["Split"],
                marker={"color": colors[row["Split"]]},
                hovertemplate=f"{row['Split']}<br>Cycles {row['First Cycle']}–{row['Last Cycle']}<extra></extra>",
            )
        )
    layout(figure, 420, "Chronological Split")
    figure.update_layout(barmode="overlay", showlegend=False, margin={"l": 110})
    figure.update_xaxes(title_text="Cycle ID")
    figure.update_yaxes(autorange="reversed")
    return figure


def baseline_bars(comparison: dict) -> go.Figure:
    metrics = [
        ("validation", "average_precision", "Validation AP"),
        ("holdout", "average_precision", "Holdout AP"),
        ("validation", "roc_auc", "Validation ROC-AUC"),
        ("holdout", "roc_auc", "Holdout ROC-AUC"),
    ]
    figure = go.Figure()
    for name in (LEGACY, STACK):
        values = [comparison[name][split][key] for split, key, _ in metrics]
        figure.add_trace(
            go.Bar(
                x=[label for _, _, label in metrics],
                y=values,
                name=name,
                marker={"color": MODEL_COLORS[name]},
                text=[f"{value:.3f}" for value in values],
                textposition="outside",
                cliponaxis=False,
            )
        )
    layout(figure, 440, "Baseline vs Stacked Ensemble")
    figure.update_layout(barmode="group")
    figure.update_yaxes(range=[0, 1])
    return figure


def pr_curves(curves: dict, comparison: dict) -> go.Figure:
    figure = go.Figure()
    for name in (LEGACY, STACK):
        figure.add_trace(
            go.Scatter(
                x=curves[name]["recall"],
                y=curves[name]["precision"],
                name=f"{name} (AP {comparison[name]['holdout']['average_precision']:.3f})",
                mode="lines",
                line={"color": MODEL_COLORS[name], "width": 3 if name == STACK else 2, "dash": "solid" if name == STACK else "dash"},
            )
        )
    base_rate = comparison[STACK]["holdout"]["base_rate"]
    figure.add_hline(y=base_rate, line={"color": EDGE, "dash": "dot"})
    layout(figure, 440, "Precision–Recall (Holdout)")
    figure.update_xaxes(title_text="Recall", range=[0, 1.01])
    figure.update_yaxes(title_text="Precision", range=[0, 1.01])
    return figure


def confusion_pair(confusion: dict) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, subplot_titles=[LEGACY, STACK], horizontal_spacing=0.14)
    for col, name in enumerate((LEGACY, STACK), start=1):
        entry = confusion[name]
        values = np.array([[entry["tn"], entry["fp"]], [entry["fn"], entry["tp"]]])
        figure.add_trace(
            go.Heatmap(
                z=values,
                x=["Normal", "Failure"],
                y=["Normal", "Failure"],
                colorscale=[[0, "#101820"], [1, MODEL_COLORS[name]]],
                text=values,
                texttemplate="%{text}",
                textfont={"size": 18},
                showscale=False,
                hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} cycles<extra></extra>",
            ),
            row=1,
            col=col,
        )
    layout(figure, 380, "Confusion Matrix at Warning Threshold (Holdout)")
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(title_text="Predicted")
    figure.update_yaxes(title_text="Actual", row=1, col=1)
    figure.update_yaxes(showticklabels=False, row=1, col=2)
    return figure


def horizon_chart(rows: list[dict], shipped: int) -> go.Figure:
    frame = pd.DataFrame(rows)
    figure = go.Figure(
        go.Bar(
            x=[f"{horizon}<br>lead {lead:.1f}" for horizon, lead in zip(frame["horizon"], frame["lead"])],
            y=frame["warned_rate"],
            marker={"color": [TEAL if value == shipped else SLATE for value in frame["horizon"]]},
            text=[f"{value:.0%}" for value in frame["warned_rate"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    layout(figure, 440, "Failures Warned by Horizon (Holdout)")
    figure.update_xaxes(title_text="Prediction Horizon (cycles)", tickangle=0)
    figure.update_yaxes(tickformat=".0%", range=[0, 1.1])
    return figure


def alert_chart(alerts: dict) -> go.Figure:
    metrics = [
        ("false_episodes", "False<br>Alarms"),
        ("state_changes", "Status<br>Changes"),
        ("warned", "Failures<br>Warned"),
    ]
    figure = go.Figure()
    peak = 0
    for kind, color, label in (("raw", GREY, "Raw Model Output"), ("validated", TEAL, "With Validation Layer")):
        values = [alerts["holdout"][kind][key] for key, _ in metrics]
        peak = max(peak, *values)
        figure.add_trace(
            go.Bar(
                x=[name for _, name in metrics],
                y=values,
                name=label,
                marker={"color": color},
                text=values,
                textposition="outside",
                cliponaxis=False,
            )
        )
    layout(figure, 440, "Raw vs Validated Alerts (Holdout)")
    figure.update_layout(barmode="group")
    figure.update_xaxes(tickangle=0)
    figure.update_yaxes(range=[0, peak * 1.2])
    return figure
