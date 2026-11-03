"""Draw a 3x3 sweep six different ways, so the layouts can be compared.

Throwaway spike (see README.md beside this file). Reads the real datasets
under out/ and writes one PNG per layout into plots/. Plotly, because that
is what the project already depends on.

    python3 tools/sweep_plot_spike/make_plots.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
OUT = HERE / "out"
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)

CASES = "disease_cases"
RAIN = "rainfall"
LOCATION = "north"          # one location, so the layouts compare themselves
INK = "#1f4e79"
DRIVER = "#9ecae1"

index = json.loads((OUT / "sweep.json").read_text())
ROWS = sorted({r["params"]["spike_height"] for r in index["runs"]})
COLS = sorted({r["params"]["lag"] for r in index["runs"]})

runs = {}
for entry in index["runs"]:
    df = pd.read_csv(OUT / entry["dataset"])
    df = df[df["location"] == LOCATION].reset_index(drop=True)
    runs[(entry["params"]["spike_height"], entry["params"]["lag"])] = df

# Shared scales: comparing panels is the whole point, so no panel gets its
# own axis range. (Small-multiple guidance is unanimous on this.)
CASE_MAX = max(df[CASES].max() for df in runs.values()) * 1.05
RAIN_MAX = max(df[RAIN].max() for df in runs.values()) * 1.05


def _save(fig, name: str, height: int, width: int = 1150) -> None:
    fig.update_layout(
        height=height, width=width, template="plotly_white",
        margin=dict(l=70, r=40, t=80, b=60), showlegend=False,
    )
    fig.write_image(PLOTS / f"{name}.png", scale=2)
    print(f"  wrote plots/{name}.png")


def _grid(titles=True):
    return make_subplots(
        rows=len(ROWS), cols=len(COLS),
        shared_xaxes=True, shared_yaxes=True,
        column_titles=[f"lag = {c}" for c in COLS] if titles else None,
        row_titles=[f"spike_height = {r}" for r in ROWS] if titles else None,
        horizontal_spacing=0.03, vertical_spacing=0.04,
    )


def layout_a():
    """A — trellis: one panel per run, disease only."""
    fig = _grid()
    for i, sh in enumerate(ROWS, start=1):
        for j, lag in enumerate(COLS, start=1):
            fig.add_trace(
                go.Scatter(y=runs[(sh, lag)][CASES], line=dict(color=INK, width=1.4)),
                row=i, col=j,
            )
    fig.update_yaxes(range=[0, CASE_MAX])
    fig.update_layout(title="A — trellis grid: one panel per run (disease_cases)")
    _save(fig, "a_grid_cases", 760)


def layout_b():
    """B — trellis, with the driver shaded behind each panel."""
    fig = _grid()
    for i, sh in enumerate(ROWS, start=1):
        for j, lag in enumerate(COLS, start=1):
            df = runs[(sh, lag)]
            # Rainfall rescaled onto the disease axis: the shape is the point
            # here, not its units, and a twin axis per panel would be nine
            # extra axes to label.
            fig.add_trace(
                go.Scatter(
                    y=df[RAIN] / RAIN_MAX * CASE_MAX, fill="tozeroy",
                    line=dict(color=DRIVER, width=0), fillcolor="rgba(158,202,225,.55)",
                ),
                row=i, col=j,
            )
            fig.add_trace(
                go.Scatter(y=df[CASES], line=dict(color=INK, width=1.4)),
                row=i, col=j,
            )
    fig.update_yaxes(range=[0, CASE_MAX])
    fig.update_layout(
        title="B — trellis + driver: disease (line) over rainfall (shaded, rescaled)",
    )
    _save(fig, "b_grid_two_series", 760)


def layout_c():
    """C — every run overlaid on one axis."""
    fig = go.Figure()
    shades = ["#d0e3f0", "#7fb3d5", "#1f4e79"]
    dashes = ["solid", "dash", "dot"]
    for i, sh in enumerate(ROWS):
        for j, lag in enumerate(COLS):
            fig.add_trace(go.Scatter(
                y=runs[(sh, lag)][CASES], name=f"sh={sh}, lag={lag}",
                line=dict(color=shades[i], width=1.4, dash=dashes[j]),
            ))
    fig.update_layout(
        title="C — all nine runs overlaid on one axis",
        xaxis_title="period", yaxis_title=CASES, showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
    )
    _save(fig, "c_overlay_all", 560)
    fig.update_layout(showlegend=True)


def layout_d():
    """D — one panel per spike_height, the lags overlaid inside it."""
    fig = make_subplots(
        rows=len(ROWS), cols=1, shared_xaxes=True, shared_yaxes=True,
        row_titles=[f"spike_height = {r}" for r in ROWS],
        vertical_spacing=0.05,
    )
    shades = ["#d0e3f0", "#7fb3d5", "#1f4e79"]
    for i, sh in enumerate(ROWS, start=1):
        for j, lag in enumerate(COLS):
            fig.add_trace(
                go.Scatter(
                    y=runs[(sh, lag)][CASES], name=f"lag = {lag}",
                    line=dict(color=shades[j], width=1.5),
                    showlegend=(i == 1),
                ),
                row=i, col=1,
            )
    fig.update_yaxes(range=[0, CASE_MAX])
    fig.update_layout(
        title="D — one panel per spike_height, lags overlaid inside",
        showlegend=True, legend=dict(orientation="h", y=1.04),
    )
    _save(fig, "d_overlay_per_row", 720)


def layout_e():
    """E — one summary number per run, as a heatmap. No time axis."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["mean disease_cases", "corr(rainfall, disease)"],
        horizontal_spacing=0.16,
    )
    for col, stat in enumerate(["mean", "corr"], start=1):
        z, text = [], []
        for sh in ROWS:
            z_row, t_row = [], []
            for lag in COLS:
                df = runs[(sh, lag)]
                value = (
                    df[CASES].mean() if stat == "mean"
                    else df[RAIN].corr(df[CASES])
                )
                z_row.append(value)
                t_row.append(f"{value:,.2f}" if stat == "corr" else f"{value:,.0f}")
            z.append(z_row)
            text.append(t_row)
        fig.add_trace(
            go.Heatmap(
                z=z, text=text, texttemplate="%{text}",
                x=[f"lag {c}" for c in COLS], y=[f"sh {r}" for r in ROWS],
                colorscale="YlGnBu", showscale=False,
            ),
            row=1, col=col,
        )
    fig.update_layout(title="E — heatmap: one summary number per run")
    _save(fig, "e_heatmap", 420)


def layout_f():
    """F — trellis, plus margins showing how the summary stat moves."""
    fig = make_subplots(
        rows=len(ROWS) + 1, cols=len(COLS) + 1,
        column_titles=[*[f"lag = {c}" for c in COLS], "mean cases"],
        row_titles=[*[f"sh = {r}" for r in ROWS], "corr"],
        shared_yaxes=False, horizontal_spacing=0.04, vertical_spacing=0.06,
        column_widths=[*([1] * len(COLS)), 0.8],
        row_heights=[*([1] * len(ROWS)), 0.7],
    )
    for i, sh in enumerate(ROWS, start=1):
        for j, lag in enumerate(COLS, start=1):
            fig.add_trace(
                go.Scatter(y=runs[(sh, lag)][CASES], line=dict(color=INK, width=1.1)),
                row=i, col=j,
            )
            fig.update_yaxes(range=[0, CASE_MAX], row=i, col=j)
    # Right margin: mean cases against spike_height, one line per lag.
    shades = ["#d0e3f0", "#7fb3d5", "#1f4e79"]
    for j, lag in enumerate(COLS):
        fig.add_trace(
            go.Scatter(
                x=[runs[(sh, lag)][CASES].mean() for sh in ROWS], y=ROWS,
                mode="lines+markers", line=dict(color=shades[j], width=1.5),
                name=f"lag {lag}", showlegend=True,
            ),
            row=1, col=len(COLS) + 1,
        )
    # Bottom margin: correlation against lag, one line per spike_height.
    for i, sh in enumerate(ROWS):
        fig.add_trace(
            go.Scatter(
                x=COLS,
                y=[runs[(sh, lag)][RAIN].corr(runs[(sh, lag)][CASES]) for lag in COLS],
                mode="lines+markers", line=dict(color=shades[i], width=1.5),
                name=f"sh {sh}",
            ),
            row=len(ROWS) + 1, col=1,
        )
    fig.update_layout(
        title="F — trellis + summary margins",
        showlegend=True, legend=dict(orientation="h", y=-0.1, font=dict(size=9)),
    )
    _save(fig, "f_grid_with_summary", 820)


if __name__ == "__main__":
    print(f"nine runs read from {OUT}")
    for fn in (layout_a, layout_b, layout_c, layout_d, layout_e, layout_f):
        fn()
    print(f"\nall six in {PLOTS}")
