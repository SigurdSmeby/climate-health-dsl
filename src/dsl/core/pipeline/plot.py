"""Plot a generated dataset for visual inspection.

A chart of the covariates and disease_cases over time makes a mistake (wrong
lag, flat signal, missing season) obvious at a glance. Output is interactive
HTML or a static image (PNG/SVG/PDF, rendered by kaleido).
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Identifier columns that are never their own panel. population is special:
# it gets a panel only when it varies over time (growth), not when constant.
_NON_SERIES = ("time_period", "location", "population")

_IMAGE_EXTENSIONS = (".png", ".svg", ".pdf", ".jpg", ".jpeg")

# Fixed qualitative palette; each location is pinned to one entry so it has
# the SAME colour in every panel.
_PALETTE = (
    "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
    "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52",
)


def _series_columns(df: pd.DataFrame) -> list[str]:
    """Columns that get a panel: covariates, disease_cases, and population
    only when it varies within at least one location (a growth trajectory
    is worth a panel; different constants across locations are not)."""
    columns = [c for c in df.columns if c not in _NON_SERIES]
    if "population" in df.columns:
        if "location" in df.columns:
            varies = df.groupby("location")["population"].nunique().gt(1).any()
        else:
            varies = df["population"].nunique() > 1
        if varies:
            columns.append("population")
    return columns


def plot_dataset(
    df: pd.DataFrame,
    out_path: str | Path,
    train_split: int | None = None,
) -> None:
    """Write a faceted time-series plot of ``df`` to ``out_path``.

    One stacked panel per variable, a line per location. The extension
    selects the format (.html interactive, image otherwise). If
    ``train_split`` is given, a dashed vertical line marks the train/test
    boundary at that period index.
    """
    out_path = Path(out_path)
    suffix = out_path.suffix.lower()
    if suffix != ".html" and suffix not in _IMAGE_EXTENSIONS:
        raise ValueError(
            f"unsupported plot extension '{out_path.suffix}'. Use .html "
            f"(interactive) or one of {_IMAGE_EXTENSIONS}."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = _build_figure(df, train_split)

    if suffix == ".html":
        fig.write_html(out_path)
    else:
        fig.write_image(out_path)


def _build_figure(df: pd.DataFrame, train_split: int | None = None) -> go.Figure:
    """One panel per series, one line per location, colours pinned per location."""
    series_columns = _series_columns(df)
    locations = list(df["location"].unique()) if "location" in df.columns else [None]
    colour_for = {
        loc: _PALETTE[i % len(_PALETTE)] for i, loc in enumerate(locations)
    }

    fig = make_subplots(
        rows=len(series_columns),
        cols=1,
        shared_xaxes=True,
        subplot_titles=series_columns,
    )

    for row, column in enumerate(series_columns, start=1):
        for loc in locations:
            block = df[df["location"] == loc] if loc is not None else df
            fig.add_trace(
                go.Scatter(
                    # Period index on x keeps panels aligned; the hover text
                    # carries the real time_period label.
                    x=list(range(len(block))),
                    y=block[column],
                    mode="lines",
                    name=str(loc),
                    line_color=colour_for[loc],
                    text=block["time_period"] if "time_period" in block else None,
                    legendgroup=str(loc),
                    # Legend only from the first row, so each location
                    # appears once rather than once per panel.
                    showlegend=(row == 1 and loc is not None),
                ),
                row=row,
                col=1,
            )

    if train_split is not None:
        for row in range(1, len(series_columns) + 1):
            fig.add_vline(
                x=train_split,
                line_dash="dash",
                line_color="red",
                row=row,
                col=1,
            )

    fig.update_layout(
        height=220 * len(series_columns),
        title="Generated dataset",
        showlegend=len(locations) > 1,
    )
    return fig
