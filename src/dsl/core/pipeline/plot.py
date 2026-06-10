"""Plot a generated dataset for visual inspection.

Reading 100+ rows of CSV is no way to sanity-check a scenario; a chart of
the covariates and disease_cases over time makes a mistake (wrong lag,
flat signal, missing season) obvious at a glance. Output is either an
interactive HTML file (zoom / hover / toggle locations) or a static image
(PNG/SVG/PDF) for embedding in a report.

This module is the one place plotting decisions live, alongside output.py.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Identifier columns that are never their own panel (time_period is the x
# axis; location splits the lines). population is handled separately: it gets
# a panel only when it varies over time (growth), not when it's constant.
_NON_SERIES = ("time_period", "location", "population")

# Which file extensions we can write, and how. HTML is interactive; the rest
# are static images rendered by kaleido.
_IMAGE_EXTENSIONS = (".png", ".svg", ".pdf", ".jpg", ".jpeg")


def _series_columns(df: pd.DataFrame) -> list[str]:
    """Pick which columns get a panel: every covariate plus disease_cases,
    and population ONLY when it varies over time (a growth trajectory is
    worth a panel; a constant population is not)."""
    columns = [c for c in df.columns if c not in _NON_SERIES]
    # Population is plotted only when it varies OVER TIME — i.e. within at
    # least one location. Different constant populations across locations
    # (each flat) is not "time-varying" and should not get a panel.
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

    One stacked panel per variable (covariates and ``disease_cases``), with
    a line per location. The file type is taken from ``out_path``'s
    extension: ``.html`` is interactive, ``.png``/``.svg``/``.pdf`` are
    static images.

    Parameters
    ----------
    df:
        A generated dataset (as the engine produces it).
    out_path:
        Where to write; its extension selects the format.
    train_split:
        If given, draw a dashed vertical line at this period index to mark
        the train/test boundary.

    Raises
    ------
    ValueError
        If ``out_path`` has an unsupported extension.
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
        # Needs kaleido (a project dependency) to rasterize.
        fig.write_image(out_path)


# A fixed qualitative palette; each location is pinned to one entry so it has
# the SAME colour in every panel (makes lines easy to follow across panels).
_PALETTE = (
    "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
    "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52",
)


def _build_figure(df: pd.DataFrame, train_split: int | None = None) -> go.Figure:
    """Build the faceted figure: one panel per series, one line per location,
    with each location pinned to a single colour across all panels."""
    series_columns = _series_columns(df)
    locations = list(df["location"].unique()) if "location" in df.columns else [None]
    # Map each location to a fixed colour (cycling if there are many).
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
                    line_color=colour_for[loc],  # same colour in every panel
                    text=block["time_period"] if "time_period" in block else None,
                    legendgroup=str(loc),
                    # Only the first row contributes to the legend, so each
                    # location appears once rather than once per panel.
                    showlegend=(row == 1 and loc is not None),
                ),
                row=row,
                col=1,
            )

    if train_split is not None:
        # A dashed line on every panel marks where training data ends.
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
