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

# Columns that are identifiers or constants, not series worth their own panel.
_NON_SERIES = ("time_period", "location", "population")

# Which file extensions we can write, and how. HTML is interactive; the rest
# are static images rendered by kaleido.
_IMAGE_EXTENSIONS = (".png", ".svg", ".pdf", ".jpg", ".jpeg")


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

    # The series to plot, in their column order: every column that isn't an
    # identifier or the constant population.
    series_columns = [c for c in df.columns if c not in _NON_SERIES]

    locations = list(df["location"].unique()) if "location" in df.columns else [None]

    # One row per variable, sharing the x-axis so periods line up vertically.
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

    if suffix == ".html":
        fig.write_html(out_path)
    else:
        # Needs kaleido (a project dependency) to rasterize.
        fig.write_image(out_path)
