"""Tests for dataset plotting.

Plotting is hard to assert on visually, so these check the contract: a file
of the right type is written, multi-location and NaN data don't crash, and a
train/test split line is added when asked. They do not assert pixel content.
"""
import numpy as np
import pandas as pd
import pytest

from dsl.core.pipeline.plot import _build_figure, _series_columns, plot_dataset


def sample_frame(n=24, locations=("oslo",)):
    """An output-shaped frame, optionally multi-location, with a NaN warm-up."""
    periods = [f"{2000 + i // 12}-{i % 12 + 1:02d}" for i in range(n)]
    frames = []
    for loc in locations:
        cases = np.arange(n, dtype=float)
        cases[:2] = np.nan  # lag warm-up, like real output
        frames.append(
            pd.DataFrame(
                {
                    "time_period": periods,
                    "location": loc,
                    "rainfall": np.linspace(0, 10, n),
                    "mean_temperature": 15.0,
                    "disease_cases": cases,
                    "population": 1000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_writes_html(tmp_path):
    out = tmp_path / "plot.html"
    plot_dataset(sample_frame(), out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_writes_png(tmp_path):
    out = tmp_path / "plot.png"
    plot_dataset(sample_frame(), out)
    assert out.is_file()
    # A real rendered PNG is more than a few hundred bytes.
    assert out.stat().st_size > 1000


def test_constant_population_not_plotted():
    # A constant population must NOT get its own panel.
    df = sample_frame()  # population is constant 1000
    assert "population" not in _series_columns(df)


def test_varying_population_is_plotted():
    # A population that changes over time DOES get a panel.
    df = sample_frame()
    df["population"] = np.arange(len(df)) + 1000  # rising
    assert "population" in _series_columns(df)


def test_varying_population_plotted_without_location_column():
    # A frame with no location column: population varies → plotted.
    df = pd.DataFrame({
        "time_period": ["2000-01", "2000-02", "2000-03"],
        "rainfall": [1.0, 2.0, 3.0],
        "disease_cases": [1.0, 2.0, 3.0],
        "population": [100, 110, 120],
    })
    assert "population" in _series_columns(df)


def test_constant_per_location_population_not_plotted():
    # Two locations with different but individually CONSTANT
    # populations must not get a population panel (it doesn't vary over time).
    df = pd.DataFrame({
        "time_period": ["2000-01", "2000-02"] * 2,
        "location": ["A", "A", "B", "B"],
        "rainfall": [1.0, 2.0, 3.0, 4.0],
        "disease_cases": [1.0, 2.0, 3.0, 4.0],
        "population": [100, 100, 200, 200],
    })
    assert "population" not in _series_columns(df)


def test_population_growing_within_location_is_plotted():
    df = pd.DataFrame({
        "time_period": ["2000-01", "2000-02"] * 2,
        "location": ["A", "A", "B", "B"],
        "rainfall": [1.0, 2.0, 3.0, 4.0],
        "disease_cases": [1.0, 2.0, 3.0, 4.0],
        "population": [100, 110, 200, 200],  # A grows
    })
    assert "population" in _series_columns(df)


def test_each_location_has_one_colour_across_panels():
    # A location must be the SAME colour in every panel, so a reader can
    # follow it down the stacked panels.
    df = sample_frame(locations=("oslo", "bergen", "trondheim"))
    fig = _build_figure(df)
    # Collect the colour used for each location name across all traces/panels.
    colours_by_location: dict[str, set] = {}
    for trace in fig.data:
        colours_by_location.setdefault(trace.name, set()).add(trace.line.color)
    # Every location uses exactly one colour...
    for name, colours in colours_by_location.items():
        assert len(colours) == 1, f"{name} has multiple colours: {colours}"
    # ...and the locations use DIFFERENT colours from each other.
    all_colours = [next(iter(c)) for c in colours_by_location.values()]
    assert len(set(all_colours)) == len(all_colours)


def test_multi_location_does_not_crash(tmp_path):
    out = tmp_path / "multi.html"
    plot_dataset(sample_frame(locations=("oslo", "bergen")), out)
    assert out.is_file()


def test_nan_values_do_not_crash(tmp_path):
    df = sample_frame()
    df.loc[5, "rainfall"] = np.nan  # a missing covariate too
    out = tmp_path / "nan.html"
    plot_dataset(df, out)
    assert out.is_file()


def test_train_split_line_accepted(tmp_path):
    # Passing a split index must not error and still writes the file.
    out = tmp_path / "split.html"
    plot_dataset(sample_frame(), out, train_split=18)
    assert out.is_file()


def test_unknown_extension_rejected(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        plot_dataset(sample_frame(), tmp_path / "plot.gif")


def test_output_dir_created(tmp_path):
    out = tmp_path / "nested" / "dir" / "plot.html"
    plot_dataset(sample_frame(), out)
    assert out.is_file()


# --------------------------------------------------------------- year gridlines


def test_monthly_ticks_land_on_whole_years():
    """A tick every 12 periods, so the same month sits under every tick.

    plotly's own choice is a round number of periods (10 on monthly data),
    which drifts two months per tick and makes a recurring annual shape
    impossible to line up by eye.
    """
    fig = _build_figure(sample_frame(), period="monthly")
    assert fig.layout.xaxis.dtick == 12


def test_tick_spacing_follows_the_resolution():
    for period, expected in (
        ("daily", 365),
        ("weekly", 52),
        ("monthly", 12),
        ("yearly", 1),
    ):
        fig = _build_figure(sample_frame(), period=period)
        assert fig.layout.xaxis.dtick == expected, period


def test_ticks_start_at_the_first_period():
    """Without an anchor plotly may offset the grid, so year 1 drifts."""
    fig = _build_figure(sample_frame(), period="monthly")
    assert fig.layout.xaxis.tick0 == 0


def test_every_panel_shares_the_year_grid():
    """Panels are stacked and compared vertically, so one grid serves all."""
    df = sample_frame()
    fig = _build_figure(df, period="monthly")
    panels = len(_series_columns(df))
    for row in range(1, panels + 1):
        axis = fig.layout[f"xaxis{row}" if row > 1 else "xaxis"]
        assert axis.dtick == 12, f"panel {row} has dtick {axis.dtick}"


def test_period_is_optional():
    """An unknown resolution leaves plotly's default rather than erroring."""
    fig = _build_figure(sample_frame())
    assert fig.layout.xaxis.dtick is None


def test_plot_dataset_passes_the_period_through(tmp_path):
    out = tmp_path / "ticks.html"
    plot_dataset(sample_frame(), out, period="monthly")
    assert out.is_file()


def test_explicit_tick_spacing_overrides_the_year_default():
    """A half-year grid on monthly data, for reading a shorter cycle."""
    fig = _build_figure(sample_frame(), period="monthly", tick_every=6)
    assert fig.layout.xaxis.dtick == 6


def test_explicit_spacing_works_without_a_period():
    fig = _build_figure(sample_frame(), tick_every=3)
    assert fig.layout.xaxis.dtick == 3


def test_tick_spacing_of_one_marks_every_period():
    fig = _build_figure(sample_frame(), period="monthly", tick_every=1)
    assert fig.layout.xaxis.dtick == 1
