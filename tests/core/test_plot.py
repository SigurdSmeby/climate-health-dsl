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
    """A CHAP-shaped frame, optionally multi-location, with a NaN warm-up."""
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
