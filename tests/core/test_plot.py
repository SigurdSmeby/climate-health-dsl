"""Tests for dataset plotting.

Plotting is hard to assert on visually, so these check the contract: a file
of the right type is written, multi-location and NaN data don't crash, and a
train/test split line is added when asked. They do not assert pixel content.
"""
import numpy as np
import pandas as pd
import pytest

from dsl.core.pipeline.plot import plot_dataset


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
