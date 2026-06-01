"""Tests for the from_csv generator (real-data-backed covariates)."""
import numpy as np
import pandas as pd
import pytest

from dsl.generators.from_csv import FromCsvGenerator


@pytest.fixture
def csv_file(tmp_path):
    """A small single-location CHAP-format CSV (24 monthly periods)."""
    periods = [f"{2010 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]
    df = pd.DataFrame(
        {
            "time_period": periods,
            "rainfall": np.arange(24, dtype=float),
            "mean_temperature": 20.0,
        }
    )
    path = tmp_path / "real.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multi_location_csv(tmp_path):
    """A CHAP-format CSV with two locations, 12 monthly periods each."""
    periods = [f"2010-{m + 1:02d}" for m in range(12)]
    df = pd.DataFrame(
        {
            "time_period": periods * 2,
            "rainfall": list(range(12)) + list(range(100, 112)),
            "location": ["north"] * 12 + ["south"] * 12,
        }
    )
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


def test_values_pass_through_unchanged(csv_file, rng):
    gen = FromCsvGenerator(file=str(csv_file), column="rainfall")
    series = gen.generate(10, "monthly", rng)
    assert np.array_equal(series, np.arange(10, dtype=float))


def test_output_length_is_n_periods(csv_file, rng):
    gen = FromCsvGenerator(file=str(csv_file), column="rainfall")
    assert len(gen.generate(7, "monthly", rng)) == 7


def test_too_few_rows_is_an_error_not_a_wrap(csv_file, rng):
    # The CSV has 24 periods; asking for 30 must refuse — never wrap or
    # extrapolate beyond the real data's dates.
    gen = FromCsvGenerator(file=str(csv_file), column="rainfall")
    with pytest.raises(ValueError, match="24"):
        gen.generate(30, "monthly", rng)


def test_missing_file_is_an_error(rng):
    with pytest.raises(ValueError, match="nope.csv"):
        FromCsvGenerator(file="nope.csv", column="rainfall")


def test_missing_column_lists_available(csv_file, rng):
    gen = FromCsvGenerator(file=str(csv_file), column="wind")
    with pytest.raises(ValueError) as excinfo:
        gen.generate(5, "monthly", rng)
    assert "wind" in str(excinfo.value)
    assert "rainfall" in str(excinfo.value)  # helpful: what IS there


def test_multi_location_requires_source_location(multi_location_csv, rng):
    gen = FromCsvGenerator(file=str(multi_location_csv), column="rainfall")
    with pytest.raises(ValueError) as excinfo:
        gen.generate(5, "monthly", rng)
    # The error must list the available source locations.
    assert "north" in str(excinfo.value) and "south" in str(excinfo.value)


def test_source_location_filters(multi_location_csv, rng):
    gen = FromCsvGenerator(
        file=str(multi_location_csv), column="rainfall", source_location="south"
    )
    series = gen.generate(5, "monthly", rng)
    assert np.array_equal(series, np.arange(100, 105, dtype=float))


def test_unknown_source_location_is_an_error(multi_location_csv, rng):
    gen = FromCsvGenerator(
        file=str(multi_location_csv), column="rainfall", source_location="east"
    )
    with pytest.raises(ValueError, match="east"):
        gen.generate(5, "monthly", rng)


def test_start_period_slices(csv_file, rng):
    gen = FromCsvGenerator(
        file=str(csv_file), column="rainfall", start_period="2010-07"
    )
    series = gen.generate(5, "monthly", rng)
    assert np.array_equal(series, np.arange(6, 11, dtype=float))


def test_unknown_start_period_is_an_error(csv_file, rng):
    gen = FromCsvGenerator(
        file=str(csv_file), column="rainfall", start_period="2030-01"
    )
    with pytest.raises(ValueError, match="2030-01"):
        gen.generate(5, "monthly", rng)


def test_period_resolution_mismatch_is_an_error(csv_file, rng):
    # Monthly source data cannot back a weekly scenario.
    gen = FromCsvGenerator(file=str(csv_file), column="rainfall")
    with pytest.raises(ValueError, match="weekly"):
        gen.generate(5, "weekly", rng)


def test_output_is_seed_independent(csv_file):
    # Real data involves no randomness: any seed gives the same series.
    gen = FromCsvGenerator(file=str(csv_file), column="rainfall")
    a = gen.generate(10, "monthly", np.random.default_rng(1))
    b = gen.generate(10, "monthly", np.random.default_rng(2))
    assert np.array_equal(a, b)


def test_works_with_bundled_laos_data(rng):
    # The shipped example: real CHAP data, 36 monthly periods per province.
    gen = FromCsvGenerator(
        file="examples/data/laos_subset.csv",
        column="rainfall",
        source_location="Bokeo",
    )
    series = gen.generate(36, "monthly", rng)
    assert len(series) == 36
    assert series[0] == pytest.approx(37.965)
