"""Tests for the from_csv generator (real-data-backed covariates)."""
import numpy as np
import pandas as pd
import pytest

from dsl.generators.from_csv import FromCsvGenerator
from tests.conftest import write_csv


@pytest.fixture
def csv_file(tmp_path):
    """A small single-location CHAP-format CSV (24 monthly periods)."""
    periods = [f"{2010 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]
    return write_csv(
        tmp_path / "real.csv", periods,
        rainfall=np.arange(24, dtype=float), mean_temperature=20.0,
    )


@pytest.fixture
def multi_location_csv(tmp_path):
    """A CHAP-format CSV with two locations, 12 monthly periods each."""
    periods = [f"2010-{m + 1:02d}" for m in range(12)]
    return write_csv(
        tmp_path / "multi.csv", periods * 2,
        rainfall=list(range(12)) + list(range(100, 112)),
        location=["north"] * 12 + ["south"] * 12,
    )


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


# --- Round 2 bug fixes (#9–#12) ---


def test_from_csv_unsorted_periods_are_sorted(tmp_path, rng):
    # Bug #9: rows out of time order must be sorted by period before slicing,
    # so values map to the right periods (not file order).
    csv = tmp_path / "unsorted.csv"
    pd.DataFrame(
        {"time_period": ["2010-03", "2010-01", "2010-02", "2010-04"],
         "rainfall": [30.0, 10.0, 20.0, 40.0]}
    ).to_csv(csv, index=False)
    gen = FromCsvGenerator(file=str(csv), column="rainfall", start_period="2010-01")
    # Jan, Feb, Mar — must be 10, 20, 30 (not 10, 20, 40 in file order).
    assert list(gen.generate(3, "monthly", rng)) == [10.0, 20.0, 30.0]


def test_from_csv_unsorted_without_start_period(tmp_path, rng):
    csv = tmp_path / "unsorted2.csv"
    pd.DataFrame(
        {"time_period": ["2010-03", "2010-01", "2010-02"],
         "rainfall": [3.0, 1.0, 2.0]}
    ).to_csv(csv, index=False)
    gen = FromCsvGenerator(file=str(csv), column="rainfall")
    assert list(gen.generate(3, "monthly", rng)) == [1.0, 2.0, 3.0]


def test_from_csv_duplicate_periods_rejected(tmp_path, rng):
    # Bug #9: a repeated period silently shifts alignment — reject it.
    csv = tmp_path / "dup.csv"
    pd.DataFrame(
        {"time_period": ["2010-01", "2010-01", "2010-02"],
         "rainfall": [10.0, 999.0, 20.0]}
    ).to_csv(csv, index=False)
    gen = FromCsvGenerator(file=str(csv), column="rainfall")
    with pytest.raises(ValueError, match="duplicate"):
        gen.generate(2, "monthly", rng)


def test_from_csv_requires_time_period_for_start(tmp_path, rng):
    # Bug #10: no time_period column means start_period can't be honored.
    csv = tmp_path / "notp.csv"
    pd.DataFrame({"date": ["2010-01", "2010-02"], "rainfall": [1.0, 2.0]}).to_csv(
        csv, index=False
    )
    gen = FromCsvGenerator(file=str(csv), column="rainfall", start_period="2010-02")
    with pytest.raises(ValueError, match="time_period"):
        gen.generate(2, "monthly", rng)


def test_from_csv_empty_file(tmp_path, rng):
    # Bug #11: an empty file must give a clear from_csv error.
    csv = tmp_path / "empty.csv"
    csv.write_text("")
    with pytest.raises(ValueError, match="from_csv"):
        FromCsvGenerator(file=str(csv), column="rainfall").generate(2, "monthly", rng)


def test_from_csv_header_only(tmp_path, rng):
    # Bug #11: a header with no data rows must give a clear error, not IndexError.
    csv = tmp_path / "ho.csv"
    csv.write_text("time_period,rainfall\n")
    with pytest.raises(ValueError, match="no data"):
        FromCsvGenerator(file=str(csv), column="rainfall").generate(2, "monthly", rng)


def test_from_csv_non_numeric_clear_error(tmp_path, rng):
    # Bug #12: text in the numeric column should give a clear from_csv message.
    csv = tmp_path / "text.csv"
    pd.DataFrame(
        {"time_period": ["2010-01", "2010-02", "2010-03"],
         "rainfall": ["1", "heavy", "3"]}
    ).to_csv(csv, index=False)
    gen = FromCsvGenerator(file=str(csv), column="rainfall")
    with pytest.raises(ValueError, match="rainfall"):
        gen.generate(3, "monthly", rng)


# --- Round 4/5 from_csv integrity (#13, #27, #31) ---


def test_from_csv_gap_in_periods_rejected(tmp_path, rng):
    # Bug #13: a missing period in the middle must be rejected, not silently
    # relabelled (Feb absent -> Mar's value written as Feb).
    csv = tmp_path / "gaps.csv"
    pd.DataFrame(
        {"time_period": ["2010-01", "2010-03", "2010-04"], "rainfall": [10.0, 30.0, 40.0]}
    ).to_csv(csv, index=False)
    gen = FromCsvGenerator(file=str(csv), column="rainfall")
    with pytest.raises(ValueError, match="consecutive|gap"):
        gen.generate(3, "monthly", rng)


def test_from_csv_consecutive_periods_ok(tmp_path, rng):
    csv = tmp_path / "ok.csv"
    pd.DataFrame(
        {"time_period": ["2010-01", "2010-02", "2010-03"], "rainfall": [1.0, 2.0, 3.0]}
    ).to_csv(csv, index=False)
    gen = FromCsvGenerator(file=str(csv), column="rainfall")
    assert list(gen.generate(3, "monthly", rng)) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("label", ["2010-00", "2010-13", "2010-99"])
def test_from_csv_impossible_monthly_label_rejected(tmp_path, rng, label):
    # Bug #27: shape-valid but impossible calendar labels must be rejected.
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"time_period": [label, "2010-02"], "rainfall": [1.0, 2.0]}).to_csv(
        csv, index=False
    )
    gen = FromCsvGenerator(file=str(csv), column="rainfall")
    with pytest.raises(ValueError):
        gen.generate(2, "monthly", rng)


def test_from_csv_infinite_value_rejected(tmp_path, rng):
    # Bug #31: inf is numeric but not a valid covariate value.
    csv = tmp_path / "inf.csv"
    pd.DataFrame(
        {"time_period": ["2010-01", "2010-02", "2010-03"],
         "rainfall": [1.0, float("inf"), 3.0]}
    ).to_csv(csv, index=False)
    gen = FromCsvGenerator(file=str(csv), column="rainfall")
    with pytest.raises(ValueError, match="finite|infinite|rainfall"):
        gen.generate(3, "monthly", rng)
