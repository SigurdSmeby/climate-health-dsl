"""Tests for the CHAP-compatibility check on the finished DataFrame."""
import numpy as np
import pandas as pd

from dsl.core.pipeline.chap_check import validate_chap


def chap_ok_frame(n: int = 24) -> pd.DataFrame:
    """A small frame that satisfies every documented CHAP rule."""
    periods = [f"{2000 + i // 12}-{i % 12 + 1:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "time_period": periods * 2,
            "location": ["oslo"] * n + ["bergen"] * n,
            "rainfall": 1.0,
            "mean_temperature": 15.0,
            "disease_cases": 5.0,
            "population": 1000,
        }
    )


def test_valid_frame_has_no_findings():
    assert validate_chap(chap_ok_frame()) == []


def test_missing_required_column_flagged():
    df = chap_ok_frame().drop(columns=["disease_cases"])
    findings = validate_chap(df)
    assert any("disease_cases" in f for f in findings)


def test_missing_standard_covariate_flagged():
    # Legal DSL output, but standard CHAP models expect these names.
    df = chap_ok_frame().rename(columns={"rainfall": "rain"})
    findings = validate_chap(df)
    assert any("rainfall" in f for f in findings)


def test_non_chap_period_format_flagged():
    # Daily (YYYYMMDD) and yearly (YYYY) are not documented CHAP formats.
    df = chap_ok_frame(4)
    df["time_period"] = ["20000101", "20000102", "20000103", "20000104"] * 2
    findings = validate_chap(df)
    assert any("time_period" in f for f in findings)


def test_weekly_format_accepted():
    n = 8
    periods = [f"2000-W{i + 1:02d}" for i in range(n)]
    df = chap_ok_frame(n)
    df["time_period"] = periods * 2
    assert validate_chap(df) == []


def test_non_consecutive_periods_flagged():
    df = chap_ok_frame(4)
    # Skip a month: Jan, Feb, Apr, May.
    df["time_period"] = ["2000-01", "2000-02", "2000-04", "2000-05"] * 2
    findings = validate_chap(df)
    assert any("consecutive" in f for f in findings)


def test_locations_with_different_periods_flagged():
    df = chap_ok_frame(4)
    # Shift bergen's periods so the two location sets differ.
    df.loc[df["location"] == "bergen", "time_period"] = [
        "2001-01",
        "2001-02",
        "2001-03",
        "2001-04",
    ]
    findings = validate_chap(df)
    assert any("location" in f and "period" in f for f in findings)


def test_nan_in_covariate_flagged():
    df = chap_ok_frame()
    df.loc[3, "rainfall"] = np.nan
    findings = validate_chap(df)
    assert any("rainfall" in f and "NaN" in f for f in findings)


def test_nan_in_disease_cases_is_allowed():
    # CHAP tolerates missing disease values (it masks them itself); the lag
    # warm-up and missing_rate gaps must not be flagged.
    df = chap_ok_frame()
    df.loc[0, "disease_cases"] = np.nan
    assert validate_chap(df) == []


def test_all_nan_disease_cases_flagged():
    df = chap_ok_frame()
    df["disease_cases"] = np.nan
    findings = validate_chap(df)
    assert any("disease_cases" in f for f in findings)


def test_negative_disease_cases_flagged():
    df = chap_ok_frame()
    df.loc[5, "disease_cases"] = -3.0
    findings = validate_chap(df)
    assert any("negative" in f for f in findings)


def test_non_numeric_covariate_flagged():
    df = chap_ok_frame()
    df["rainfall"] = "wet"
    findings = validate_chap(df)
    assert any("rainfall" in f and "numeric" in f for f in findings)
