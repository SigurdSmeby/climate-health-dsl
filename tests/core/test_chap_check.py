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


def test_custom_covariate_name_is_allowed():
    # CHAP accepts arbitrary covariate columns (verified against
    # chap_core CSV ingest), so a non-standard name must NOT be flagged.
    df = chap_ok_frame().rename(columns={"rainfall": "wind"})
    assert validate_chap(df) == []


def test_population_optional():
    # chap_core has HealthData (no population) alongside FullData; a frame
    # without population is valid CHAP.
    df = chap_ok_frame().drop(columns=["population"])
    assert validate_chap(df) == []


def test_daily_format_accepted():
    # CHAP's TimePeriod.parse accepts daily YYYYMMDD — must not be flagged.
    df = chap_ok_frame(4)
    df["time_period"] = ["20000101", "20000102", "20000103", "20000104"] * 2
    assert validate_chap(df) == []


def test_yearly_format_accepted():
    df = chap_ok_frame(3)
    df["time_period"] = ["2000", "2001", "2002"] * 2
    assert validate_chap(df) == []


def test_weekly_format_accepted():
    n = 8
    periods = [f"2000-W{i + 1:02d}" for i in range(n)]
    df = chap_ok_frame(n)
    df["time_period"] = periods * 2
    assert validate_chap(df) == []


def test_unparseable_period_flagged():
    # A label matching no CHAP resolution is still a real problem.
    df = chap_ok_frame(4)
    df["time_period"] = ["junk1", "junk2", "junk3", "junk4"] * 2
    findings = validate_chap(df)
    assert any("time_period" in f for f in findings)


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



def _frame(periods):
    return pd.DataFrame({
        "time_period": periods,
        "location": ["x"] * len(periods),
        "rainfall": [1.0] * len(periods),
        "disease_cases": [1.0] * len(periods),
    })


def test_daily_gap_flagged():
    # Daily consecutiveness must be checked.
    findings = validate_chap(_frame(["20000101", "20000103"]))
    assert any("consecutive" in f for f in findings)


def test_yearly_gap_flagged():
    findings = validate_chap(_frame(["2000", "2002"]))
    assert any("consecutive" in f for f in findings)


def test_daily_consecutive_ok():
    assert validate_chap(_frame(["20000101", "20000102", "20000103"])) == []


def test_yearly_consecutive_ok():
    assert validate_chap(_frame(["2000", "2001", "2002"])) == []


def test_string_disease_cases_does_not_crash():
    # Validate_chap must never raise, even on a string target.
    df = pd.DataFrame({
        "time_period": ["2000-01", "2000-02"],
        "location": ["x", "x"],
        "rainfall": [1.0, 2.0],
        "disease_cases": ["1", "2"],
    })
    findings = validate_chap(df)  # must not raise
    assert any("disease_cases" in f for f in findings)


def test_sunday_weeks_not_falsely_flagged():
    # YYYY-Snn is a valid CHAP form; consecutive S-weeks are fine.
    assert validate_chap(_frame(["2000-S01", "2000-S02", "2000-S03"])) == []


def test_week_53_not_falsely_flagged():
    # Week 53 exists in some years and must not be flagged.
    assert validate_chap(_frame(["2020-W52", "2020-W53", "2021-W01"])) == []


def test_flat52_week_rollover_not_flagged():
    # The DSL emits flat-52 weekly labels: W52 rolls straight to next year's
    # W01 (no W53). chap_check must accept its OWN output as consecutive, even
    # in an ISO-53-week year like 2020.
    assert validate_chap(_frame(["2020-W51", "2020-W52", "2021-W01"])) == []


def test_genuinely_non_consecutive_weeks_still_flagged():
    # A real gap (W50 -> W52) must still be flagged.
    findings = validate_chap(_frame(["2020-W50", "2020-W52"]))
    assert any("consecutive" in f for f in findings)


def test_year_boundary_week_gaps_still_flagged():
    # The rollover acceptance must be tight: only W52/W53 -> W01 is allowed.
    # A gap across the boundary (W01 or W52 missing) is still a real gap.
    assert any(  # W01 skipped
        "consecutive" in f for f in validate_chap(_frame(["2020-W52", "2021-W02"]))
    )
    assert any(  # W52 skipped
        "consecutive" in f for f in validate_chap(_frame(["2020-W51", "2021-W01"]))
    )


def test_infinite_covariate_flagged():
    # An infinite covariate value is not CHAP-valid data.
    df = _frame(["2000-01", "2000-02"])
    df.loc[0, "rainfall"] = np.inf
    findings = validate_chap(df)
    assert any("rainfall" in f for f in findings)
