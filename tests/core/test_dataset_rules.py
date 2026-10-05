"""Tests for the dataset-rule check on the finished DataFrame."""
import numpy as np
import pandas as pd

from dsl.core.pipeline.dataset_rules import check_dataset


def valid_frame(n: int = 24) -> pd.DataFrame:
    """A small frame that satisfies every documented rule."""
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
    assert check_dataset(valid_frame()) == []


def test_missing_required_column_flagged():
    df = valid_frame().drop(columns=["disease_cases"])
    findings = check_dataset(df)
    assert any("disease_cases" in f for f in findings)


def test_custom_covariate_name_is_allowed():
    # Driver columns may have any name (verified against a real consumer's
    # CSV ingest), so a non-standard name must NOT be flagged.
    df = valid_frame().rename(columns={"rainfall": "wind"})
    assert check_dataset(df) == []


def test_population_optional():
    # A consumer may accept case counts without population, so a frame
    # without it is valid.
    df = valid_frame().drop(columns=["population"])
    assert check_dataset(df) == []


def test_daily_format_accepted():
    # the format's TimePeriod.parse accepts daily YYYYMMDD — must not be flagged.
    df = valid_frame(4)
    df["time_period"] = ["20000101", "20000102", "20000103", "20000104"] * 2
    assert check_dataset(df) == []


def test_yearly_format_accepted():
    df = valid_frame(3)
    df["time_period"] = ["2000", "2001", "2002"] * 2
    assert check_dataset(df) == []


def test_weekly_format_accepted():
    n = 8
    periods = [f"2000-W{i + 1:02d}" for i in range(n)]
    df = valid_frame(n)
    df["time_period"] = periods * 2
    assert check_dataset(df) == []


def test_unparseable_period_flagged():
    # A label matching no the rules resolution is still a real problem.
    df = valid_frame(4)
    df["time_period"] = ["junk1", "junk2", "junk3", "junk4"] * 2
    findings = check_dataset(df)
    assert any("time_period" in f for f in findings)


def test_non_consecutive_periods_flagged():
    df = valid_frame(4)
    # Skip a month: Jan, Feb, Apr, May.
    df["time_period"] = ["2000-01", "2000-02", "2000-04", "2000-05"] * 2
    findings = check_dataset(df)
    assert any("consecutive" in f for f in findings)


def test_locations_with_different_periods_flagged():
    df = valid_frame(4)
    # Shift bergen's periods so the two location sets differ.
    df.loc[df["location"] == "bergen", "time_period"] = [
        "2001-01",
        "2001-02",
        "2001-03",
        "2001-04",
    ]
    findings = check_dataset(df)
    assert any("location" in f and "period" in f for f in findings)


def test_nan_in_covariate_flagged():
    df = valid_frame()
    df.loc[3, "rainfall"] = np.nan
    findings = check_dataset(df)
    assert any("rainfall" in f and "NaN" in f for f in findings)


def test_nan_in_disease_cases_is_allowed():
    # the rules tolerates missing disease values (it masks them itself); the lag
    # warm-up and missing_rate gaps must not be flagged.
    df = valid_frame()
    df.loc[0, "disease_cases"] = np.nan
    assert check_dataset(df) == []


def test_all_nan_disease_cases_flagged():
    df = valid_frame()
    df["disease_cases"] = np.nan
    findings = check_dataset(df)
    assert any("disease_cases" in f for f in findings)


def test_negative_disease_cases_flagged():
    df = valid_frame()
    df.loc[5, "disease_cases"] = -3.0
    findings = check_dataset(df)
    assert any("negative" in f for f in findings)


def test_non_numeric_covariate_flagged():
    df = valid_frame()
    df["rainfall"] = "wet"
    findings = check_dataset(df)
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
    findings = check_dataset(_frame(["20000101", "20000103"]))
    assert any("consecutive" in f for f in findings)


def test_yearly_gap_flagged():
    findings = check_dataset(_frame(["2000", "2002"]))
    assert any("consecutive" in f for f in findings)


def test_daily_consecutive_ok():
    assert check_dataset(_frame(["20000101", "20000102", "20000103"])) == []


def test_yearly_consecutive_ok():
    assert check_dataset(_frame(["2000", "2001", "2002"])) == []


def test_string_disease_cases_does_not_crash():
    # check_dataset must never raise, even on a string target.
    df = pd.DataFrame({
        "time_period": ["2000-01", "2000-02"],
        "location": ["x", "x"],
        "rainfall": [1.0, 2.0],
        "disease_cases": ["1", "2"],
    })
    findings = check_dataset(df)  # must not raise
    assert any("disease_cases" in f for f in findings)


def test_sunday_weeks_not_falsely_flagged():
    # YYYY-Snn is a valid the rules form; consecutive S-weeks are fine.
    assert check_dataset(_frame(["2000-S01", "2000-S02", "2000-S03"])) == []


def test_week_53_not_falsely_flagged():
    # Week 53 exists in some years and must not be flagged.
    assert check_dataset(_frame(["2020-W52", "2020-W53", "2021-W01"])) == []


def test_flat52_week_rollover_not_flagged():
    # The DSL emits flat-52 weekly labels: W52 rolls straight to next year's
    # W01 (no W53). dataset_rules must accept its OWN output as consecutive, even
    # in an ISO-53-week year like 2020.
    assert check_dataset(_frame(["2020-W51", "2020-W52", "2021-W01"])) == []


def test_genuinely_non_consecutive_weeks_still_flagged():
    # A real gap (W50 -> W52) must still be flagged.
    findings = check_dataset(_frame(["2020-W50", "2020-W52"]))
    assert any("consecutive" in f for f in findings)


def test_year_boundary_week_gaps_still_flagged():
    # The rollover acceptance must be tight: only W52/W53 -> W01 is allowed.
    # A gap across the boundary (W01 or W52 missing) is still a real gap.
    assert any(  # W01 skipped
        "consecutive" in f for f in check_dataset(_frame(["2020-W52", "2021-W02"]))
    )
    assert any(  # W52 skipped
        "consecutive" in f for f in check_dataset(_frame(["2020-W51", "2021-W01"]))
    )


def test_infinite_covariate_flagged():
    # An infinite covariate value is not the rules-valid data.
    df = _frame(["2000-01", "2000-02"])
    df.loc[0, "rainfall"] = np.inf
    findings = check_dataset(df)
    assert any("rainfall" in f for f in findings)


def test_warmup_nan_in_a_covariate_is_not_a_finding():
    """A series built from a lagged parent has no input for its opening
    periods, so a LEADING run of NaN is the declared warm-up, not a defect."""
    periods = [f"2010-{m:02d}" for m in range(1, 13)]
    df = pd.DataFrame({
        "time_period": periods,
        "location": "loc",
        "soil_moisture": [float("nan")] * 2 + [1.0] * 10,
        "disease_cases": [5.0] * 12,
        "population": [1000] * 12,
    })
    assert not any("soil_moisture" in f for f in check_dataset(df))


def test_a_gap_after_the_warmup_is_still_a_finding():
    periods = [f"2010-{m:02d}" for m in range(1, 13)]
    values = [1.0] * 12
    values[7] = float("nan")  # a hole in the middle, not warm-up
    df = pd.DataFrame({
        "time_period": periods,
        "location": "loc",
        "rainfall": values,
        "disease_cases": [5.0] * 12,
        "population": [1000] * 12,
    })
    assert any("rainfall" in f for f in check_dataset(df))
