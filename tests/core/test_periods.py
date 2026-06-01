"""Tests for the time-axis helpers: periods-per-year and CHAP period labels."""
import pytest

from dsl.core.pipeline.periods import format_period, parse_period, periods_per_year


# ------------------------------------------------------------ periods_per_year


def test_periods_per_year_values():
    assert periods_per_year("daily") == 365
    assert periods_per_year("weekly") == 52
    assert periods_per_year("monthly") == 12
    assert periods_per_year("yearly") == 1


def test_periods_per_year_unknown_raises():
    with pytest.raises(KeyError, match="fortnightly"):
        periods_per_year("fortnightly")


# -------------------------------------------------------------- format_period


def test_weekly_first_period():
    assert format_period(0, "weekly") == "2000-W01"


def test_weekly_zero_padded():
    assert format_period(4, "weekly") == "2000-W05"


def test_weekly_year_rollover():
    # 52 weeks per year: index 52 is the first week of the next year.
    assert format_period(51, "weekly") == "2000-W52"
    assert format_period(52, "weekly") == "2001-W01"


def test_monthly_first_period():
    assert format_period(0, "monthly") == "2000-01"


def test_monthly_year_rollover():
    assert format_period(11, "monthly") == "2000-12"
    assert format_period(12, "monthly") == "2001-01"


def test_daily_format_is_yyyymmdd():
    # CHAP daily periods are compact YYYYMMDD strings, not ISO dates.
    assert format_period(0, "daily") == "20000101"
    assert format_period(31, "daily") == "20000201"


def test_daily_year_rollover():
    # 2000 is a leap year (366 days), so index 366 lands on 2001-01-01.
    assert format_period(365, "daily") == "20001231"
    assert format_period(366, "daily") == "20010101"


def test_yearly_format():
    assert format_period(0, "yearly") == "2000"
    assert format_period(3, "yearly") == "2003"


def test_custom_start_year():
    assert format_period(0, "monthly", start_year=2015) == "2015-01"
    assert format_period(0, "yearly", start_year=1990) == "1990"


def test_unknown_period_raises():
    with pytest.raises(KeyError, match="fortnightly"):
        format_period(0, "fortnightly")


# --------------------------------------------------------------- parse_period


def test_parse_period_round_trips_all_resolutions():
    # parse_period is format_period's inverse: parsing a label and
    # formatting the result must give the label back, for every resolution.
    for period, label in [
        ("daily", "20100615"),
        ("weekly", "2015-W10"),
        ("monthly", "2010-07"),
        ("yearly", "2003"),
    ]:
        year, offset = parse_period(label, period)
        assert format_period(offset, period, start_year=year) == label


def test_parse_period_values():
    assert parse_period("2010-07", "monthly") == (2010, 6)
    assert parse_period("2015-W10", "weekly") == (2015, 9)
    assert parse_period("2003", "yearly") == (2003, 0)
    assert parse_period("20100103", "daily") == (2010, 2)


def test_parse_period_wrong_format_raises():
    with pytest.raises(ValueError, match="2010-07"):
        parse_period("2010-07", "weekly")  # monthly label, weekly resolution
    with pytest.raises(ValueError, match="garbage"):
        parse_period("garbage", "monthly")
