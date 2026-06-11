"""Checks the finished DataFrame against CHAP's dataset rules.

The DSL can generate datasets that are valid but not usable by CHAP. This
module catches that *before* the files are written. The rules were verified
against the chap-core source (not just one example dataset):

- Required columns: ``time_period``, ``location``, ``disease_cases``
  (``chap_core/datatypes.py``). ``population`` is optional — chap-core has a
  ``HealthData`` class without it — and covariate columns may have any name,
  so neither is required here.
- ``time_period`` may be any resolution CHAP's ``TimePeriod.parse`` accepts:
  daily (``YYYYMMDD``), weekly (``YYYY-Wnn``, ``YYYY-Snn``, or a
  ``start/end`` date range), monthly (``YYYY-MM``), or yearly (``YYYY``).
- Periods consecutive, and identical across locations. CHAP can auto-fill
  these, so they are advisory findings (they may still matter to a model),
  not hard requirements.
- No NaN in covariate columns (NaN in ``disease_cases`` is fine: CHAP masks
  missing case counts itself).

Like ``validate_scenario``, this returns human-readable findings and never
raises; the CLI prints them as warnings, or refuses to write with
``--strict-chap``.
"""
import datetime
import re

import numpy as np
import pandas as pd

# The columns CHAP genuinely requires (chap_core/datatypes.py).
REQUIRED_COLUMNS = ("time_period", "location", "disease_cases")

# Period label formats CHAP's TimePeriod.parse accepts, by resolution. The
# weekly cases cover Monday-start (-W), Sunday-start (-S), and the
# start/end date-range form; monthly, yearly, and daily are the others.
_PERIOD_FORMATS = {
    "monthly": re.compile(r"^\d{4}-(0[1-9]|1[0-2])$"),
    "weekly": re.compile(r"^\d{4}-[WS](0[1-9]|[1-4]\d|5[0-3])$"),
    "weekly_range": re.compile(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$"),
    "daily": re.compile(r"^\d{8}$"),
    "yearly": re.compile(r"^\d{4}$"),
}


def validate_chap(df: pd.DataFrame) -> list[str]:
    """Return findings for everything CHAP would reject. Never raises.

    An empty list means the DataFrame follows CHAP's dataset rules.
    """
    findings: list[str] = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            findings.append(f"CHAP requires a '{column}' column, which is missing.")

    # The remaining checks read the columns, so they need them present.
    if "time_period" in df.columns:
        findings.extend(_check_periods(df))
    findings.extend(_check_values(df))
    return findings


def _detect_resolution(periods: pd.Series) -> str | None:
    """Return the resolution name all labels share, or None if none fits."""
    for resolution, pattern in _PERIOD_FORMATS.items():
        if periods.str.match(pattern).all():
            return resolution
    return None


def _check_periods(df: pd.DataFrame) -> list[str]:
    """Period format, consecutiveness, and equality across locations."""
    findings: list[str] = []
    periods = df["time_period"].astype(str)

    resolution = _detect_resolution(periods)
    if resolution is None:
        findings.append(
            "time_period values are not in a CHAP-parseable format "
            "(expected daily YYYYMMDD, weekly YYYY-Wnn, monthly YYYY-MM, "
            "yearly YYYY, or a YYYY-MM-DD/YYYY-MM-DD week range)."
        )
        return findings  # format unknown → can't check order either

    groups = (
        df.groupby("location", sort=False)["time_period"].apply(tuple)
        if "location" in df.columns
        else pd.Series({"all": tuple(periods)})
    )

    # Advisory: CHAP can auto-fill, but mismatched periods across locations
    # often signal a mistake and may trip up some models.
    if groups.nunique() > 1:
        findings.append(
            "locations do not share the same set of time periods (CHAP can "
            "auto-fill, but this is often a mistake)."
        )

    # Consecutiveness: map each label to its start date and require each step
    # to advance by exactly one period's worth of time. This works uniformly
    # for daily/weekly/monthly/yearly and for Sunday weeks / week 53 (the
    # date-range week form is skipped — its span is self-describing).
    if resolution == "weekly_range":
        return findings
    for location, sequence in groups.items():
        for current, following in zip(sequence, sequence[1:], strict=False):
            if not _consecutive(current, following, resolution):
                findings.append(
                    f"time periods for location '{location}' are not "
                    f"consecutive: '{following}' follows '{current}'."
                )
                break  # one finding per location is enough

    return findings


def _consecutive(current: str, following: str, resolution: str) -> bool:
    """True if ``following`` is exactly one period after ``current``."""
    if resolution == "weekly":
        return _weekly_consecutive(current, following)
    try:
        a = _period_start_date(current, resolution)
        b = _period_start_date(following, resolution)
    except ValueError:
        return True  # unparseable label; the format check already flagged it
    return _is_one_step(a, b, resolution)


def _weekly_consecutive(current: str, following: str) -> bool:
    """Accept BOTH weekly conventions the ecosystem uses.

    The DSL emits flat-52 labels (W52 rolls straight to next year's W01, no
    W53), while CHAP also accepts ISO weeks (W52 -> W53 -> W01 in 53-week
    years). So a step is consecutive if it advances the week by one within the
    year, OR rolls from the last week of one year (W52 or W53) to W01/S01 of
    the next.
    """
    cy, cw = int(current[:4]), int(current[6:8])
    fy, fw = int(following[:4]), int(following[6:8])
    if fy == cy and fw == cw + 1:
        return True  # same year, next week
    if fy == cy + 1 and fw == 1 and cw in (52, 53):
        return True  # year rollover (flat-52 from W52, or ISO from W52/W53)
    return False


def _period_start_date(label: str, resolution: str) -> "datetime.date":
    """The calendar start date of a period label, for any resolution."""
    if resolution == "daily":
        return datetime.datetime.strptime(label, "%Y%m%d").date()
    if resolution == "monthly":
        year, month = int(label[:4]), int(label[5:7])
        return datetime.date(year, month, 1)
    if resolution == "yearly":
        return datetime.date(int(label), 1, 1)
    # weekly: YYYY-Wnn (ISO, Monday) or YYYY-Snn (Sunday-start).
    year, week = int(label[:4]), int(label[6:8])
    iso_day = 7 if label[5] == "S" else 1
    # %G-%V-%u is ISO year-week-weekday; gives that week's Monday/Sunday.
    return datetime.datetime.strptime(
        f"{year}-{week:02d}-{iso_day}", "%G-%V-%u"
    ).date()


def _is_one_step(a: "datetime.date", b: "datetime.date", resolution: str) -> bool:
    """True if ``b`` is exactly one period after ``a``."""
    if resolution == "daily":
        return (b - a).days == 1
    if resolution == "weekly":
        return (b - a).days == 7
    if resolution == "monthly":
        # One month later, accounting for year rollover.
        months = (b.year - a.year) * 12 + (b.month - a.month)
        return months == 1 and b.day == 1 and a.day == 1
    return b.year - a.year == 1  # yearly


def _check_values(df: pd.DataFrame) -> list[str]:
    """NaN/type/value rules for the data columns."""
    findings: list[str] = []

    # Covariates are every column that isn't an identifier or the target.
    covariates = [
        c for c in df.columns if c not in ("time_period", "location", "disease_cases")
    ]
    for column in covariates:
        if not pd.api.types.is_numeric_dtype(df[column]):
            findings.append(f"covariate '{column}' is not numeric.")
            continue
        if df[column].isna().any():
            findings.append(
                f"covariate '{column}' contains NaN values; CHAP requires "
                f"complete covariates."
            )
        if np.isinf(df[column].to_numpy(dtype=float)).any():
            findings.append(
                f"covariate '{column}' contains non-finite (infinite) values."
            )

    if "disease_cases" in df.columns:
        cases = df["disease_cases"]
        # Never raise: a non-numeric target is a finding, not a crash.
        if not pd.api.types.is_numeric_dtype(cases):
            findings.append("disease_cases is not numeric.")
        elif cases.isna().all():
            findings.append("disease_cases contains no values at all (all NaN).")
        elif (cases.dropna() < 0).any():
            findings.append("disease_cases contains negative values.")

    return findings
