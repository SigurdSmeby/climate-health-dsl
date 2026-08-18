"""Checks the finished DataFrame against CHAP's dataset rules.

Catches datasets that are valid but not usable by CHAP, before the files are
written. Rules verified against the chap-core source:

- Required columns: ``time_period``, ``location``, ``disease_cases``.
  ``population`` is optional and covariates may have any name.
- ``time_period`` in any resolution CHAP's ``TimePeriod.parse`` accepts.
- Periods consecutive and identical across locations (advisory — CHAP can
  auto-fill, but a mismatch often signals a mistake).
- No NaN in covariate columns (NaN in ``disease_cases`` is fine: CHAP masks
  missing case counts itself).

Returns human-readable findings and never raises; the CLI prints them as
warnings.
"""
import datetime
import itertools
import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("time_period", "location", "disease_cases")

# Label formats CHAP accepts, by resolution. Weekly covers Monday-start (-W),
# Sunday-start (-S), and the start/end date-range form.
_PERIOD_FORMATS = {
    "monthly": re.compile(r"^\d{4}-(0[1-9]|1[0-2])$"),
    "weekly": re.compile(r"^\d{4}-[WS](0[1-9]|[1-4]\d|5[0-3])$"),
    "weekly_range": re.compile(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$"),
    "daily": re.compile(r"^\d{8}$"),
    "yearly": re.compile(r"^\d{4}$"),
}


def validate_chap(df: pd.DataFrame) -> list[str]:
    """Return findings for everything CHAP would reject; empty means clean."""
    findings: list[str] = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            findings.append(f"CHAP requires a '{column}' column, which is missing.")

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

    if groups.nunique() > 1:
        findings.append(
            "locations do not share the same set of time periods (CHAP can "
            "auto-fill, but this is often a mistake)."
        )

    # The date-range week form is skipped — its span is self-describing.
    if resolution == "weekly_range":
        return findings
    for location, sequence in groups.items():
        for current, following in itertools.pairwise(sequence):
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

    The DSL emits flat-52 labels (W52 rolls straight to W01); CHAP also
    accepts ISO weeks (W53 in 53-week years). A step is consecutive if it
    advances the week by one within the year, or rolls from W52/W53 to
    W01 of the next.
    """
    cy, cw = int(current[:4]), int(current[6:8])
    fy, fw = int(following[:4]), int(following[6:8])
    return (fy == cy and fw == cw + 1) or (
        fy == cy + 1 and fw == 1 and cw in (52, 53)
    )


def _period_start_date(label: str, resolution: str) -> "datetime.date":
    """The calendar start date of a non-weekly period label."""
    if resolution == "daily":
        # Date-only label, no timezone semantics involved.
        return datetime.datetime.strptime(label, "%Y%m%d").date()  # noqa: DTZ007
    if resolution == "monthly":
        return datetime.date(int(label[:4]), int(label[5:7]), 1)
    return datetime.date(int(label), 1, 1)  # yearly


def _is_one_step(a: "datetime.date", b: "datetime.date", resolution: str) -> bool:
    """True if ``b`` is exactly one non-weekly period after ``a``."""
    if resolution == "daily":
        return (b - a).days == 1
    if resolution == "monthly":
        months = (b.year - a.year) * 12 + (b.month - a.month)
        return months == 1 and b.day == 1 and a.day == 1
    return b.year - a.year == 1  # yearly


def _check_values(df: pd.DataFrame) -> list[str]:
    """NaN/type/value rules for the data columns."""
    findings: list[str] = []

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
        if not pd.api.types.is_numeric_dtype(cases):
            findings.append("disease_cases is not numeric.")
        elif cases.isna().all():
            findings.append("disease_cases contains no values at all (all NaN).")
        elif (cases.dropna() < 0).any():
            findings.append("disease_cases contains negative values.")

    return findings
