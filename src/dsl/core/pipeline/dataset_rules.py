"""Checks the finished DataFrame against the rules a forecasting tool expects.

Catches datasets that are internally valid but awkward or unusable downstream,
before the files are written. The rules are the ones long-format epidemiological
data generally has to satisfy; they were checked against a real consumer
(chap-core) so they are not invented:

- Required columns: ``time_period``, ``location``, ``disease_cases``.
  ``population`` is optional and driver columns may have any name.
- ``time_period`` in a standard, parseable label format for its resolution.
- Periods consecutive and identical across locations (advisory — a consumer may
  auto-fill, but a mismatch often signals a mistake).
- No NaN in driver columns (NaN in the case counts is fine: a missing case
  count is ordinarily treated as masked rather than as zero).

Returns human-readable findings and never raises; the CLI prints them as
warnings.
"""
import datetime
import itertools
import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("time_period", "location", "disease_cases")

# Accepted label formats, by resolution. Weekly covers Monday-start (-W),
# Sunday-start (-S), and the start/end date-range form.
_PERIOD_FORMATS = {
    "monthly": re.compile(r"^\d{4}-(0[1-9]|1[0-2])$"),
    "weekly": re.compile(r"^\d{4}-[WS](0[1-9]|[1-4]\d|5[0-3])$"),
    "weekly_range": re.compile(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$"),
    "daily": re.compile(r"^\d{8}$"),
    "yearly": re.compile(r"^\d{4}$"),
}


def check_dataset(df: pd.DataFrame) -> list[str]:
    """Check the DataFrame against the dataset rules.

    Validates: required columns present, time_period in a parseable format
    and consecutive/aligned across locations, driver columns numeric with no
    NaN/Inf, disease_cases non-negative. Never raises — the CLI prints the
    findings as warnings.

    Args:
        df: The output DataFrame.

    Returns:
        A list of findings (empty if all checks pass).
        Each finding is a human-readable string.
    """
    findings: list[str] = []

    # Step 1: Check required columns.
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            findings.append(f"the '{column}' column is required and missing.")

    # Step 2: Check time_period format/consecutiveness and data values.
    if "time_period" in df.columns:
        findings.extend(_check_periods(df))
    findings.extend(_check_values(df))
    return findings


def _check_periods(df: pd.DataFrame) -> list[str]:
    """Check time_period format, consecutiveness, and equality across locations.

    Args:
        df: The output DataFrame (must have a time_period column).

    Returns:
        A list of human-readable findings (empty if the periods are clean).
    """
    findings: list[str] = []
    periods = df["time_period"].astype(str)

    resolution = _detect_resolution(periods)
    if resolution is None:
        findings.append(
            "time_period values are not in a parseable format "
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
            "locations do not share the same set of time periods (a consumer can "
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


def _detect_resolution(periods: pd.Series) -> str | None:
    """Detect the period resolution all labels share.

    Args:
        periods: The time_period column, as strings.

    Returns:
        The resolution name (e.g. "monthly", "weekly") whose format all
        labels match, or None if no single resolution fits every label.
    """
    for resolution, pattern in _PERIOD_FORMATS.items():
        if periods.str.match(pattern).all():
            return resolution
    return None


def _consecutive(current: str, following: str, resolution: str) -> bool:
    """Is ``following`` exactly one period after ``current``?

    Args:
        current: The earlier time_period label.
        following: The later time_period label.
        resolution: The period resolution both labels share.

    Returns:
        True if following comes immediately after current at this resolution.
    """
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

    The DSL emits flat-52 labels (W52 rolls straight to W01); consumers also
    accepts ISO weeks (W53 in 53-week years). A step is consecutive if it
    advances the week by one within the year, or rolls from W52/W53 to
    W01 of the next.

    Args:
        current: The earlier weekly time_period label (e.g. "2020-W12").
        following: The later weekly time_period label.

    Returns:
        True if following comes immediately after current, in either
        convention.
    """
    cy, cw = int(current[:4]), int(current[6:8])
    fy, fw = int(following[:4]), int(following[6:8])
    return (fy == cy and fw == cw + 1) or (
        fy == cy + 1 and fw == 1 and cw in (52, 53)
    )


def _period_start_date(label: str, resolution: str) -> "datetime.date":
    """The calendar start date of a non-weekly period label.

    Args:
        label: A time_period label (daily YYYYMMDD, monthly YYYY-MM, or
            yearly YYYY).
        resolution: "daily", "monthly", or "yearly".

    Returns:
        The date the period starts on.

    Errors Caught (raised to caller):
        ValueError: If label doesn't match resolution's expected format.
    """
    if resolution == "daily":
        # Date-only label, no timezone semantics involved.
        return datetime.datetime.strptime(label, "%Y%m%d").date()  # noqa: DTZ007
    if resolution == "monthly":
        return datetime.date(int(label[:4]), int(label[5:7]), 1)
    return datetime.date(int(label), 1, 1)  # yearly


def _is_one_step(a: "datetime.date", b: "datetime.date", resolution: str) -> bool:
    """Is ``b`` exactly one non-weekly period after ``a``?

    Args:
        a: The earlier period's start date.
        b: The later period's start date.
        resolution: "daily", "monthly", or "yearly".

    Returns:
        True if b comes immediately after a at this resolution.
    """
    if resolution == "daily":
        return (b - a).days == 1
    if resolution == "monthly":
        months = (b.year - a.year) * 12 + (b.month - a.month)
        return months == 1 and b.day == 1 and a.day == 1
    return b.year - a.year == 1  # yearly


def _has_gap_after_warmup(df: pd.DataFrame, column: str) -> bool:
    """Report whether a covariate is missing values anywhere but its start.

    A series built from a lagged parent has no input for its first few
    periods, so a LEADING run of NaN is the declared warm-up rather than a
    defect. A gap further in is real missing data, which a consumer cannot use.

    Args:
        df: The output DataFrame.
        column: The covariate column to check.

    Returns:
        True if any NaN appears after the column's first non-NaN value, in
        any location. False for a clean column, or one whose only NaN is the
        opening warm-up.
    """
    for _, group in df.groupby("location", sort=False):
        missing = group[column].isna().to_numpy()
        if not missing.any() or missing.all():
            # All-NaN is caught as an empty column elsewhere; skip it here.
            continue
        first_value = int(np.argmin(missing))
        if missing[first_value:].any():
            return True
    return False


def _check_values(df: pd.DataFrame) -> list[str]:
    """Check NaN/type/value rules for the covariate and disease_cases columns.

    Args:
        df: The output DataFrame.

    Returns:
        A list of human-readable findings (empty if the values are clean).
    """
    findings: list[str] = []

    # Step 1: Covariates must be numeric, with no NaN or Inf.
    covariates = [
        c for c in df.columns if c not in ("time_period", "location", "disease_cases")
    ]
    for column in covariates:
        if not pd.api.types.is_numeric_dtype(df[column]):
            findings.append(f"covariate '{column}' is not numeric.")
            continue
        if _has_gap_after_warmup(df, column):
            findings.append(
                f"covariate '{column}' contains NaN values after its opening "
                f"periods; driver columns must be complete."
            )
        if np.isinf(df[column].to_numpy(dtype=float)).any():
            findings.append(
                f"covariate '{column}' contains non-finite (infinite) values."
            )

    # Step 2: disease_cases must be numeric and non-negative (NaN is fine —
    # a missing case count is ordinarily treated as masked).
    if "disease_cases" in df.columns:
        cases = df["disease_cases"]
        if not pd.api.types.is_numeric_dtype(cases):
            findings.append("disease_cases is not numeric.")
        elif cases.isna().all():
            findings.append("disease_cases contains no values at all (all NaN).")
        elif (cases.dropna() < 0).any():
            findings.append("disease_cases contains negative values.")

    return findings
