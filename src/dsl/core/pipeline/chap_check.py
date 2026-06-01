"""Checks the finished DataFrame against CHAP's documented dataset rules.

The DSL can generate datasets that are perfectly valid but not usable by
CHAP (e.g. a variable named ``wind`` instead of the covariates standard
CHAP models expect). This module catches that *before* the files are
written, per the rules in CHAP's data-preparation docs:

- required columns: ``time_period``, ``location``, ``disease_cases``;
- ``time_period`` in ``YYYY-MM`` (monthly) or ``YYYY-Wnn`` (weekly) format —
  the only two formats CHAP documents;
- periods consecutive, and identical across locations;
- no NaN in covariate columns (NaN in ``disease_cases`` is fine: CHAP
  masks missing case counts itself).

Like ``validate_scenario``, this returns human-readable findings and never
raises; the CLI prints them as warnings, or refuses to write with
``--strict-chap``.
"""
import re

import pandas as pd

# The columns CHAP requires, and the covariates its standard models expect.
REQUIRED_COLUMNS = ("time_period", "location", "disease_cases")
STANDARD_COVARIATES = ("rainfall", "mean_temperature", "population")

# CHAP's two documented period formats.
_MONTHLY = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_WEEKLY = re.compile(r"^\d{4}-W(0[1-9]|[1-4]\d|5[0-2])$")


def validate_chap(df: pd.DataFrame) -> list[str]:
    """Return findings for everything CHAP would reject. Never raises.

    An empty list means the DataFrame follows every documented CHAP rule.
    """
    findings: list[str] = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            findings.append(f"CHAP requires a '{column}' column, which is missing.")
    for column in STANDARD_COVARIATES:
        if column not in df.columns:
            findings.append(
                f"standard CHAP models expect a '{column}' column, which is missing."
            )

    # The remaining checks read the columns, so they need them present.
    if "time_period" in df.columns:
        findings.extend(_check_periods(df))
    findings.extend(_check_values(df))
    return findings


def _check_periods(df: pd.DataFrame) -> list[str]:
    """Period format, consecutiveness, and equality across locations."""
    findings: list[str] = []
    periods = df["time_period"].astype(str)

    # All labels must share one of the two documented formats.
    if periods.str.match(_MONTHLY).all():
        step = _next_month
    elif periods.str.match(_WEEKLY).all():
        step = _next_week
    else:
        findings.append(
            "time_period values are not in a CHAP-documented format "
            "(YYYY-MM monthly or YYYY-Wnn weekly)."
        )
        return findings  # format unknown → can't check order either

    groups = (
        df.groupby("location", sort=False)["time_period"].apply(tuple)
        if "location" in df.columns
        else pd.Series({"all": tuple(periods)})
    )

    # Every location must cover the same periods.
    if groups.nunique() > 1:
        findings.append(
            "locations do not share the same set of time periods; CHAP "
            "requires every location to cover identical periods."
        )

    # Periods must follow each other without gaps.
    for location, sequence in groups.items():
        for current, following in zip(sequence, sequence[1:]):
            if following != step(current):
                findings.append(
                    f"time periods for location '{location}' are not "
                    f"consecutive: '{following}' follows '{current}'."
                )
                break  # one finding per location is enough

    return findings


def _next_month(period: str) -> str:
    """'2000-12' -> '2001-01' (the label one month later)."""
    year, month = int(period[:4]), int(period[5:7])
    if month == 12:
        return f"{year + 1}-01"
    return f"{year}-{month + 1:02d}"


def _next_week(period: str) -> str:
    """'2000-W52' -> '2001-W01' (the label one week later, 52-week years)."""
    year, week = int(period[:4]), int(period[6:8])
    if week == 52:
        return f"{year + 1}-W01"
    return f"{year}-W{week + 1:02d}"


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
        elif df[column].isna().any():
            findings.append(
                f"covariate '{column}' contains NaN values; CHAP requires "
                f"complete covariates."
            )

    if "disease_cases" in df.columns:
        cases = df["disease_cases"]
        if cases.isna().all():
            findings.append("disease_cases contains no values at all (all NaN).")
        elif (cases.dropna() < 0).any():
            findings.append("disease_cases contains negative values.")

    return findings
