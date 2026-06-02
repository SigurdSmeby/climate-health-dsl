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
import re

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
# How to step one period forward, for the resolutions we can check ordering
# on. (Date-range weeks are left unchecked — the format already proves the
# span; ordering there would need real date arithmetic.)
_STEPPERS = {"monthly": "_next_month", "weekly": "_next_week"}


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

    # Consecutiveness is only checked for resolutions we can step.
    stepper_name = _STEPPERS.get(resolution)
    if stepper_name is None:
        return findings
    step = globals()[stepper_name]
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
