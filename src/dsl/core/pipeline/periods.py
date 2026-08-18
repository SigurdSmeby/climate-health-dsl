"""Time-axis helpers: periods per year, and CHAP period labels.

Used by generators (to scale seasonality), the schema (to validate
start_period), and the engine (to label rows).
"""
import datetime
import re

_PERIODS_PER_YEAR: dict[str, int] = {
    "daily": 365,
    "weekly": 52,
    "monthly": 12,
    "yearly": 1,
}

# What a valid label looks like for each resolution.
_LABEL_PATTERNS = {
    "daily": re.compile(r"^\d{4}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$"),
    "weekly": re.compile(r"^\d{4}-W(0[1-9]|[1-4]\d|5[0-2])$"),
    "monthly": re.compile(r"^\d{4}-(0[1-9]|1[0-2])$"),
    "yearly": re.compile(r"^\d{4}$"),
}


def periods_per_year(period: str) -> int:
    """How many periods of the given resolution make up one year."""
    if period not in _PERIODS_PER_YEAR:
        raise KeyError(
            f"Unknown period '{period}'. Expected one of "
            f"{sorted(_PERIODS_PER_YEAR)}."
        )
    return _PERIODS_PER_YEAR[period]


def format_period(index: int, period: str, start_year: int = 2000) -> str:
    """Turn a row index into a CHAP-compatible period label.

    Index 0 is the first period of ``start_year``. Formats (verified against
    chap_core): daily ``20000101``, weekly ``2000-W01`` (flat 52 weeks/year),
    monthly ``2000-01``, yearly ``2000``.
    """
    if period == "daily":
        # Real calendar dates, so month lengths and leap years are correct.
        date = datetime.date(start_year, 1, 1) + datetime.timedelta(days=index)
        return date.strftime("%Y%m%d")

    if period == "weekly":
        years, week = divmod(index, 52)
        return f"{start_year + years}-W{week + 1:02d}"

    if period == "monthly":
        years, month = divmod(index, 12)
        return f"{start_year + years}-{month + 1:02d}"

    if period == "yearly":
        return str(start_year + index)

    raise KeyError(
        f"Unknown period '{period}'. Expected one of {sorted(_PERIODS_PER_YEAR)}."
    )


def parse_period(label: str, period: str) -> tuple[int, int]:
    """The inverse of ``format_period``: a label → (year, offset within year).

    E.g. ``parse_period("2010-07", "monthly")`` is ``(2010, 6)``. Raises
    ValueError if the label doesn't match the resolution's format.
    """
    if period not in _LABEL_PATTERNS:
        raise KeyError(
            f"Unknown period '{period}'. Expected one of {sorted(_LABEL_PATTERNS)}."
        )
    if not _LABEL_PATTERNS[period].match(label):
        examples = {
            "daily": "20100615",
            "weekly": "2015-W10",
            "monthly": "2010-07",
            "yearly": "2003",
        }
        # The date-range weekly form (YYYY-MM-DD/...) is readable from CSVs
        # but is not a usable start_period; point users at YYYY-Wnn.
        hint = ""
        if period == "weekly" and "/" in label:
            hint = " The date-range week form is read from CSVs but is not a "
            hint += "usable start_period; use the YYYY-Wnn form instead."
        raise ValueError(
            f"'{label}' is not a valid {period} period label "
            f"(expected something like '{examples[period]}').{hint}"
        )

    year = int(label[:4])
    if period == "daily":
        date = datetime.date(year, int(label[4:6]), int(label[6:8]))
        return year, (date - datetime.date(year, 1, 1)).days  # Jan 1 → 0
    if period == "weekly":
        return year, int(label[6:8]) - 1
    if period == "monthly":
        return year, int(label[5:7]) - 1
    return year, 0  # yearly
