"""Time-axis helpers: how many periods make a year, and CHAP period labels.

Used by generators (to scale seasonality to the resolution), by the schema
(to warn when a scenario is shorter than one seasonal cycle), and by output
(to label rows).
"""
import datetime

# One seasonal cycle per resolution. A plain dict keeps the mapping explicit
# and easy to extend.
_PERIODS_PER_YEAR: dict[str, int] = {
    "daily": 365,
    "weekly": 52,
    "monthly": 12,
    "yearly": 1,
}


def periods_per_year(period: str) -> int:
    """Return how many periods of the given resolution make up one year.

    Parameters
    ----------
    period:
        One of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"yearly"``.

    Raises
    ------
    KeyError
        If the period name is unknown (the schema normally catches this
        first, so hitting it here indicates a programming error).
    """
    if period not in _PERIODS_PER_YEAR:
        raise KeyError(
            f"Unknown period '{period}'. Expected one of "
            f"{sorted(_PERIODS_PER_YEAR)}."
        )
    return _PERIODS_PER_YEAR[period]


def format_period(index: int, period: str, start_year: int = 2000) -> str:
    """Turn a row index into a CHAP-compatible period string.

    The formats match CHAP's conventions (verified against ``chap_core``):

    - daily   → ``20000101`` (compact YYYYMMDD, real calendar dates)
    - weekly  → ``2000-W01`` (52 weeks per year, zero-padded, rolls over)
    - monthly → ``2000-01``  (12 months per year, rolls over)
    - yearly  → ``2000``

    Parameters
    ----------
    index:
        Zero-based row index: 0 is the first period of ``start_year``.
    period:
        One of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"yearly"``.
    start_year:
        The calendar year that index 0 falls in (default 2000).

    Returns
    -------
    str
        The CHAP period label for that row.
    """
    if period == "daily":
        # Use a real calendar so month lengths and leap years are correct
        # (e.g. index 366 from a 2000 start is 2001-01-01, not 2000-12-32).
        date = datetime.date(start_year, 1, 1) + datetime.timedelta(days=index)
        return date.strftime("%Y%m%d")  # strftime formats a date as a string

    if period == "weekly":
        # divmod returns (quotient, remainder) in one step: how many whole
        # years have passed, and which week within the current year.
        years, week = divmod(index, 52)
        # :02d pads to two digits, so week 1 prints as "W01" not "W1".
        return f"{start_year + years}-W{week + 1:02d}"

    if period == "monthly":
        years, month = divmod(index, 12)
        return f"{start_year + years}-{month + 1:02d}"

    if period == "yearly":
        return str(start_year + index)

    raise KeyError(
        f"Unknown period '{period}'. Expected one of {sorted(_PERIODS_PER_YEAR)}."
    )
