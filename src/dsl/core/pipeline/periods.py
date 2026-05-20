"""Time-axis helpers: how many periods make a year, and CHAP period labels.

Used by generators (to scale seasonality to the resolution), by the schema
(to warn when a scenario is shorter than one seasonal cycle), and by output
(to label rows).
"""

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
