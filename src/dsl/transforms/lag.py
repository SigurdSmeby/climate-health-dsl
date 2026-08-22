"""Causal lag: delay a series by n periods so its effect appears later.

This is what makes "disease follows rainfall by 3 weeks" expressible. The
shift is CAUSAL: the first n positions have no valid past and become NaN
(the warm-up the disease model blanks). It deliberately does NOT use
``np.roll``, which is circular — it wraps the end of the series onto the
start, leaking future values into the past, wrong for a forecasting benchmark.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("lag")  # this string is what you write in YAML
class LagTransform(Transform):
    """Shift a series forward by n periods, NaN-filling the warm-up.

    Registered as "lag" in the transform registry. apply() returns the
    series shifted so value at time t becomes the input's value at t - n;
    the first n positions have no valid past and become NaN.
    Example: array([nan, nan, 50.5, 59.3, ...]) for n=2.
    """

    def __init__(self, n: int = 0):
        """Store the YAML params: for this transform. n=0 means no shift.

        Errors Caught (raised to caller):
            ValueError: If n < 0.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.n = n

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Shift the series forward by n periods.

        Args:
            series: The input time series.
            rng: Seeded random generator (required by the transform API,
                unused — lag is deterministic).

        Returns:
            A lagged copy of series, same length. The first n positions
            are NaN (no valid past to read from).
            Example: array([nan, nan, 50.5, 59.3, ...]) for n=2.
        """
        # astype(float) both copies (input stays untouched) and guarantees
        # the array can hold NaN — integer arrays cannot.
        result = series.astype(float)
        if self.n == 0:
            return result
        if self.n >= len(series):
            # The whole series is warm-up: no position has a valid past.
            result[:] = np.nan
            return result
        # Value at time t becomes the input's value at time t - n ...
        result[self.n:] = series[: -self.n]
        # ... and the first n positions have no t - n to read from.
        result[: self.n] = np.nan
        return result
