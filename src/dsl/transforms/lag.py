"""Causal lag: delay a series by n periods so its effect appears later.

This is what makes "disease follows rainfall by 3 weeks" expressible. The
shift is CAUSAL: the first n positions have no valid past and become NaN
(the warm-up the disease model blanks). We deliberately do NOT use
``np.roll``: roll is circular, wrapping the end of the series onto the
start, which leaks future values into the past — wrong for a forecasting
benchmark. (The old reference code did exactly that; this is a fix.)
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("lag")  # this string is what you write in YAML
class LagTransform(Transform):
    """Shift a series forward by ``n`` periods, NaN-filling the warm-up."""

    def __init__(self, n: int = 0):
        """``n`` is the delay in periods; 0 means no shift. Must be >= 0."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.n = n

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a lagged copy of ``series``. ``rng`` is unused (no randomness)."""
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
