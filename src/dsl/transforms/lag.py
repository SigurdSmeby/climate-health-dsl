"""Causal lag: delay a series by n periods so its effect appears later.

Deliberately NOT np.roll, which is circular and would leak future values
into the past — wrong for a forecasting benchmark. The first n positions
have no valid past and become NaN (the warm-up the disease model blanks).
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("lag")
class LagTransform(Transform):
    """Shift a series forward by ``n`` periods, NaN-filling the warm-up."""

    def __init__(self, n: int = 0):
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.n = n

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = series.astype(float)  # copy + NaN-capable
        if self.n == 0:
            return result
        if self.n >= len(series):
            result[:] = np.nan
            return result
        result[self.n:] = series[: -self.n]
        result[: self.n] = np.nan
        return result
