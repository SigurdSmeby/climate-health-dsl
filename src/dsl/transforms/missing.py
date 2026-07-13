"""Missing-value injection: blank a random fraction of a series to NaN."""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("missing")
class MissingTransform(Transform):
    """Replace a ``rate`` fraction of entries with NaN, each hit independently."""

    def __init__(self, rate: float = 0.0):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = series.astype(float)  # copy + NaN-capable
        if self.rate == 0.0:
            return result
        mask = rng.random(len(series)) < self.rate
        result[mask] = np.nan
        return result
