"""Missing-value injection: blank a random fraction of a series to NaN.

Real surveillance data has gaps; this transform simulates them in a
controlled, reproducible way (the mask comes from the seeded rng).
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("missing")  # this string is what you write in YAML
class MissingTransform(Transform):
    """Replace a ``rate`` fraction of entries with NaN."""

    def __init__(self, rate: float = 0.0):
        """``rate`` is the expected fraction of entries blanked, in [0, 1]."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a copy of ``series`` with ~rate of its entries set to NaN."""
        result = series.astype(float)  # copy + make the array NaN-capable
        if self.rate == 0.0:
            return result
        # One uniform draw per entry; entries whose draw falls below `rate`
        # go missing. Each entry is hit independently with probability rate.
        mask = rng.random(len(series)) < self.rate
        result[mask] = np.nan
        return result
