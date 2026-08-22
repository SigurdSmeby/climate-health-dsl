"""Missing-value injection: blank a random fraction of a series to NaN.

Real surveillance data has gaps; this transform simulates them in a
controlled, reproducible way (the mask comes from the seeded rng).
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("missing")  # this string is what you write in YAML
class MissingTransform(Transform):
    """Replace a rate fraction of entries with NaN.

    Registered as "missing" in the transform registry. apply() blanks each
    entry independently with probability rate (a reproducible random mask
    from the seeded rng), not exactly rate*len(series) entries.
    """

    def __init__(self, rate: float = 0.0):
        """Store the YAML params: for this transform. rate is the expected
        fraction of entries blanked, in [0, 1].

        Errors Caught (raised to caller):
            ValueError: If rate is not in [0, 1].
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Blank ~rate of the series' entries to NaN.

        Args:
            series: The input time series.
            rng: Seeded random generator, used to draw the missing mask.

        Returns:
            A copy of series with each entry independently set to NaN with
            probability self.rate. Unchanged if rate is 0.
        """
        result = series.astype(float)  # copy + make the array NaN-capable
        if self.rate == 0.0:
            return result
        # One uniform draw per entry; entries whose draw falls below `rate`
        # go missing. Each entry is hit independently with probability rate.
        mask = rng.random(len(series)) < self.rate
        result[mask] = np.nan
        return result
