"""Heavy-tailed noise: add Student-t noise so a series has fat tails/outliers.

Gaussian noise rarely produces extreme values; real surveillance data does.
Low ``df`` means heavy tails; as df grows the t approaches a Gaussian.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("heavy_tail")
class HeavyTailTransform(Transform):
    """Add Student-t noise for fat tails/outliers.

    Registered as "heavy_tail" in the transform registry. apply() adds
    scale * Student-t(df) noise to each value; low df means heavy tails,
    and as df grows the distribution approaches Gaussian.
    Example: array([51.2, 48.7, 62.4, 49.9, ...]) for a series around 50
    with scale=1, df=3 (occasional larger jumps than Gaussian noise).
    """

    def __init__(self, scale: float = 1.0, df: float = 3.0):
        """Store the YAML params: for this transform.

        Errors Caught (raised to caller):
            ValueError: If scale isn't a finite number >= 0, or df isn't a
                finite number > 0. np.isfinite also rejects NaN/Inf, unlike
                a plain `scale < 0` check — every comparison against NaN is
                False, so a NaN scale would otherwise silently pass here
                and corrupt the output series in apply() instead.
        """
        if not np.isfinite(scale) or scale < 0:
            raise ValueError(f"scale must be a finite number >= 0, got {scale}")
        if not np.isfinite(df) or df <= 0:
            raise ValueError(f"df must be a finite number > 0, got {df}")
        self.scale = scale
        self.df = df

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add Student-t noise to the series.

        Args:
            series: The input time series.
            rng: Seeded random generator for reproducibility.

        Returns:
            series plus scale * Student-t(df) noise, same length. NaN
            values stay NaN. Unchanged if scale is 0.
        """
        result = series.astype(float)  # copy; NaN + noise stays NaN
        if self.scale == 0.0:
            return result
        return result + self.scale * rng.standard_t(self.df, size=len(result))
