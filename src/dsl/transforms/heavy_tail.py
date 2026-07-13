"""Heavy-tailed noise: add Student-t noise so a series has fat tails/outliers.

Gaussian noise rarely produces extreme values; real surveillance data does.
Low ``df`` means heavy tails; as df grows the t approaches a Gaussian.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("heavy_tail")
class HeavyTailTransform(Transform):
    """Add Student-t noise (scale ``scale``, ``df`` degrees of freedom)."""

    def __init__(self, scale: float = 1.0, df: float = 3.0):
        if scale < 0:
            raise ValueError(f"scale must be >= 0, got {scale}")
        if df <= 0:
            raise ValueError(f"df must be > 0, got {df}")
        self.scale = scale
        self.df = df

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = series.astype(float)  # copy; NaN + noise stays NaN
        if self.scale == 0.0:
            return result
        return result + self.scale * rng.standard_t(self.df, size=len(result))
