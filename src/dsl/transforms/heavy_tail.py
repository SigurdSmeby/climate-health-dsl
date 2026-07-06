"""Heavy-tailed noise: add Student-t noise so a series has fat tails / outliers.

Gaussian noise (what the generators add) rarely produces extreme values. Real
surveillance data does — a reporting error, a genuine spike. Student-t noise
with low degrees of freedom has much heavier tails, so this lets a scenario
stress a model's robustness to outliers as known ground truth. As df grows, the
t-distribution approaches a Gaussian.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("heavy_tail")  # this string is what you write in YAML
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
        """Return a copy with additive Student-t noise. NaN preserved (adding
        to NaN stays NaN, so a missing input stays missing)."""
        result = series.astype(float)  # copy; input untouched
        if self.scale == 0.0:
            return result
        return result + self.scale * rng.standard_t(self.df, size=len(result))
