"""A linear trend — a value that rises or falls steadily over time.

Models slow drift real series carry (population growth, gradual warming);
useful as a non-seasonal confounder when testing whether a model separates
trend from signal.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator


@register_generator("linear_trend")
class LinearTrendGenerator(VariableGenerator):
    """A straight line ``start + slope * t`` with optional Gaussian noise.

    Params: ``start`` (value at period 0), ``slope`` (change per period),
    ``noise`` (Gaussian std), ``clamp_min`` (optional floor).
    """

    def __init__(
        self,
        start: float = 0.0,
        slope: float = 1.0,
        noise: float = 0.0,
        clamp_min: float | None = None,
    ):
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.start = start
        self.slope = slope
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        t = np.arange(n_periods)
        series = self.start + self.slope * t
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
