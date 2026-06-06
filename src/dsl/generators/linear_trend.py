"""A linear trend — a value that rises or falls steadily over time.

Models slow drift that real series often carry: population growth, gradual
warming, urbanization, changing reporting completeness. Useful as a
confounder (a trend that is NOT seasonal but still correlates with anything
else trending) when testing whether a model separates trend from signal.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator


@register_generator("linear_trend")  # this string is what you write in YAML
class LinearTrendGenerator(VariableGenerator):
    """A straight line ``start + slope * t`` with optional Gaussian noise."""

    def __init__(
        self,
        start: float = 0.0,
        slope: float = 1.0,
        noise: float = 0.0,
        clamp_min: float | None = None,
    ):
        """Store and validate the YAML ``params:`` for this variable.

        Parameters
        ----------
        start:
            The value at period 0.
        slope:
            How much the value changes each period (negative falls).
        noise:
            Standard deviation of additive Gaussian noise (0 → a clean line).
        clamp_min:
            If set, values are floored at this minimum (e.g. 0).
        """
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.start = start
        self.slope = slope
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Return the trend series, length ``n_periods``."""
        t = np.arange(n_periods)  # 0, 1, 2, ...
        series = self.start + self.slope * t
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
