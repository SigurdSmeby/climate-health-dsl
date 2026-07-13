"""A smooth seasonal sine wave — used for variables like temperature.

One full cycle per year at any resolution: 52 points per cycle in weekly
data, 12 in monthly.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year


@register_generator("seasonal_smooth")
class SeasonalSmoothGenerator(VariableGenerator):
    """A yearly sine wave around a mean, plus optional noise.

    Params: ``mean`` (center of the wave), ``amplitude`` (swing above/below),
    ``phase`` (radians, shifts where the peak falls), ``noise`` (Gaussian std),
    ``clamp_min`` (optional floor).
    """

    def __init__(
        self,
        mean: float = 15.0,
        amplitude: float = 10.0,
        phase: float = 0.0,
        noise: float = 0.5,
        clamp_min: float | None = None,
    ):
        if amplitude < 0:
            raise ValueError(f"amplitude must be >= 0, got {amplitude}")
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.mean = mean
        self.amplitude = amplitude
        self.phase = phase
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        ppy = periods_per_year(period)
        t = np.arange(n_periods)
        series = self.mean + self.amplitude * np.sin(
            2 * np.pi * t / ppy + self.phase
        )
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
