"""A smooth seasonal sine wave — used for variables like temperature.

One full cycle per year, whatever the period resolution: 52 points per
cycle in weekly data, 12 in monthly data, and so on.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year


@register_generator("seasonal_smooth")  # this string is what you write in YAML
class SeasonalSmoothGenerator(VariableGenerator):
    """A yearly sine wave around a mean, plus optional noise."""

    def __init__(
        self,
        mean: float = 15.0,
        amplitude: float = 10.0,
        phase: float = 0.0,
        noise: float = 0.5,
        clamp_min: float | None = None,
    ):
        """Store and validate the YAML ``params:`` for this variable.

        Parameters
        ----------
        mean:
            The value the wave oscillates around.
        amplitude:
            How far above/below the mean the wave swings. Must be >= 0.
        phase:
            Phase offset in radians — shifts where in the year the peak falls.
        noise:
            Standard deviation of additive Gaussian noise (0 disables it).
        clamp_min:
            If set, values are floored at this minimum (e.g. 0 for
            quantities that cannot be negative).
        """
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
        """Return the smooth seasonal series, length ``n_periods``."""
        ppy = periods_per_year(period)  # 52 for weekly, 12 for monthly, ...
        t = np.arange(n_periods)  # the time axis: 0, 1, 2, ...
        # One full sine cycle per year, scaled to the period resolution.
        series = self.mean + self.amplitude * np.sin(
            2 * np.pi * t / ppy + self.phase
        )
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            # np.maximum floors every value at clamp_min (element-wise max).
            series = np.maximum(series, self.clamp_min)
        return series
