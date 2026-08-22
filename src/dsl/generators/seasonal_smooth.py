"""A smooth seasonal sine wave — used for variables like temperature.

One full cycle per year, whatever the period resolution: 52 points per
cycle in weekly data, 12 in monthly data, and so on.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year


@register_generator("seasonal_smooth")  # this string is what you write in YAML
class SeasonalSmoothGenerator(VariableGenerator):
    """A yearly sine wave around a mean, plus optional noise.

    Registered as "seasonal_smooth" in the generator registry. generate()
    returns one full sine cycle per year around mean, plus noise, floored
    at clamp_min if set.
    Example: array([15.0, 19.3, 22.7, 24.9, ...]) for mean=15, amplitude=10.
    """

    def __init__(
        self,
        mean: float = 15.0,
        amplitude: float = 10.0,
        phase: float = 0.0,
        noise: float = 0.5,
        clamp_min: float | None = None,
    ):
        """Store the YAML params: for this variable.

        Args:
            phase: Phase offset in radians — shifts where in the year the
                peak falls.

        Errors Caught (raised to caller):
            ValueError: If amplitude < 0 or noise < 0.
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
        """Generate the smooth seasonal series.

        Args:
            n_periods: Number of time periods.
            period: Period type (e.g., "monthly", "daily") — sets how many
                periods make up one full sine cycle.
            rng: Seeded random generator for reproducibility.

        Returns:
            A numpy array of length n_periods, holding one full sine cycle
            per year around self.mean, plus optional noise, floored at
            clamp_min if set.
            Example: array([15.0, 19.3, 22.7, 24.9, 24.3, 21.2, ...]) for
            mean=15, amplitude=10.
        """
        ppy = periods_per_year(period)  # 52 for weekly, 12 for monthly, ...
        t = np.arange(n_periods)
        # One full sine cycle per year, scaled to the period resolution.
        series = self.mean + self.amplitude * np.sin(
            2 * np.pi * t / ppy + self.phase
        )
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
