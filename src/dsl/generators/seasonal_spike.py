"""A variable with a low baseline and a pronounced seasonal spike.

Models a "rainy season" shape: flat most of the year, with a smooth
Gaussian-shaped bump that peaks at a configurable point in the yearly cycle
and repeats every year. Used for variables like rainfall.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year


@register_generator("seasonal_spike")  # this string is what you write in YAML
class SeasonalSpikeGenerator(VariableGenerator):
    """Low baseline plus a yearly Gaussian-shaped spike, plus optional noise.

    Registered as "seasonal_spike" in the generator registry. generate()
    returns baseline plus a Gaussian bump around spike_center, repeating
    every year, plus noise, floored at clamp_min if set.
    Example: array([2.1, 2.4, 8.7, 20.3, 19.8, 7.2, 2.3, ...]) for
    baseline=2, spike_height=20.
    """

    def __init__(
        self,
        baseline: float = 2.0,
        spike_height: float = 20.0,
        spike_center: int | None = None,
        spike_width: float = 4.0,
        noise: float = 0.5,
        clamp_min: float | None = None,
    ):
        """Store the YAML params: for this variable.

        Args:
            baseline: The value far from the spike (the dry-season level).
            spike_height: How far above baseline the peak rises.
            spike_center: The period offset of the peak within the yearly
                cycle (e.g. 26 for mid-year in weekly data, 6 for July in
                monthly data). Values beyond one cycle wrap (24 on monthly
                data == month 0). The default (None) is mid-year at any
                resolution — 26 for weekly, 6 for monthly — so a monthly
                series peaks correctly without tuning.
            spike_width: The standard deviation of the Gaussian bump, in
                periods. Bigger means a broader rainy season.
            noise: Standard deviation of additive Gaussian noise.
            clamp_min: If set, floors the result at this minimum — e.g. 0
                for rainfall, which can't be negative.

        Errors Caught (raised to caller):
            ValueError: If spike_width <= 0 or noise < 0.
        """
        if spike_width <= 0:
            raise ValueError(f"spike_width must be > 0, got {spike_width}")
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.baseline = baseline
        self.spike_height = spike_height
        self.spike_center = spike_center
        self.spike_width = spike_width
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate the spiky seasonal series.

        Args:
            n_periods: Number of time periods.
            period: Period type (e.g., "monthly", "daily") — sets how many
                periods make up one yearly cycle.
            rng: Seeded random generator for reproducibility.

        Returns:
            A numpy array of length n_periods, holding baseline plus a
            Gaussian bump around spike_center (repeating every year), plus
            optional noise, floored at clamp_min if set.
            Example: array([2.1, 2.4, 8.7, 20.3, 19.8, 7.2, 2.3, ...]) for
            baseline=2, spike_height=20.
        """
        ppy = periods_per_year(period)  # 52 for weekly, 12 for monthly, ...
        t = np.arange(n_periods)
        # Position within the current year, so the spike repeats annually.
        pos = t % ppy
        # The peak's position within the year. None → mid-year (works at any
        # resolution); an explicit value beyond one cycle wraps via % ppy.
        center = (ppy // 2 if self.spike_center is None else self.spike_center) % ppy
        # Circular distance to the spike center: week 51 is only 2 weeks
        # away from a week-1 peak, not 50, because the seasons wrap around.
        raw = np.abs(pos - center)
        dist = np.minimum(raw, ppy - raw)
        # A Gaussian bump: exp(-d²/2σ²) is 1 at the center, ~0 far away.
        spike = self.spike_height * np.exp(-0.5 * (dist / self.spike_width) ** 2)
        series = self.baseline + spike
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
