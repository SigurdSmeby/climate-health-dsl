"""A variable with a low baseline and a pronounced seasonal spike.

A "rainy season" shape: flat most of the year, with a Gaussian-shaped bump
that peaks at a configurable point in the yearly cycle and repeats annually.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year


@register_generator("seasonal_spike")
class SeasonalSpikeGenerator(VariableGenerator):
    """Low baseline plus a yearly Gaussian-shaped spike, plus optional noise.

    Params: ``baseline`` (dry-season level), ``spike_height`` (peak above
    baseline), ``spike_center`` (period offset of the peak within the yearly
    cycle, e.g. 6 for July in monthly data; wraps past one cycle; None →
    mid-year at any resolution), ``spike_width`` (Gaussian std in periods),
    ``noise`` (Gaussian std), ``clamp_min`` (optional floor, e.g. 0 for rain).
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
        ppy = periods_per_year(period)
        pos = np.arange(n_periods) % ppy  # position within the current year
        center = (ppy // 2 if self.spike_center is None else self.spike_center) % ppy
        # Circular distance to the peak: week 51 is 2 weeks from a week-1
        # peak, not 50 — seasons wrap around the year boundary.
        raw = np.abs(pos - center)
        dist = np.minimum(raw, ppy - raw)
        spike = self.spike_height * np.exp(-0.5 * (dist / self.spike_width) ** 2)
        series = self.baseline + spike
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
