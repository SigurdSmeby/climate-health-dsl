"""Rare outbreak shocks: a low baseline punctuated by sudden sharp spikes.

Models extreme events — a heavy-rain week, a heatwave — that trigger an
outbreak, the rare-event case forecasters handle poorly. Distinct from
``seasonal_spike``, which is a smooth bump that repeats every year: here shocks
are rare and their timing is random (Poisson), so no two years look alike.

Use it as a driver whose spikes propagate into disease, or as the disease-like
shape itself.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year


@register_generator("outbreak_shocks")  # this string is what you write in YAML
class OutbreakShocksGenerator(VariableGenerator):
    """A ``baseline`` (+noise) with rare shocks of size ``magnitude``."""

    def __init__(
        self,
        baseline: float = 0.0,
        noise: float = 0.0,
        rate: float = 1.0,
        magnitude: float = 20.0,
        duration: int = 1,
        clamp_min: float | None = None,
    ):
        """
        Parameters
        ----------
        baseline: the quiet-period level the series sits at.
        noise: Gaussian noise std on the baseline (0 → flat between shocks).
        rate: expected number of shock events per year (Poisson mean).
        magnitude: how far above baseline each shock rises.
        duration: how many consecutive periods each shock elevates.
        clamp_min: optional floor (e.g. 0).
        """
        if rate < 0:
            raise ValueError(f"rate must be >= 0, got {rate}")
        if duration < 1:
            raise ValueError(f"duration must be >= 1, got {duration}")
        self.baseline = baseline
        self.noise = noise
        self.rate = rate
        self.magnitude = magnitude
        self.duration = duration
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        series = np.full(n_periods, float(self.baseline))
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)

        if self.rate > 0 and self.magnitude != 0:
            ppy = periods_per_year(period)
            # Expected total events over the whole span, drawn as one Poisson;
            # each event's start is uniform over the timeline. This keeps the
            # per-year rate at `rate` regardless of series length.
            expected = self.rate * n_periods / ppy
            n_events = rng.poisson(expected)
            starts = rng.integers(0, n_periods, size=n_events)
            for s in starts:
                series[s : s + self.duration] += self.magnitude

        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
