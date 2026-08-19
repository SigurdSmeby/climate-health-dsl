"""Rare outbreak shocks: a low baseline punctuated by sudden sharp spikes.

Models extreme events — a heavy-rain week, a heatwave — the rare-event case
forecasters handle poorly. Unlike ``seasonal_spike`` (a smooth yearly bump),
shock timing is random (Poisson), so no two years look alike.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year


@register_generator("outbreak_shocks")
class OutbreakShocksGenerator(VariableGenerator):
    """A ``baseline`` (+noise) with rare shocks of size ``magnitude``.

    Params: ``baseline`` (quiet-period level), ``noise`` (Gaussian std),
    ``rate`` (expected shocks per year), ``magnitude`` (rise above baseline),
    ``duration`` (periods each shock lasts), ``clamp_min`` (optional floor).
    """

    def __init__(
        self,
        baseline: float = 0.0,
        noise: float = 0.0,
        rate: float = 1.0,
        magnitude: float = 20.0,
        duration: int = 1,
        clamp_min: float | None = None,
    ):
        if rate < 0:
            raise ValueError(f"rate must be >= 0, got {rate}")
        if not isinstance(duration, int) or duration < 1:
            raise ValueError(f"duration must be an int >= 1, got {duration!r}")
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        if magnitude <= 0:
            raise ValueError(f"magnitude must be > 0, got {magnitude}")
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

        if self.rate > 0:
            ppy = periods_per_year(period)
            # One Poisson draw for the whole span, starts uniform over the
            # timeline — keeps the per-year rate constant at any length.
            expected = self.rate * n_periods / ppy
            n_events = rng.poisson(expected)
            starts = rng.integers(0, n_periods, size=n_events)
            # Shock windows drawn independently can overlap; build a single
            # shock layer with np.maximum (not +=) so an overlap caps at one
            # magnitude per period, matching the "rise above baseline"
            # contract rather than letting overlapping events stack
            # unboundedly.
            shock = np.zeros(n_periods)
            for s in starts:
                shock[s : s + self.duration] = np.maximum(
                    shock[s : s + self.duration], self.magnitude
                )
            series = series + shock

        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
