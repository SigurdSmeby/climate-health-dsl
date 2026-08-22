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
    """A baseline (+noise) with rare shocks of size magnitude.

    Registered as "outbreak_shocks" in the generator registry. generate()
    returns baseline plus noise, with rare Poisson-timed shock windows of
    height magnitude layered on top (overlaps cap at one magnitude, not
    stacked), floored at clamp_min if set.
    Example: array([5.1, 4.8, 25.3, 24.9, 5.2, ...]) for baseline=5,
    magnitude=20, duration=2.
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
        """Store the YAML params: for this variable.

        Args:
            rate: Expected shocks per year (Poisson), not a fixed count.
            duration: How many periods each shock stays elevated.

        Errors Caught (raised to caller):
            ValueError: If rate < 0, duration isn't an int >= 1, noise < 0,
                or magnitude <= 0.
        """
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
        """Generate the baseline-plus-shocks series.

        Args:
            n_periods: Number of time periods.
            period: Period type (e.g., "monthly", "daily") — sets how many
                periods make up a year, for converting rate to a per-span
                expected shock count.
            rng: Seeded random generator for reproducibility.

        Returns:
            A numpy array of length n_periods, holding baseline plus noise
            plus any shock windows, floored at clamp_min if set.
            Example: array([5.1, 4.8, 25.3, 24.9, 5.2, ...]) for baseline=5,
            magnitude=20, duration=2.
        """
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
