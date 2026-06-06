"""A flat, non-seasonal series — a constant level plus optional noise.

Useful as a CONTROL or DECOY covariate: a variable with no seasonal signal,
so a scenario can test whether a model correctly ignores an irrelevant
driver (or whether it's fooled into using one). Unlike the seasonal
generators, successive years do NOT repeat.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator


@register_generator("flat")  # this string is what you write in YAML
class FlatGenerator(VariableGenerator):
    """A constant ``level`` with optional Gaussian noise — no seasonality."""

    def __init__(
        self,
        level: float = 0.0,
        noise: float = 1.0,
        clamp_min: float | None = None,
    ):
        """Store and validate the YAML ``params:`` for this variable.

        Parameters
        ----------
        level:
            The constant value the series sits at.
        noise:
            Standard deviation of additive Gaussian noise (0 → a flat line).
        clamp_min:
            If set, values are floored at this minimum (e.g. 0).
        """
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.level = level
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Return the flat series, length ``n_periods``."""
        # np.full makes an array of n_periods copies of `level`.
        series = np.full(n_periods, float(self.level))
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
