"""A flat, non-seasonal series — a constant level plus optional noise.

Useful as a control or decoy covariate: a scenario can test whether a model
correctly ignores an irrelevant driver.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator


@register_generator("flat")
class FlatGenerator(VariableGenerator):
    """A constant ``level`` with optional Gaussian noise — no seasonality.

    Params: ``level`` (the constant), ``noise`` (Gaussian std, 0 → flat line),
    ``clamp_min`` (optional floor, e.g. 0).
    """

    def __init__(
        self,
        level: float = 0.0,
        noise: float = 1.0,
        clamp_min: float | None = None,
    ):
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.level = level
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        series = np.full(n_periods, float(self.level))
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
