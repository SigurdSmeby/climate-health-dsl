"""A linear trend — a value that rises or falls steadily over time.

Models slow drift that real series often carry: population growth, gradual
warming, urbanization, changing reporting completeness. Useful as a
confounder (a trend that is NOT seasonal but still correlates with anything
else trending) when testing whether a model separates trend from signal.
"""
import numpy as np

from dsl.core.extension.generator_base import VariableGenerator, register_generator


@register_generator("linear_trend")  # this string is what you write in YAML
class LinearTrendGenerator(VariableGenerator):
    """A straight line start + slope * t with optional Gaussian noise.

    Registered as "linear_trend" in the generator registry. generate()
    returns start + slope*t plus noise, floored at clamp_min if set.
    Example: array([70000, 70090, 70180, 70270, ...]) for start=70000,
    slope=90.
    """

    def __init__(
        self,
        start: float = 0.0,
        slope: float = 1.0,
        noise: float = 0.0,
        clamp_min: float | None = None,
    ):
        """Store the YAML params: for this variable.

        Errors Caught (raised to caller):
            ValueError: If noise < 0.
        """
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.start = start
        self.slope = slope
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate the trend series.

        Args:
            n_periods: Number of time periods.
            period: Period type (e.g., "monthly", "daily"); unused —
                linear_trend has no seasonal shape to align.
            rng: Seeded random generator for reproducibility.

        Returns:
            A numpy array of length n_periods, holding start + slope*t plus
            optional noise, floored at clamp_min if set.
            Example: array([70000, 70090, 70180, 70270, ...]) for
            start=70000, slope=90.
        """
        t = np.arange(n_periods)
        series = self.start + self.slope * t
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
