"""Case counts: a population-relative incidence model.

Turns a float signal into whole case counts. In order: shift the signal so a
typical period sits at ``median_rate``, squash it through a sigmoid capped at
``max_rate``, scale by population to get an expected count, then draw a whole
number. The result is incidence relative to the population, not a weighted
sum of the drivers.
"""
import numpy as np

from dsl.core.extension.emitter_base import Emitter, register_emitter


@register_emitter("counts")  # this string is the block name written in YAML
class CountsEmitter(Emitter):
    """Draws integer case counts against a population.

    Registered as "counts" in the emitter registry. emit() maps a signal of 0
    to roughly ``median_rate`` of the population and caps every period at
    ``max_rate``, then draws Poisson or negative-binomial counts.
    Example: array([nan, nan, 9985.0, 10207.0, ...]) for population=100000,
    median_rate=0.1.
    """

    def __init__(
        self,
        population: np.ndarray,
        max_rate: float = 0.3,
        median_rate: float = 0.1,
        distribution: str = "poisson",
        overdispersion: float = 10.0,
    ):
        """Store the counts: block's params, plus the resolved population.

        Args:
            population: Headcount per period — the scale a rate is applied
                to, and what each period's count is capped at.
            max_rate: Ceiling, as a fraction of the population.
            median_rate: Where a typical period (signal 0) sits.
            distribution: "poisson" (variance == mean) or
                "negative_binomial" (overdispersed, spikier).
            overdispersion: Negative binomial only; smaller means more
                variable.

        Errors Caught (raised to caller):
            ValueError: If median_rate is not below max_rate — the sigmoid
                shift is undefined outside (0, 1).
        """
        if median_rate >= max_rate:
            raise ValueError(
                f"median_rate ({median_rate}) must be smaller than "
                f"max_rate ({max_rate})."
            )
        self.population = population
        self.max_rate = max_rate
        self.median_rate = median_rate
        self.distribution = distribution
        self.overdispersion = overdispersion

    def emit(self, signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw case counts from a finished float signal.

        Args:
            signal: The series' combined signal, NaN where a lag warm-up or
                a missing parent leaves it undefined.
            rng: Seeded generator, so the draws are reproducible.

        Returns:
            A float array of integer-valued counts, NaN wherever the input
            was NaN — a period with a missing input gets no fabricated count.
            Example: array([nan, nan, 5.0, 8.0, 12.0, ...]).
        """
        # The sampler can't take NaN rates: zero them for the draw but
        # remember the rows, so they can be blanked again afterwards.
        missing_input = np.isnan(signal)
        signal = np.nan_to_num(signal, nan=0.0)

        # The logit shift maps signal = 0 (a typical period) to median_rate;
        # the sigmoid keeps every rate below max_rate.
        shifted = signal + _logit(self.median_rate / self.max_rate)
        # Clip before exp only to silence a spurious overflow warning — the
        # sigmoid has already saturated by +-700.
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(shifted, -700.0, 700.0)))
        incidence_rate = sigmoid * self.population * self.max_rate

        drawn = self._draw(incidence_rate, rng)
        drawn = np.minimum(drawn, self.population)
        drawn[missing_input] = np.nan
        return drawn

    def _draw(
        self, rate: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw integer counts from the per-period incidence rate.

        Poisson has variance == mean. Negative binomial adds overdispersion
        via the gamma-Poisson mixture: with dispersion k,
        Var = rate + rate^2/k, so a smaller k means spikier counts.

        Args:
            rate: Per-period incidence rate (non-negative floats).
            rng: Seeded random generator for reproducibility.

        Returns:
            A float array the same length as rate, holding non-negative
            integer-valued counts (as floats, so NaN can be assigned later).
        """
        if self.distribution == "poisson":
            return rng.poisson(rate).astype(float)

        k = self.overdispersion
        # rate can be 0 in quiet periods; gamma handles scale 0 as a
        # deterministic 0.
        gamma_rate = rng.gamma(shape=k, scale=rate / k)
        return rng.poisson(gamma_rate).astype(float)


def _logit(p: float) -> float:
    """The inverse sigmoid, used to shift the median incidence rate.

    Args:
        p: A probability in (0, 1) — here, median_rate / max_rate.

    Returns:
        log(p / (1 - p)).
    """
    return float(np.log(p / (1.0 - p)))
