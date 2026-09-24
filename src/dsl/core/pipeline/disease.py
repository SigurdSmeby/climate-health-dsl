"""Turn a series' float signal into integer case counts.

The incidence model, in order: shift the signal so a typical period sits at
``median_rate``, squash it through a sigmoid capped at ``max_rate``, scale by
population to get an expected count, then draw a whole number from it. The
result is population-relative incidence, not a weighted sum of the drivers.
"""
import numpy as np

from dsl.core.config.schema import CountsSpec


def build_counts(
    signal: np.ndarray,
    counts: CountsSpec,
    population: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw case counts from a finished float signal.

    Args:
        signal: The series' combined driver signal, NaN where a lag warm-up
            or a missing driver leaves it undefined.
        counts: The series' counts block (rates and draw distribution).
        population: Headcount per period, the scale the rate is applied to.
        rng: Seeded generator, so the draws are reproducible.

    Returns:
        A float array of integer-valued counts, NaN wherever the input
        signal was NaN — a period with a missing input gets no fabricated
        count.
        Example: array([nan, nan, 5.0, 8.0, 12.0, ...]).
    """
    # The sampler can't take NaN rates: zero them for the draw but remember
    # the rows, so they can be blanked again afterwards.
    missing_input = np.isnan(signal)
    signal = np.nan_to_num(signal, nan=0.0)

    # The logit shift maps signal = 0 (a typical period) to median_rate; the
    # sigmoid keeps every rate below max_rate.
    shifted = signal + _logit(counts.median_rate / counts.max_rate)
    # Clip before exp only to silence a spurious overflow warning — the
    # sigmoid has already saturated by +-700.
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(shifted, -700.0, 700.0)))
    incidence_rate = sigmoid * population * counts.max_rate

    drawn = _draw(incidence_rate, counts, rng)
    drawn = np.minimum(drawn, population)
    drawn[missing_input] = np.nan
    return drawn


def _logit(p: float) -> float:
    """The inverse sigmoid, used to shift the median incidence rate.

    Args:
        p: A probability in (0, 1) — here, median_rate / max_rate.

    Returns:
        log(p / (1 - p)).
    """
    return float(np.log(p / (1.0 - p)))


def _draw(
    rate: np.ndarray, counts: CountsSpec, rng: np.random.Generator
) -> np.ndarray:
    """Draw integer case counts from the per-period incidence rate.

    Poisson has variance == mean. Negative binomial adds overdispersion via
    the gamma-Poisson mixture: with dispersion k, Var = rate + rate^2/k, so
    a smaller k means spikier counts.

    Args:
        rate: Per-period incidence rate (non-negative floats).
        counts: The counts block, for distribution and overdispersion.
        rng: Seeded random generator for reproducibility.

    Returns:
        A float array the same length as rate, holding non-negative
        integer-valued counts (as floats, so NaN can be assigned later).
    """
    if counts.distribution == "poisson":
        return rng.poisson(rate).astype(float)

    k = counts.overdispersion
    # rate can be 0 in quiet periods; gamma handles scale 0 as a
    # deterministic 0.
    gamma_rate = rng.gamma(shape=k, scale=rate / k)
    return rng.poisson(gamma_rate).astype(float)
