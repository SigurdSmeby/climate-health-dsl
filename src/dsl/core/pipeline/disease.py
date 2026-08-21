"""Builds the dependent disease signal — where the known ground truth is made.

A population-relative Poisson incidence model, NOT a plain linear sum:

1. Seasonal baseline ``eta`` (one sine cycle per year).
2. Per dependency: lag the driver, apply its transforms, standardize to a
   z-score, weight, add to ``eta``.
3. Optional autoregressive random walk.
4. Sigmoid-squash ``eta`` (shifted so the median lands near ``median_rate``),
   then scale: ``rate x population x max_rate``.
5. Draw counts, cap at the population, blank the lag warm-up.
6. Inject missing values last.

All randomness comes from the passed seeded ``rng``.
"""
import numpy as np

from dsl.core.config.schema import DiseaseSpec
from dsl.core.extension.transform_base import get_transform
from dsl.core.pipeline.periods import periods_per_year
from dsl.transforms.lag import LagTransform
from dsl.transforms.missing import MissingTransform


def build_disease_cases(
    drivers: dict[str, np.ndarray],
    spec: DiseaseSpec,
    rng: np.random.Generator,
    n_periods: int,
    period: str,
) -> np.ndarray:
    """Build the disease_cases series from the generated drivers.

    Execute the 7-step disease model: seasonal baseline -> weighted lagged
    drivers -> autoregressive walk -> sigmoid squash -> Poisson (or negative
    binomial) draw -> cap at population -> blank warm-up/missing-input rows
    -> inject missing values.

    Args:
        drivers: Dict mapping variable names to time series arrays.
        spec: DiseaseSpec with dependencies, rates, population, and
            missing_rate.
        rng: Seeded random generator for reproducibility.
        n_periods: Number of time periods.
        period: Period type (e.g., "monthly", "daily").

    Returns:
        A float array of length n_periods with Poisson-drawn case counts.
        NaN in the lag warm-up and wherever missing_rate struck.
        Example: array([nan, nan, 5.0, 8.0, 12.0, nan, 15.0, 11.0, ...])
    """
    ppy = periods_per_year(period)
    t = np.arange(n_periods)

    # Step 1: Seasonal baseline, so disease has its own seasonality even
    # with no drivers attached.
    eta = np.sin(2 * np.pi * (t % ppy) / ppy)
    # Now eta = [0.0, sin(2*pi/ppy), sin(4*pi/ppy), ...] (one cycle/year)

    # Step 2: Add each lagged, transformed, standardized, weighted driver.
    # A weight-0 dependency is skipped so its NaNs don't blank rows it
    # doesn't influence.
    for dep in spec.depends_on:
        if dep.weight == 0:
            continue
        series = LagTransform(n=dep.lag).apply(drivers[dep.variable], rng)
        for tf in dep.transforms:
            series = get_transform(tf.name)(**tf.params).apply(series, rng)
        eta = eta + dep.weight * _standardize(series)
    # Now eta = seasonal baseline + sum of weighted, standardized drivers

    # Step 3: Optional autoregressive component: a random walk gives the
    # signal memory of its own past.
    if spec.autoregressive:
        eta = eta + np.cumsum(rng.normal(0.0, 0.2, size=n_periods))

    # The sampler can't take NaN rates: zero them for the draw but remember
    # the rows — a period with a missing input must not get a fabricated
    # count.
    missing_input = np.isnan(eta)
    eta = np.nan_to_num(eta, nan=0.0)

    # Step 4: eta -> incidence rate. The logit shift maps eta = 0 (a
    # typical period) to median_rate; the sigmoid keeps every rate below
    # max_rate * population. Clip before exp only to avoid a spurious
    # overflow warning — the sigmoid has already saturated by +-700.
    shifted = eta + _logit(spec.median_rate / spec.max_rate)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(shifted, -700.0, 700.0)))
    incidence_rate = sigmoid * spec.population * spec.max_rate
    # Now incidence_rate = [rate_0, rate_1, ...], each in [0, population*max_rate]

    # Step 5: Draw counts and cap at the population.
    counts = _draw_counts(incidence_rate, spec, rng)
    counts = np.minimum(counts, spec.population)

    # Step 6: Blank rows whose input was missing (warm-up + missing driver
    # values).
    counts[missing_input] = np.nan

    # Step 7: Missing data last, so gaps land on the finished signal.
    if spec.missing_rate > 0:
        counts = MissingTransform(rate=spec.missing_rate).apply(counts, rng)

    return counts
    # Returns a float array of length n_periods; NaN marks warm-up and
    # missing periods, everything else is a non-negative case count.


def _standardize(series: np.ndarray) -> np.ndarray:
    """Z-score a series, ignoring NaN, guarding zero variance.

    Standardizing puts every driver on the same scale, so YAML weight
    values are comparable regardless of raw units (mm of rain vs degrees).

    Args:
        series: The driver's time series (may contain NaN).

    Returns:
        (series - mean) / std, ignoring NaN in mean/std. A constant series
        (std == 0) returns zeros with its NaN positions preserved, so those
        disease rows still get blanked instead of fabricated.
    """
    if np.all(np.isnan(series)):
        return series.astype(float)
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if std == 0:
        out = np.zeros_like(series, dtype=float)
        out[np.isnan(series)] = np.nan
        return out
    return (series - mean) / std


def _logit(p: float) -> float:
    """The inverse sigmoid, used to shift the median incidence rate.

    Args:
        p: A probability in (0, 1) — here, median_rate / max_rate.

    Returns:
        log(p / (1 - p)).
    """
    return float(np.log(p / (1.0 - p)))


def _draw_counts(
    rate: np.ndarray, spec: DiseaseSpec, rng: np.random.Generator
) -> np.ndarray:
    """Draw integer case counts from the per-period incidence rate.

    Poisson (default) has variance == mean. Negative binomial adds
    overdispersion via the gamma-Poisson mixture: with dispersion k,
    Var = rate + rate^2/k — a smaller k means spikier counts.

    Args:
        rate: Per-period incidence rate (non-negative floats).
        spec: DiseaseSpec with count_distribution and overdispersion.
        rng: Seeded random generator for reproducibility.

    Returns:
        A float array the same length as rate, holding non-negative
        integer-valued counts (as floats, so NaN can be assigned later).
    """
    if spec.count_distribution == "poisson":
        return rng.poisson(rate).astype(float)

    k = spec.overdispersion
    # rate can be 0 in quiet periods; gamma handles scale 0 as a
    # deterministic 0.
    gamma_rate = rng.gamma(shape=k, scale=rate / k)
    return rng.poisson(gamma_rate).astype(float)
