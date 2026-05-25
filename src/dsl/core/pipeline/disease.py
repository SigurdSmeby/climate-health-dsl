"""Builds the dependent disease signal — where the known ground truth is made.

This is a population-relative Poisson incidence model (ported from the
reference implementation), NOT a plain linear sum:

1. Start from a seasonal baseline ``eta`` (one sine cycle per year).
2. For each dependency: lag the named driver (causally), standardize it to
   a z-score, multiply by its weight, and add to ``eta``.
3. Optionally add an autoregressive random walk.
4. Squash ``eta`` through a sigmoid (shifted so the median lands near
   ``median_rate``), then scale: ``rate × population × max_rate``.
5. Draw Poisson counts, cap at the population, and blank the lag warm-up.
6. Inject missing values last.

All randomness comes from the passed ``rng`` — the reference code used the
unseeded global ``np.random`` and was not reproducible; this is a fix.
"""
import numpy as np

from dsl.core.config.schema import DiseaseSpec
from dsl.core.pipeline.periods import periods_per_year
from dsl.transforms.lag import LagTransform
from dsl.transforms.missing import MissingTransform


def _standardize(series: np.ndarray) -> np.ndarray:
    """Z-score a series (mean 0, std 1), ignoring NaN, guarding zero variance.

    Standardizing puts every driver on the same scale, so the ``weight``
    values in the YAML are comparable across drivers regardless of their
    raw units (mm of rain vs degrees).
    """
    # nanmean/nanstd skip the NaN warm-up a lagged series carries.
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if std == 0:
        # A constant series carries no signal; mapping it to zeros avoids a
        # divide-by-zero turning the whole eta into NaN.
        return np.zeros_like(series)
    return (series - mean) / std


def _logit(p: float) -> float:
    """The inverse sigmoid, log(p / (1 - p)) — used to shift the median."""
    return float(np.log(p / (1.0 - p)))


def build_disease_cases(
    drivers: dict[str, np.ndarray],
    spec: DiseaseSpec,
    rng: np.random.Generator,
    n_periods: int,
    period: str,
) -> np.ndarray:
    """Build the ``disease_cases`` series from the generated drivers.

    Parameters
    ----------
    drivers:
        Generated variable series, keyed by variable name (the engine builds
        this from the YAML ``variables:`` section).
    spec:
        The validated ``disease_cases:`` section of the scenario.
    rng:
        The single seeded random generator for the whole run.
    n_periods:
        Length of the series to produce.
    period:
        Resolution ("daily"/"weekly"/...), used for the seasonal baseline.

    Returns
    -------
    np.ndarray
        A float array of length ``n_periods`` holding integer counts, with
        NaN in the first ``max_lag`` rows (warm-up) and wherever
        ``missing_rate`` struck.
    """
    ppy = periods_per_year(period)
    t = np.arange(n_periods)

    # 1. Seasonal baseline: one sine cycle per year (the reference model's
    #    season weights), so disease has its own seasonality even with no
    #    drivers attached.
    eta = np.sin(2 * np.pi * (t % ppy) / ppy)

    # 2. Add each lagged, standardized, weighted driver.
    max_lag = 0
    for dep in spec.depends_on:
        lagged = LagTransform(n=dep.lag).apply(drivers[dep.variable], rng)
        eta = eta + dep.weight * _standardize(lagged)
        max_lag = max(max_lag, dep.lag)

    # 3. Optional autoregressive component: a random walk (cumulative sum of
    #    white noise), giving the signal memory of its own past.
    if spec.autoregressive:
        eta = eta + np.cumsum(rng.normal(0.0, 0.2, size=n_periods))

    # The warm-up rows are NaN (from the lag); the Poisson sampler cannot
    # take NaN rates, so temporarily zero them — they are blanked below.
    eta = np.nan_to_num(eta, nan=0.0)

    # 4. eta → incidence rate. Adding logit(median/max) shifts the sigmoid
    #    so that eta = 0 (the typical period) maps to median_rate once
    #    multiplied by max_rate; the sigmoid keeps every rate below
    #    max_rate * population no matter how extreme eta gets.
    shifted = eta + _logit(spec.median_rate / spec.max_rate)
    sigmoid = 1.0 / (1.0 + np.exp(-shifted))
    poisson_rate = sigmoid * spec.population * spec.max_rate

    # 5. Draw integer counts and cap at the population (a count of sick
    #    people can't exceed the people that exist).
    counts = rng.poisson(poisson_rate).astype(float)
    counts = np.minimum(counts, spec.population)

    # 6. Blank the warm-up: the first max_lag rows have no valid lagged
    #    signal, so reporting counts there would be fabricated data.
    if max_lag > 0:
        counts[:max_lag] = np.nan

    # 7. Missing data last, so gaps land on the finished signal.
    if spec.missing_rate > 0:
        counts = MissingTransform(rate=spec.missing_rate).apply(counts, rng)

    return counts
