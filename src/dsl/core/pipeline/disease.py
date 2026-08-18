"""Builds the dependent disease signal — where the known ground truth is made.

This is a population-relative Poisson incidence model, NOT a plain linear sum:

1. Start from a seasonal baseline ``eta`` (one sine cycle per year).
2. For each dependency: lag the named driver (causally), standardize it to
   a z-score, multiply by its weight, and add to ``eta``.
3. Optionally add an autoregressive random walk.
4. Squash ``eta`` through a sigmoid (shifted so the median lands near
   ``median_rate``), then scale: ``rate × population × max_rate``.
5. Draw Poisson counts, cap at the population, and blank the lag warm-up.
6. Inject missing values last.

All randomness comes from the passed seeded ``rng``, so output is reproducible.
"""
import numpy as np

from dsl.core.config.schema import DiseaseSpec
from dsl.core.extension.transform_base import get_transform
from dsl.core.pipeline.periods import periods_per_year
from dsl.transforms.lag import LagTransform
from dsl.transforms.missing import MissingTransform


def _standardize(series: np.ndarray) -> np.ndarray:
    """Z-score a series (mean 0, std 1), ignoring NaN, guarding zero variance.

    Standardizing puts every driver on the same scale, so the ``weight``
    values in the YAML are comparable across drivers regardless of their
    raw units (mm of rain vs degrees).
    """
    # nanmean/nanstd skip the NaN warm-up a lagged series carries. With all
    # values NaN they'd warn and return NaN; guard that.
    if np.all(np.isnan(series)):
        return series.astype(float)
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if std == 0:
        # A constant series carries no signal; map it to zeros — but keep the
        # NaN positions (a missing input must stay missing so the disease row
        # is blanked, not fabricated).
        out = np.zeros_like(series, dtype=float)
        out[np.isnan(series)] = np.nan
        return out
    return (series - mean) / std


def _logit(p: float) -> float:
    """The inverse sigmoid, log(p / (1 - p)) — used to shift the median."""
    return float(np.log(p / (1.0 - p)))


def _draw_counts(
    rate: np.ndarray, spec: DiseaseSpec, rng: np.random.Generator
) -> np.ndarray:
    """Draw integer case counts from the per-period incidence ``rate``.

    Poisson (the default) has variance equal to its mean. Negative binomial
    adds overdispersion: with dispersion ``k`` the variance is
    ``mean + mean^2 / k``, so real-looking spiky surveillance counts come
    from a smaller ``k``. We sample it via the gamma-Poisson mixture (draw a
    gamma-distributed rate per period, then a Poisson from that), which keeps
    the mean at ``rate`` while inflating the variance.
    """
    if spec.count_distribution == "poisson":
        return rng.poisson(rate).astype(float)

    # Negative binomial as a gamma-Poisson mixture. shape = k, scale =
    # rate / k → the gamma has mean `rate` and variance `rate^2 / k`; the
    # Poisson draw on top adds its own `rate`, giving Var = rate + rate^2/k.
    k = spec.overdispersion
    # np.errstate: rate can be 0 in quiet periods, making scale 0 — gamma
    # handles that as a deterministic 0, no warning needed.
    gamma_rate = rng.gamma(shape=k, scale=rate / k)
    return rng.poisson(gamma_rate).astype(float)


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

    # 1. Seasonal baseline: one sine cycle per year, so disease has its own
    #    seasonality even with no drivers attached.
    eta = np.sin(2 * np.pi * (t % ppy) / ppy)

    # 2. Add each lagged, standardized, weighted driver. A weight-0 dependency
    #    is skipped: it has no effect, so its lag/missing NaNs must not blank
    #    rows it doesn't influence.
    for dep in spec.depends_on:
        if dep.weight == 0:
            continue
        series = LagTransform(n=dep.lag).apply(drivers[dep.variable], rng)
        # Apply any registry transforms named in the YAML, in order, after the
        # causal lag and before standardize (same mechanism the engine uses for
        # generators: get_transform(name)(**params)).
        for tf in dep.transforms:
            series = get_transform(tf.name)(**tf.params).apply(series, rng)
        eta = eta + dep.weight * _standardize(series)

    # 3. Optional autoregressive component: a random walk (cumulative sum of
    #    white noise), giving the signal memory of its own past.
    if spec.autoregressive:
        eta = eta + np.cumsum(rng.normal(0.0, 0.2, size=n_periods))

    # eta is NaN at the lag warm-up and wherever a driver value is missing.
    # The sampler can't take NaN rates, so zero them for the draw but remember
    # the rows — a period with a missing input must not get a fabricated count.
    missing_input = np.isnan(eta)
    eta = np.nan_to_num(eta, nan=0.0)

    # 4. eta → incidence rate. Adding logit(median/max) shifts the sigmoid
    #    so that eta = 0 (the typical period) maps to median_rate once
    #    multiplied by max_rate; the sigmoid keeps every rate below
    #    max_rate * population no matter how extreme eta gets.
    shifted = eta + _logit(spec.median_rate / spec.max_rate)
    # Clip before exp: an extreme driver/weight can push `shifted` past the
    # point where exp(-shifted) overflows float64 (~±710). The sigmoid has
    # already saturated to 0/1 well before then, so clipping changes nothing
    # numerically but avoids a spurious overflow RuntimeWarning.
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(shifted, -700.0, 700.0)))
    incidence_rate = sigmoid * spec.population * spec.max_rate

    # 5. Draw integer counts (Poisson or overdispersed negative binomial) and
    #    cap at the population (a count of sick people can't exceed the people
    #    that exist).
    counts = _draw_counts(incidence_rate, spec, rng)
    counts = np.minimum(counts, spec.population)

    # 6. Blank rows whose input was missing (warm-up + missing driver values).
    counts[missing_input] = np.nan

    # 7. Missing data last, so gaps land on the finished signal.
    if spec.missing_rate > 0:
        counts = MissingTransform(rate=spec.missing_rate).apply(counts, rng)

    return counts
