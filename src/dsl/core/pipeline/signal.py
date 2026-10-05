"""Build one series' float signal from its generator and its parents.

Every series is a float array before anything else happens to it. This module
owns that shared path — base generator, weighted lagged parents, persistence,
gaps — so a plain climate column and a disease signal are built the same way.
What happens *after* is what differs: a plain series is written as-is, a count
series goes on through the incidence model in ``disease.py``.
"""
import numpy as np

from dsl.core.extension.transform_base import get_transform
from dsl.transforms.lag import LagTransform
from dsl.transforms.missing import MissingTransform


def build_signal(
    base: np.ndarray,
    spec,
    parents: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Combine a series' base with its parents' contributions.

    Args:
        base: The series' own generated values, or zeros if it has no
            generator of its own.
        spec: The SeriesSpec being built (its depends_on and autoregressive).
        parents: Already-built values of every series this one depends on,
            keyed by name.
        rng: Seeded generator for this series' randomness.

    Returns:
        The finished float signal, NaN wherever a lag warm-up or a missing
        parent value leaves it undefined.
        Example: array([nan, 0.4, 1.9, -0.7, ...]) for one lag-1 parent.

    Errors Caught (raised to caller):
        KeyError: If a dependency's transform name is not registered.
    """
    signal = base.astype(float)

    # A weight-0 parent is skipped rather than multiplied out, so its warm-up
    # NaNs don't blank rows it has no influence on.
    for dep in spec.depends_on:
        if dep.weight == 0:
            continue
        values = LagTransform(n=dep.lag).apply(parents[dep.series], rng)
        for tf in dep.transforms:
            values = get_transform(tf.name)(**tf.params).apply(values, rng)
        signal = signal + dep.weight * _standardize(values)

    if spec.autoregressive is not None:
        signal = signal + _ar1(
            spec.autoregressive.phi, spec.autoregressive.noise, len(signal), rng
        )

    return signal


def apply_missing(
    values: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Blank a fraction of a finished series, imitating reporting gaps.

    Args:
        values: The finished series.
        rate: Expected fraction of periods to blank, in [0, 1].
        rng: Seeded generator for the mask.

    Returns:
        values unchanged if rate is 0, else a copy with ~rate of its entries
        set to NaN.
    """
    if rate <= 0:
        return values
    return MissingTransform(rate=rate).apply(values, rng)


def _ar1(
    phi: float, noise: float, n_periods: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw an AR(1) process: ``x[t] = phi * x[t-1] + e[t]``.

    Persistence that stays bounded: with phi < 1 the series is pulled back
    toward zero, so it never drifts away the way a cumulative sum would.

    Args:
        phi: Fraction of each period's value carried into the next, in [0, 1).
        noise: Standard deviation of the per-period shock.
        n_periods: Length of the series.
        rng: Seeded generator for the shocks.

    Returns:
        A float array of length n_periods, starting at 0.
        Example: array([0.0, 0.21, 0.35, 0.18, ...]) for phi=0.8.
    """
    shocks = rng.normal(0.0, noise, size=n_periods)
    out = np.zeros(n_periods)
    for t in range(1, n_periods):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def _standardize(series: np.ndarray) -> np.ndarray:
    """Z-score a series, ignoring NaN, guarding zero variance.

    Standardizing puts every parent on the same scale, so YAML weight values
    are comparable regardless of raw units (mm of rain vs degrees).

    Args:
        series: The parent's values (may contain NaN).

    Returns:
        (series - mean) / std, ignoring NaN in mean/std. A constant series
        (std == 0) returns zeros with its NaN positions preserved, so those
        rows still get blanked instead of fabricated.
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
