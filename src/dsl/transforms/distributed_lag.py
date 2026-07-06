"""Distributed lag: spread a driver's effect across a kernel of lags.

A single ``lag`` says "disease follows rainfall by exactly N periods". Reality
smears the effect over a window — a rain event raises risk this week, more next
week, tapering off — which is the distributed-lag-nonlinear-model (DLNM) setup
forecasters are evaluated against. This convolves the driver with a weight
kernel so a scenario can plant that smeared relationship as known ground truth.

Causal, like ``lag``: weight index i is the effect at lag i, and the first
``len(weights) - 1`` positions have no full past, so they become NaN (the
warm-up the disease model blanks). Never wraps.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("distributed_lag")  # this string is what you write in YAML
class DistributedLagTransform(Transform):
    """Convolve a series with a causal weight kernel over lags 0..len-1."""

    def __init__(self, weights: list[float] = None):
        if not weights:
            raise ValueError("weights must be a non-empty list")
        self.weights = np.asarray(weights, dtype=float)

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the kernel-weighted series. ``rng`` unused (deterministic)."""
        x = series.astype(float)  # copy; input untouched
        k = len(self.weights)
        # 'full' convolution then take the first len(x) outputs: position t is
        # sum_i weights[i] * x[t - i], i.e. weight i is the lag-i contribution.
        # This is causal — output t only reads inputs at t and earlier.
        conv = np.convolve(x, self.weights)[: len(x)]
        # The first k-1 positions summed over an incomplete past → warm-up NaN,
        # matching the single-lag convention. (NaN in x already propagates
        # through convolve for later positions.)
        if k > 1:
            conv[: k - 1] = np.nan
        return conv
