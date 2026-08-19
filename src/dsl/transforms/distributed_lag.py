"""Distributed lag: spread a driver's effect across a kernel of lags.

Reality smears an effect over a window — rain raises risk this week, more
next week, tapering off (the DLNM setup). This convolves the driver with a
weight kernel to plant that smeared relationship as ground truth. Causal
like ``lag``: weight i is the effect at lag i, and the first
``len(weights) - 1`` positions become warm-up NaN. Never wraps.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("distributed_lag")
class DistributedLagTransform(Transform):
    """Convolve a series with a causal weight kernel over lags 0..len-1."""

    def __init__(self, weights: list[float] | None = None):
        if not weights:
            raise ValueError("weights must be a non-empty list")
        self.weights = np.asarray(weights, dtype=float)

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = series.astype(float)
        k = len(self.weights)
        # Position t of the convolution is sum_i weights[i] * x[t - i]:
        # causal, output t only reads inputs at t and earlier.
        conv = np.convolve(x, self.weights)[: len(x)]
        if k > 1:
            conv[: k - 1] = np.nan  # incomplete past → warm-up NaN
        return conv
