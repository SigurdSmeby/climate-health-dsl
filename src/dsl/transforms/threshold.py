"""Nonlinear threshold response: reshape a driver so its effect is nonlinear.

Real climate-disease links are rarely linear (the distributed-lag-nonlinear-
model literature): an effect may appear only past a rainfall/temperature
threshold, switch on/off, or grow with distance from an optimum. This lets a
scenario plant such a relationship as known ground truth, so you can test
whether a forecaster recovers a nonlinearity — not just a straight weight.

Applied per-dependency (after the causal lag, before standardize) via the
``transforms:`` list on a ``depends_on`` entry.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform

_MODES = ("hinge", "step", "quadratic")


@register_transform("threshold")  # this string is what you write in YAML
class ThresholdTransform(Transform):
    """Reshape a series nonlinearly around ``threshold``.

    mode:
      - ``hinge``: max(0, x - threshold) — effect only above the threshold.
      - ``step``: 1 where x >= threshold else 0 — a binary switch.
      - ``quadratic``: (x - threshold)**2 — U-shape around an optimum.
    """

    def __init__(self, mode: str = "hinge", threshold: float = 0.0):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self.threshold = threshold

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a reshaped copy. ``rng`` unused (deterministic). NaN preserved."""
        x = series.astype(float)  # copy + NaN-capable; NaN flows through each op
        if self.mode == "hinge":
            return np.maximum(0.0, x - self.threshold)
        if self.mode == "step":
            # NaN >= t is False → 0.0; a missing input carries no signal, which
            # matches how the disease model treats a 0-effect period. Keep NaN
            # explicit so the row is still blanked, not fabricated.
            out = np.where(x >= self.threshold, 1.0, 0.0)
            out[np.isnan(x)] = np.nan
            return out
        # quadratic
        return (x - self.threshold) ** 2
