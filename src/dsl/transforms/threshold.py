"""Nonlinear threshold response: reshape a driver so its effect is nonlinear.

Real climate-disease links are rarely linear: an effect may appear only past
a threshold, switch on/off, or grow with distance from an optimum. Applied
per-dependency (after the causal lag, before standardize) via the
``transforms:`` list on a ``depends_on`` entry.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform

_MODES = ("hinge", "step", "quadratic")


@register_transform("threshold")
class ThresholdTransform(Transform):
    """Reshape a series nonlinearly around threshold.

    Registered as "threshold" in the transform registry.

    mode:
      - ``hinge``: max(0, x - threshold) — effect only above the threshold.
      - ``step``: 1 where x >= threshold else 0 — a binary switch.
      - ``quadratic``: (x - threshold)**2 — U-shape around an optimum.
    """

    def __init__(self, mode: str = "hinge", threshold: float = 0.0):
        """Store the YAML params: for this transform.

        Errors Caught (raised to caller):
            ValueError: If mode is not one of "hinge", "step", "quadratic".
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self.threshold = threshold

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply the configured nonlinear reshape.

        Args:
            series: The input time series.
            rng: Seeded random generator (required by the transform API,
                unused — threshold is deterministic).

        Returns:
            The reshaped series, same length as series, per self.mode
            (see the class docstring). NaN in series stays NaN.
        """
        x = series.astype(float)  # copy; NaN flows through each op
        if self.mode == "hinge":
            return np.maximum(0.0, x - self.threshold)
        if self.mode == "step":
            # Keep NaN explicit so a missing input still blanks the row.
            out = np.where(x >= self.threshold, 1.0, 0.0)
            out[np.isnan(x)] = np.nan
            return out
        return (x - self.threshold) ** 2  # quadratic
