"""Nonlinear threshold response: reshape a driver so its effect is nonlinear.

Real climate-disease links are rarely linear: an effect may appear only past
a threshold, switch on/off, or grow with distance from an optimum. Applied
per-dependency (after the causal lag, before standardize) via the
``transforms:`` list on a ``depends_on`` entry.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform

_MODES = ("hinge", "step", "quadratic", "peak")


@register_transform("threshold")
class ThresholdTransform(Transform):
    """Reshape a series nonlinearly around threshold.

    Registered as "threshold" in the transform registry. The output shape
    depends on mode:
      - ``hinge``: max(0, x - threshold) — effect only above the threshold.
        Example: array([0.0, 0.0, 3.2, 7.8, ...]) for threshold=5.
      - ``step``: 1 where x >= threshold else 0 — a binary switch.
        Example: array([0.0, 0.0, 1.0, 1.0, ...]) for threshold=5.
      - ``quadratic``: (x - threshold)**2 — U-shape: the extremes
        contribute most, the threshold itself contributes nothing.
        Example: array([25.0, 9.0, 4.9, 60.8, ...]) for threshold=5.
      - ``peak``: -(x - threshold)**2 — inverted U: the threshold is an
        optimum and both sides contribute less. This is the shape
        temperature-driven transmission usually has.
        Example: array([-25.0, -9.0, -4.9, -60.8, ...]) for threshold=5.

    ``quadratic`` and ``peak`` are each other's negation, so a ``quadratic``
    with a negative ``weight:`` on the dependency gives the same curve as
    ``peak``. Both exist because a mode names the intent: "an optimum in the
    middle" reads off the scenario, whereas a sign flip on a weight does not.
    """

    def __init__(self, mode: str = "hinge", threshold: float = 0.0):
        """Store the YAML params: for this transform.

        Errors Caught (raised to caller):
            ValueError: If mode is not one of "hinge", "step", "quadratic",
                "peak", or threshold is not a finite number.
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        # A string reached numpy as a raw UFuncTypeError, and NaN compared
        # false against everything — blanking the whole series silently,
        # which looks like a driver that simply stopped mattering.
        # isinstance rather than float(): float("5") would quietly accept a
        # quoted number, and bool is an int in Python, so `threshold: true`
        # would mean 1 without saying so.
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            # ValueError, not TypeError: _build_transform converts a
            # transform's ValueError into a message naming the series, and
            # a TypeError there means "wrong signature" instead.
            raise ValueError(  # noqa: TRY004
                f"threshold must be a number, got {threshold!r}"
            )
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError(
                f"threshold must be a finite number, got {threshold!r}; "
                f"NaN compares false against every value and infinity sits "
                f"past all of them, so either would flatten the series."
            )
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
        if self.mode == "quadratic":
            return (x - self.threshold) ** 2
        if self.mode == "peak":
            return -((x - self.threshold) ** 2)  # inverted U, optimum at threshold
        # Unreachable from YAML: __init__ rejects any mode outside _MODES.
        # Explicit anyway so a mode added to _MODES without a branch here
        # fails loudly instead of silently returning the negated quadratic.
        raise ValueError(f"unhandled mode {self.mode!r}")
