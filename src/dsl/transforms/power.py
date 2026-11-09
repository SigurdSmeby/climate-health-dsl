"""Power response: one exponent covering a family of curve shapes.

``exponent`` 0.5 is a square root (mild saturation), 1.0 is linear (identity),
2.0 accelerates. Sweeping the exponent sweeps the *shape* itself, which asks
whether a model can recover a nonlinearity at all. Applied per-dependency
(after the causal lag, before standardize) via the ``transforms:`` list on a
``depends_on`` entry.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("power")
class PowerTransform(Transform):
    """Raise a series to ``exponent``.

    Registered as "power" in the transform registry. apply() returns
    ``x ** exponent``.
    Example: array([0.0, 3.2, 4.5, 10.0, ...]) for a rainfall series
    [0, 10, 20, 100] with exponent=0.5.

    ``exponent: 1.0`` is an identity, which makes it the neutral midpoint of
    a sweep: ``exponent: { sweep: [0.5, 1.0, 2.0] }`` compares mild
    saturation, no nonlinearity, and acceleration against each other.
    """

    def __init__(self, exponent: float = 1.0):
        """Store the YAML params: for this transform.

        Errors Caught (raised to caller):
            ValueError: If exponent is not a finite number.
        """
        # isinstance rather than float(): see log.py — a quoted number or a
        # bool would otherwise pass silently.
        if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
            raise ValueError(  # noqa: TRY004
                f"exponent must be a number, got {exponent!r}"
            )
        exponent = float(exponent)
        if not np.isfinite(exponent):
            raise ValueError(
                f"exponent must be a finite number, got {exponent!r}"
            )
        self.exponent = exponent

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Raise the series to self.exponent.

        Args:
            series: The input time series.
            rng: Seeded random generator (required by the transform API,
                unused — power is deterministic).

        Returns:
            series ** exponent, same length as series. NaN stays NaN.

        Errors Caught (raised to caller):
            ValueError: If the exponent is fractional and any non-NaN value
                is negative. (-5) ** 0.5 is a complex number in Python and
                NaN in numpy, so the alternative is silent NaN in the
                disease signal. Erroring matches how `threshold` rejects
                params it cannot honour.
        """
        x = series.astype(float)  # copy; NaN flows through
        if self.exponent != int(self.exponent):
            bad = ~np.isnan(x) & (x < 0.0)
            if bad.any():
                raise ValueError(
                    f"power with a fractional exponent ({self.exponent:g}) "
                    f"needs every value >= 0, but {int(bad.sum())} of "
                    f"{bad.size} are negative (lowest: {np.nanmin(x):g}). "
                    f"A negative base with a fractional exponent is not a "
                    f"real number. Shift the series first, or use a whole "
                    f"exponent."
                )
        return x**self.exponent
