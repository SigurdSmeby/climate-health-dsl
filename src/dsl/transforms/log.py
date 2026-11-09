"""Saturating response: reshape a driver so equal ratios, not equal
differences, matter equally.

Doubling an already-high driver usually matters less than doubling a low one —
rainfall-to-breeding-site relationships are normally described that way.
Applied per-dependency (after the causal lag, before standardize) via the
``transforms:`` list on a ``depends_on`` entry.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("log")
class LogTransform(Transform):
    """Take the natural log of a series, after adding ``shift``.

    Registered as "log" in the transform registry. apply() returns
    ``log(x + shift)``, so the default ``shift=1.0`` gives log1p-style
    behaviour and keeps a legitimate ``rainfall: 0`` finite at log(1) = 0.
    Example: array([0.0, 2.4, 3.0, 4.6, ...]) for a rainfall series
    [0, 10, 20, 100] with shift=1.

    Climate series legitimately hit zero (rainfall) or go negative
    (temperature in C), and log is undefined there. ``shift`` moves the
    series into the domain; a value still non-positive after shifting is a
    scenario error, not something to pass through as -inf or NaN.
    """

    def __init__(self, shift: float = 1.0):
        """Store the YAML params: for this transform.

        Errors Caught (raised to caller):
            ValueError: If shift is not a finite number. np.isfinite also
                rejects NaN/Inf: a NaN shift would make every output NaN,
                which looks like a driver that stopped mattering rather
                than like a bad param.
        """
        # isinstance rather than float(): float("5") would quietly accept a
        # quoted number, and bool is an int in Python, so `shift: true`
        # would mean 1 without saying so. Mirrors threshold's check.
        if isinstance(shift, bool) or not isinstance(shift, (int, float)):
            # ValueError, not TypeError: _build_transform converts a
            # transform's ValueError into a message naming the series, and
            # a TypeError there means "wrong signature" instead.
            raise ValueError(f"shift must be a number, got {shift!r}")  # noqa: TRY004
        shift = float(shift)
        if not np.isfinite(shift):
            raise ValueError(f"shift must be a finite number, got {shift!r}")
        self.shift = shift

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Take log(series + shift).

        Args:
            series: The input time series.
            rng: Seeded random generator (required by the transform API,
                unused — log is deterministic).

        Returns:
            log(series + shift), same length as series. NaN stays NaN.

        Errors Caught (raised to caller):
            ValueError: If any non-NaN value is <= 0 after shifting, or is
                infinite. Naming the offending value is the point: log
                would otherwise emit -inf, inf or NaN into the disease
                signal, and a transform's output is not schema-validated,
                so nothing downstream would catch it. An infinity is the
                worst of the three — _standardize divides by the series'
                spread, and that spread is NaN once the series holds an
                inf, so one value turns the whole column into NaN.
        """
        x = series.astype(float)  # copy; NaN flows through
        # Before the domain check: +inf survives `shifted <= 0` and
        # np.log(inf) is inf, so it would pass straight through.
        wild = ~np.isnan(x) & ~np.isfinite(x)
        if wild.any():
            raise ValueError(
                f"log needs every value to be finite, but {int(wild.sum())} "
                f"of {wild.size} are not (first: {x[wild][0]:g}). One "
                f"infinite value would turn this driver's whole column into "
                f"NaN when it is standardized."
            )
        shifted = x + self.shift
        bad = ~np.isnan(shifted) & (shifted <= 0.0)
        if bad.any():
            # Report the original value as it actually is: reconstructing it
            # as (shifted - shift) reintroduces the rounding the addition
            # just caused, so -0.3 would print as -0.30000000000000004.
            worst = np.nanmin(x)
            raise ValueError(
                f"log needs every value above 0 after shifting, but "
                f"{int(bad.sum())} of {bad.size} are not (lowest: "
                f"{worst:g} + shift {self.shift:g} = "
                f"{worst + self.shift:g}). "
                f"Raise shift to move the series into the domain."
            )
        return np.log(shifted)
