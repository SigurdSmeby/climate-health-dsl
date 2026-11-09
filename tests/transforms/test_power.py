"""Tests for the power transform.

`power` is one knob over a family of shapes: 0.5 saturates mildly, 1.0 is
linear, 2.0 accelerates. The identity at 1.0 is what makes it sweepable, so
that is pinned explicitly.
"""
import numpy as np
import pytest

from dsl.transforms.power import PowerTransform


def test_exponent_one_is_identity(rng):
    # Pinned because it is the neutral midpoint of a sweep over the shape.
    series = np.array([0.0, 2.5, 10.0])
    result = PowerTransform(exponent=1.0).apply(series, rng)
    assert np.allclose(result, series)


def test_square_root_saturates(rng):
    result = PowerTransform(exponent=0.5).apply(np.array([0.0, 4.0, 16.0]), rng)
    assert np.allclose(result, np.array([0.0, 2.0, 4.0]))


def test_squaring_accelerates(rng):
    result = PowerTransform(exponent=2.0).apply(np.array([1.0, 2.0, 3.0]), rng)
    assert np.allclose(result, np.array([1.0, 4.0, 9.0]))


def test_nan_preserved(rng):
    result = PowerTransform(exponent=2.0).apply(np.array([2.0, np.nan]), rng)
    assert result[0] == 4.0
    assert np.isnan(result[1])


def test_does_not_modify_input(rng):
    series = np.array([2.0, 3.0])
    original = series.copy()
    PowerTransform(exponent=2.0).apply(series, rng)
    assert np.array_equal(series, original)


def test_negative_base_with_fractional_exponent_is_rejected(rng):
    # (-5) ** 0.5 is complex in Python and NaN in numpy. Temperature in C
    # reaches negative values, so this is reachable from a real scenario.
    with pytest.raises(ValueError, match="fractional exponent"):
        PowerTransform(exponent=0.5).apply(np.array([4.0, -5.0]), rng)


def test_negative_base_with_whole_exponent_is_fine(rng):
    # (-5) ** 2 is 25 — a real number, so no reason to refuse it.
    result = PowerTransform(exponent=2.0).apply(np.array([-5.0]), rng)
    assert result[0] == 25.0


def test_nan_alone_does_not_trip_the_domain_guard(rng):
    result = PowerTransform(exponent=0.5).apply(np.array([np.nan, 4.0]), rng)
    assert np.isnan(result[0])
    assert result[1] == 2.0


def test_a_non_numeric_exponent_is_rejected():
    with pytest.raises(ValueError, match="exponent must be a number"):
        PowerTransform(exponent="2")


def test_a_non_finite_exponent_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        PowerTransform(exponent=float("inf"))


def test_registered_and_reachable_from_scenario():
    from dsl.core.extension.transform_base import get_transform

    assert get_transform("power") is PowerTransform
