"""Tests for the log transform (saturating response).

The point of `log` is diminishing returns: 0 -> 100mm of rain must matter more
than 100 -> 200mm. The tests pin that ordering, the domain guard, and that
shift keeps a legitimate zero finite.
"""
import numpy as np
import pytest

from dsl.transforms.log import LogTransform


def test_zero_stays_finite_with_default_shift(rng):
    # rainfall: 0 is legitimate, and log(0 + 1) = 0 rather than -inf.
    result = LogTransform().apply(np.array([0.0]), rng)
    assert result[0] == 0.0


def test_diminishing_returns(rng):
    # The whole reason to use log: equal ratios matter equally, so the first
    # 100mm gains more than the second 100mm.
    result = LogTransform(shift=1.0).apply(np.array([0.0, 100.0, 200.0]), rng)
    first_step = result[1] - result[0]
    second_step = result[2] - result[1]
    assert first_step > second_step


def test_matches_numpy_log_of_shifted(rng):
    series = np.array([0.0, 10.0, 20.0, 100.0])
    result = LogTransform(shift=1.0).apply(series, rng)
    assert np.allclose(result, np.log(series + 1.0))


def test_nan_preserved(rng):
    result = LogTransform().apply(np.array([0.0, np.nan, 10.0]), rng)
    assert np.isnan(result[1])
    assert not np.isnan(result[0]) and not np.isnan(result[2])


def test_does_not_modify_input(rng):
    series = np.array([0.0, 10.0])
    original = series.copy()
    LogTransform().apply(series, rng)
    assert np.array_equal(series, original)


def test_non_positive_after_shift_is_rejected(rng):
    # Temperature in C goes negative; without a big enough shift log is
    # undefined there. Erroring beats emitting -inf/NaN into the signal,
    # which nothing downstream would catch.
    with pytest.raises(ValueError, match="above 0 after shifting"):
        LogTransform(shift=1.0).apply(np.array([5.0, -3.0]), rng)


def test_a_bigger_shift_rescues_a_negative_series(rng):
    # The error message says to raise shift; this proves that works.
    result = LogTransform(shift=10.0).apply(np.array([5.0, -3.0]), rng)
    assert np.all(np.isfinite(result))


def test_nan_alone_does_not_trip_the_domain_guard(rng):
    # NaN is a reporting gap, not a domain error.
    result = LogTransform().apply(np.array([np.nan, 5.0]), rng)
    assert np.isnan(result[0])


def test_a_non_numeric_shift_is_rejected():
    with pytest.raises(ValueError, match="shift must be a number"):
        LogTransform(shift="5")


def test_a_non_finite_shift_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        LogTransform(shift=float("nan"))


def test_registered_and_reachable_from_scenario():
    from dsl.core.extension.transform_base import get_transform

    assert get_transform("log") is LogTransform
