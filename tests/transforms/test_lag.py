"""Tests for the causal lag transform."""
import numpy as np
import pytest

from dsl.transforms.lag import LagTransform


def test_lag_shifts_series(rng):
    series = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    result = LagTransform(n=2).apply(series, rng)
    assert result[5] == 5.0  # the spike moved forward by exactly 2


def test_lag_zero_is_identity(rng):
    series = np.array([1.0, 2.0, 3.0])
    result = LagTransform(n=0).apply(series, rng)
    assert np.array_equal(result, series)


def test_warmup_is_nan_not_wrapped(rng):
    # Causal shift: the first n positions have no valid past, so they are
    # NaN. np.roll would wrap the series END here — leaking the future into
    # the past — which is exactly what this transform must NOT do.
    series = np.array([10.0, 20.0, 30.0, 40.0])
    result = LagTransform(n=2).apply(series, rng)
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert result[2] == 10.0 and result[3] == 20.0


def test_lag_does_not_modify_input(rng):
    series = np.array([1.0, 2.0, 3.0, 4.0])
    original = series.copy()
    LagTransform(n=1).apply(series, rng)
    assert np.array_equal(series, original)


def test_negative_lag_rejected():
    with pytest.raises(ValueError, match="n"):
        LagTransform(n=-1)


def test_lag_longer_than_series_is_all_nan(rng):
    series = np.array([1.0, 2.0, 3.0])
    result = LagTransform(n=5).apply(series, rng)
    assert np.isnan(result).all()
