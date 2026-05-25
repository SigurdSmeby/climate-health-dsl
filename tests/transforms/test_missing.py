"""Tests for the missing-value injection transform."""
import numpy as np
import pytest

from dsl.transforms.missing import MissingTransform


def test_rate_zero_changes_nothing(rng):
    series = np.arange(100.0)
    result = MissingTransform(rate=0.0).apply(series, rng)
    assert np.array_equal(result, series)


def test_rate_half_gives_about_half_nan(rng):
    series = np.ones(10_000)
    result = MissingTransform(rate=0.5).apply(series, rng)
    nan_fraction = np.isnan(result).mean()
    # Random draw, so allow sampling slack around 0.5.
    assert nan_fraction == pytest.approx(0.5, abs=0.03)


def test_same_seed_same_mask():
    series = np.ones(1000)
    a = MissingTransform(rate=0.3).apply(series, np.random.default_rng(7))
    b = MissingTransform(rate=0.3).apply(series, np.random.default_rng(7))
    # NaN != NaN, so compare the masks rather than the values.
    assert np.array_equal(np.isnan(a), np.isnan(b))


def test_different_seed_different_mask():
    series = np.ones(1000)
    a = MissingTransform(rate=0.3).apply(series, np.random.default_rng(7))
    b = MissingTransform(rate=0.3).apply(series, np.random.default_rng(8))
    assert not np.array_equal(np.isnan(a), np.isnan(b))


def test_surviving_values_unchanged(rng):
    series = np.arange(100.0)
    result = MissingTransform(rate=0.3).apply(series, rng)
    kept = ~np.isnan(result)
    assert np.array_equal(result[kept], series[kept])


def test_does_not_modify_input(rng):
    series = np.arange(100.0)
    original = series.copy()
    MissingTransform(rate=0.5).apply(series, rng)
    assert np.array_equal(series, original)


def test_rate_out_of_range_rejected():
    with pytest.raises(ValueError, match="rate"):
        MissingTransform(rate=1.5)
    with pytest.raises(ValueError, match="rate"):
        MissingTransform(rate=-0.1)
