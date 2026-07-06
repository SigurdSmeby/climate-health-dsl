"""Tests for the heavy_tail transform (Student-t noise, fat tails / outliers).

TDD: written before src/dsl/transforms/heavy_tail.py. Real surveillance data
has occasional extreme values Gaussian noise can't produce; this adds heavy-
tailed noise so a scenario can stress a model's robustness to outliers.
"""
import numpy as np
import pytest

from dsl.transforms.heavy_tail import HeavyTailTransform


def test_zero_scale_is_identity(rng):
    series = np.array([1.0, 2.0, 3.0, 4.0])
    result = HeavyTailTransform(scale=0.0).apply(series, rng)
    assert np.array_equal(result, series)


def test_adds_noise_around_series(rng):
    series = np.full(5000, 10.0)
    result = HeavyTailTransform(scale=1.0, df=3).apply(series, rng)
    # Centered on the series, but with heavier tails than Gaussian.
    assert np.median(result) == pytest.approx(10.0, abs=0.2)
    assert not np.array_equal(result, series)


def test_tails_are_heavier_than_gaussian(rng):
    # Student-t with low df produces more extreme outliers than a Gaussian of
    # the same scale. Compare max absolute deviation.
    n = 20000
    series = np.zeros(n)
    t_noise = HeavyTailTransform(scale=1.0, df=2).apply(
        series, np.random.default_rng(0))
    gauss = np.random.default_rng(0).normal(0.0, 1.0, size=n)
    assert np.abs(t_noise).max() > np.abs(gauss).max()


def test_nan_preserved(rng):
    series = np.array([np.nan, 5.0])
    result = HeavyTailTransform(scale=1.0).apply(series, rng)
    assert np.isnan(result[0])
    assert not np.isnan(result[1])


def test_does_not_modify_input(rng):
    series = np.array([1.0, 2.0, 3.0])
    original = series.copy()
    HeavyTailTransform(scale=1.0).apply(series, rng)
    assert np.array_equal(series, original)


def test_deterministic_under_seed():
    a = HeavyTailTransform(scale=2.0).apply(
        np.zeros(100), np.random.default_rng(0))
    b = HeavyTailTransform(scale=2.0).apply(
        np.zeros(100), np.random.default_rng(0))
    assert np.array_equal(a, b)


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="scale"):
        HeavyTailTransform(scale=-1.0)
    with pytest.raises(ValueError, match="df"):
        HeavyTailTransform(df=0)


def test_registered_and_reachable():
    from dsl.core.extension.transform_base import get_transform

    assert get_transform("heavy_tail") is HeavyTailTransform
