"""Tests for the linear_trend generator (a slow rising/falling baseline)."""
import numpy as np
import pytest

from dsl.generators.linear_trend import LinearTrendGenerator


def test_output_length_matches_n_periods(rng):
    assert len(LinearTrendGenerator().generate(40, "weekly", rng)) == 40


def test_rises_by_slope_per_period(rng):
    # No noise: value at period t is start + slope * t.
    series = LinearTrendGenerator(start=10.0, slope=2.0, noise=0.0).generate(
        5, "weekly", rng
    )
    assert np.array_equal(series, np.array([10.0, 12.0, 14.0, 16.0, 18.0]))


def test_negative_slope_falls(rng):
    series = LinearTrendGenerator(start=100.0, slope=-5.0, noise=0.0).generate(
        4, "monthly", rng
    )
    assert np.array_equal(series, np.array([100.0, 95.0, 90.0, 85.0]))


def test_zero_slope_is_constant(rng):
    series = LinearTrendGenerator(start=7.0, slope=0.0, noise=0.0).generate(
        20, "weekly", rng
    )
    assert np.all(series == 7.0)


def test_monotonic_without_noise(rng):
    series = LinearTrendGenerator(start=0.0, slope=1.0, noise=0.0).generate(
        50, "weekly", rng
    )
    assert np.all(np.diff(series) > 0)


def test_same_seed_identical_different_seed_different():
    gen = LinearTrendGenerator(slope=1.0, noise=1.0)
    a = gen.generate(50, "weekly", np.random.default_rng(1))
    b = gen.generate(50, "weekly", np.random.default_rng(1))
    c = gen.generate(50, "weekly", np.random.default_rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_clamp_min_floors_values(rng):
    # A falling trend that would go negative is floored.
    series = LinearTrendGenerator(
        start=10.0, slope=-1.0, noise=0.0, clamp_min=0.0
    ).generate(30, "weekly", rng)
    assert (series >= 0.0).all()
    assert (series == 0.0).any()


def test_negative_noise_rejected():
    with pytest.raises(ValueError, match="noise"):
        LinearTrendGenerator(noise=-1.0)
