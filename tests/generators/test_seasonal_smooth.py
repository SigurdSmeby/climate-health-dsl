"""Tests for the seasonal_smooth generator (temperature-style sine wave)."""
import numpy as np
import pytest

from dsl.generators.seasonal_smooth import SeasonalSmoothGenerator


def test_output_length_matches_n_periods(rng):
    series = SeasonalSmoothGenerator().generate(78, "weekly", rng)
    assert len(series) == 78


def test_same_seed_identical_different_seed_different():
    gen = SeasonalSmoothGenerator()
    a = gen.generate(78, "weekly", np.random.default_rng(1))
    b = gen.generate(78, "weekly", np.random.default_rng(1))
    c = gen.generate(78, "weekly", np.random.default_rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_approximately_periodic_over_one_year(rng):
    # With noise off the wave is exactly periodic: value at t equals value
    # at t + one year.
    gen = SeasonalSmoothGenerator(mean=15.0, amplitude=10.0, noise=0.0)
    series = gen.generate(104, "weekly", rng)
    assert np.allclose(series[:52], series[52:104])


def test_mean_and_amplitude_respected(rng):
    gen = SeasonalSmoothGenerator(mean=15.0, amplitude=10.0, noise=0.0)
    series = gen.generate(52, "weekly", rng)
    assert np.mean(series) == pytest.approx(15.0, abs=0.01)
    assert np.max(series) == pytest.approx(25.0, abs=0.1)
    assert np.min(series) == pytest.approx(5.0, abs=0.1)


def test_scales_to_monthly_resolution(rng):
    # One full cycle per year regardless of resolution: 12 points for monthly.
    gen = SeasonalSmoothGenerator(noise=0.0)
    series = gen.generate(24, "monthly", rng)
    assert np.allclose(series[:12], series[12:24])


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="amplitude"):
        SeasonalSmoothGenerator(amplitude=-1.0)
    with pytest.raises(ValueError, match="noise"):
        SeasonalSmoothGenerator(noise=-0.5)
