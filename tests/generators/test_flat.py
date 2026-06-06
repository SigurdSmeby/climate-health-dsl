"""Tests for the flat generator (a non-seasonal control/decoy covariate)."""
import numpy as np
import pytest

from dsl.generators.flat import FlatGenerator


def test_output_length_matches_n_periods(rng):
    assert len(FlatGenerator().generate(40, "weekly", rng)) == 40


def test_no_noise_is_constant(rng):
    series = FlatGenerator(level=5.0, noise=0.0).generate(30, "weekly", rng)
    assert np.all(series == 5.0)


def test_noise_varies_around_level(rng):
    series = FlatGenerator(level=10.0, noise=2.0).generate(2000, "weekly", rng)
    # Mean near the level, real spread, but no seasonal structure.
    assert np.mean(series) == pytest.approx(10.0, abs=0.3)
    assert np.std(series) == pytest.approx(2.0, abs=0.3)


def test_not_seasonal(rng):
    # The defining property: the first year and second year are NOT a repeat
    # of each other (unlike the seasonal generators).
    series = FlatGenerator(level=10.0, noise=1.0).generate(104, "weekly", rng)
    assert not np.allclose(series[:52], series[52:104])


def test_same_seed_identical_different_seed_different():
    gen = FlatGenerator(noise=1.0)
    a = gen.generate(50, "weekly", np.random.default_rng(1))
    b = gen.generate(50, "weekly", np.random.default_rng(1))
    c = gen.generate(50, "weekly", np.random.default_rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_clamp_min_floors_values(rng):
    series = FlatGenerator(level=0.5, noise=5.0, clamp_min=0.0).generate(
        500, "weekly", rng
    )
    assert (series >= 0.0).all()


def test_negative_noise_rejected():
    with pytest.raises(ValueError, match="noise"):
        FlatGenerator(noise=-1.0)
