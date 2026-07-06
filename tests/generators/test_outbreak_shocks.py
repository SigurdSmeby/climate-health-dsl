"""Tests for the outbreak_shocks generator (rare sharp spikes on a baseline).

TDD: written before src/dsl/generators/outbreak_shocks.py. Distinct from
seasonal_spike (a smooth annual bump) — these are rare, sudden shocks, the
extreme-event case forecasters struggle with.
"""
import numpy as np
import pytest

from dsl.generators.outbreak_shocks import OutbreakShocksGenerator


def test_output_length_matches_n_periods(rng):
    assert len(OutbreakShocksGenerator().generate(60, "weekly", rng)) == 60


def test_no_shocks_when_rate_zero(rng):
    series = OutbreakShocksGenerator(
        baseline=2.0, noise=0.0, rate=0.0,
    ).generate(200, "weekly", rng)
    assert np.all(series == 2.0)  # pure baseline, no spikes


def test_shocks_are_rare_and_large(rng):
    # rate ~1 event/year over 20 years → a handful of shocks, each well above
    # the baseline. Most periods sit at baseline; few are spikes.
    series = OutbreakShocksGenerator(
        baseline=1.0, noise=0.0, rate=1.0, magnitude=50.0, duration=1,
    ).generate(52 * 20, "weekly", rng)
    spikes = series[series > 10.0]
    assert 5 <= len(spikes) <= 60          # rare, not every period
    assert len(spikes) < len(series) * 0.1  # <10% of periods are shocks


def test_duration_widens_each_shock(rng):
    # duration=3 → each event elevates ~3 consecutive periods.
    short = OutbreakShocksGenerator(
        baseline=0.0, noise=0.0, rate=2.0, magnitude=30.0, duration=1,
    ).generate(52 * 10, "weekly", np.random.default_rng(1))
    wide = OutbreakShocksGenerator(
        baseline=0.0, noise=0.0, rate=2.0, magnitude=30.0, duration=4,
    ).generate(52 * 10, "weekly", np.random.default_rng(1))
    assert (wide > 1.0).sum() > (short > 1.0).sum()


def test_deterministic_under_seed():
    a = OutbreakShocksGenerator(rate=2.0, magnitude=20.0).generate(
        300, "weekly", np.random.default_rng(0))
    b = OutbreakShocksGenerator(rate=2.0, magnitude=20.0).generate(
        300, "weekly", np.random.default_rng(0))
    assert np.array_equal(a, b)


def test_negative_rate_rejected():
    with pytest.raises(ValueError, match="rate"):
        OutbreakShocksGenerator(rate=-1.0)


def test_registered_and_reachable():
    from dsl.core.extension.generator_base import get_generator

    assert get_generator("outbreak_shocks") is OutbreakShocksGenerator
