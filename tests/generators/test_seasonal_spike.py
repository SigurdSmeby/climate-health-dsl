"""Tests for the seasonal_spike generator (rainy-season-style variable)."""
import numpy as np
import pytest

from dsl.generators.seasonal_spike import SeasonalSpikeGenerator


def test_output_length_matches_n_periods(rng):
    series = SeasonalSpikeGenerator().generate(78, "weekly", rng)
    assert len(series) == 78


def test_same_seed_identical_different_seed_different():
    gen = SeasonalSpikeGenerator()
    a = gen.generate(78, "weekly", np.random.default_rng(1))
    b = gen.generate(78, "weekly", np.random.default_rng(1))
    c = gen.generate(78, "weekly", np.random.default_rng(2))
    assert np.array_equal(a, b)  # same seed → bit-identical
    assert not np.array_equal(a, c)  # different seed → different draw


def test_peak_is_at_spike_center(rng):
    # With noise off, the maximum of one yearly cycle must sit exactly at
    # the configured center.
    gen = SeasonalSpikeGenerator(spike_center=20, noise=0.0)
    series = gen.generate(52, "weekly", rng)
    assert int(np.argmax(series)) == 20


def test_spike_repeats_each_year(rng):
    gen = SeasonalSpikeGenerator(spike_center=10, noise=0.0)
    series = gen.generate(104, "weekly", rng)  # two full years
    assert int(np.argmax(series[:52])) == 10
    assert int(np.argmax(series[52:])) == 10


def test_baseline_away_from_spike(rng):
    gen = SeasonalSpikeGenerator(
        baseline=2.0, spike_height=50.0, spike_center=10, spike_width=2.0, noise=0.0
    )
    series = gen.generate(52, "weekly", rng)
    # Half a year away from a narrow spike, the value is essentially baseline.
    assert series[36] == pytest.approx(2.0, abs=0.1)


def test_scales_to_monthly_resolution(rng):
    # The same spike_center means "period offset within the year", so for
    # monthly data it must land within the 12-month cycle.
    gen = SeasonalSpikeGenerator(spike_center=5, spike_width=1.0, noise=0.0)
    series = gen.generate(24, "monthly", rng)
    assert int(np.argmax(series[:12])) == 5


def test_explicit_center_beyond_one_cycle_wraps(rng):
    # Bug #35: a center >= ppy must wrap (24 on monthly == month 0), so the
    # spike actually reaches its configured peak.
    gen = SeasonalSpikeGenerator(
        baseline=0, spike_height=1, spike_center=24, spike_width=0.5, noise=0
    )
    series = gen.generate(12, "monthly", rng)
    assert int(np.argmax(series)) == 0  # 24 % 12 == 0
    assert series.max() == pytest.approx(1.0, abs=0.05)


def test_monthly_default_reaches_its_peak(rng):
    # Bug #35: with the default center, a monthly series must still actually
    # reach baseline+spike_height somewhere (it previously topped out below).
    gen = SeasonalSpikeGenerator(noise=0)
    series = gen.generate(12, "monthly", rng)
    assert series.max() == pytest.approx(
        gen.baseline + gen.spike_height, abs=0.5
    )


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="spike_width"):
        SeasonalSpikeGenerator(spike_width=0)
    with pytest.raises(ValueError, match="noise"):
        SeasonalSpikeGenerator(noise=-1.0)


def test_clamp_min_prevents_negative_values(rng):
    # Big noise on a low baseline would dip below zero (impossible for
    # rainfall); clamp_min floors the series.
    gen = SeasonalSpikeGenerator(baseline=2.0, noise=50.0, clamp_min=0.0)
    series = gen.generate(520, "weekly", rng)
    assert (series >= 0.0).all()


def test_no_clamp_by_default(rng):
    gen = SeasonalSpikeGenerator(baseline=2.0, noise=50.0)
    series = gen.generate(520, "weekly", rng)
    assert (series < 0.0).any()  # without a clamp, noise can go negative
