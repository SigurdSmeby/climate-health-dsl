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
    # the configured center. spike_center counts from 1, so week 20 is
    # index 19.
    gen = SeasonalSpikeGenerator(spike_center=20, noise=0.0)
    series = gen.generate(52, "weekly", rng)
    assert int(np.argmax(series)) == 19


def test_spike_repeats_each_year(rng):
    gen = SeasonalSpikeGenerator(spike_center=10, noise=0.0)
    series = gen.generate(104, "weekly", rng)  # two full years
    assert int(np.argmax(series[:52])) == 9
    assert int(np.argmax(series[52:])) == 9


def test_baseline_away_from_spike(rng):
    gen = SeasonalSpikeGenerator(
        baseline=2.0, spike_height=50.0, spike_center=10, spike_width=2.0, noise=0.0
    )
    series = gen.generate(52, "weekly", rng)
    # Half a year away from a narrow spike, the value is essentially baseline.
    assert series[35] == pytest.approx(2.0, abs=0.1)


def test_scales_to_monthly_resolution(rng):
    # spike_center is a position within the year at any resolution, so on
    # monthly data it must land inside the 12-month cycle.
    gen = SeasonalSpikeGenerator(spike_center=5, spike_width=1.0, noise=0.0)
    series = gen.generate(24, "monthly", rng)
    assert int(np.argmax(series[:12])) == 4  # May


def test_explicit_center_beyond_one_cycle_wraps(rng):
    # A center past the end of the cycle wraps, so the spike still reaches
    # its configured peak. Month 25 of a 12-month year is month 1.
    gen = SeasonalSpikeGenerator(
        baseline=0, spike_height=1, spike_center=25, spike_width=0.5, noise=0
    )
    series = gen.generate(12, "monthly", rng)
    assert int(np.argmax(series)) == 0  # (25 - 1) % 12 == 0, January
    assert series.max() == pytest.approx(1.0, abs=0.05)


def test_monthly_default_reaches_its_peak(rng):
    # With the default center, a monthly series must reach
    # baseline+spike_height somewhere.
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


# ------------------------------------------------------- calendar alignment


def test_spike_center_is_a_month_number(rng):
    """spike_center: 7 means July — the number a reader expects to write."""
    gen = SeasonalSpikeGenerator(spike_center=7, spike_width=1.0, noise=0.0)
    series = gen.generate(12, "monthly", rng)
    assert int(np.argmax(series)) == 6  # index 6 is the 7th month


def test_month_one_is_january(rng):
    gen = SeasonalSpikeGenerator(spike_center=1, spike_width=1.0, noise=0.0)
    series = gen.generate(12, "monthly", rng)
    assert int(np.argmax(series)) == 0


def test_zero_is_rejected(rng):
    """There is no month 0, so it is a typo rather than a silent December."""
    with pytest.raises(ValueError, match="spike_center"):
        SeasonalSpikeGenerator(spike_center=0)


def test_negative_is_rejected(rng):
    with pytest.raises(ValueError, match="spike_center"):
        SeasonalSpikeGenerator(spike_center=-3)


def test_start_period_mid_year_keeps_the_peak_in_its_month(rng):
    """A July peak stays in July, wherever the series happens to begin.

    Position within the cycle is counted from the calendar, not from index
    0, so the label a row carries and the season it holds agree.
    """
    gen = SeasonalSpikeGenerator(
        spike_center=7, spike_width=1.0, noise=0.0, start_period="2000-07"
    )
    series = gen.generate(12, "monthly", rng)
    # The series opens in July, so the first row IS the peak.
    assert int(np.argmax(series)) == 0


def test_start_period_offsets_the_cycle(rng):
    gen = SeasonalSpikeGenerator(
        spike_center=1, spike_width=1.0, noise=0.0, start_period="2000-04"
    )
    series = gen.generate(12, "monthly", rng)
    # Starting in April, January is nine rows later.
    assert int(np.argmax(series)) == 9


def test_without_start_period_the_year_starts_at_index_zero(rng):
    gen = SeasonalSpikeGenerator(spike_center=3, spike_width=1.0, noise=0.0)
    series = gen.generate(12, "monthly", rng)
    assert int(np.argmax(series)) == 2


def test_events_report_the_same_periods_the_data_peaks_at(rng):
    """The fold report reads events(), so a mismatch misplaces every spike."""
    gen = SeasonalSpikeGenerator(
        spike_center=7, spike_width=1.0, noise=0.0, start_period="2000-04"
    )
    series = gen.generate(36, "monthly", rng)
    reported = [e["index"] for e in gen.events()]
    for index in reported:
        year = series[max(index - 5, 0) : index + 6]
        assert series[index] == pytest.approx(year.max()), (
            f"events() says a peak at {index}, but the data does not peak there"
        )
    assert len(reported) == 3  # three years, three peaks


def test_events_follow_start_period(rng):
    gen = SeasonalSpikeGenerator(
        spike_center=7, spike_width=1.0, noise=0.0, start_period="2000-07"
    )
    gen.generate(24, "monthly", rng)
    assert [e["index"] for e in gen.events()] == [0, 12]
