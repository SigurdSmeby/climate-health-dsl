"""Tests for generators reporting the features they place.

A generator that deliberately puts something at a known period — a seasonal
peak, an outbreak shock — can say so. That is exact ground truth: the periods
come from the generator itself, not from detecting spikes after the fact, so
counting them per fold is a lookup rather than an estimate.

Generators that place nothing identifiable report nothing, and need no code.
"""
import numpy as np
import pytest

from dsl.core.config.schema import parse_config
from dsl.core.extension.generator_base import VariableGenerator, get_generator
from dsl.core.pipeline.engine import run, series_events
from dsl.core.pipeline.periods import periods_per_year
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict


def _build(name, **params):
    """Instantiate a registered generator and run it once."""
    generator = get_generator(name)(**params)
    values = generator.generate(104, "weekly", np.random.default_rng(0))
    return generator, values


# --------------------------------------------------------------- the default


def test_base_class_reports_nothing():
    """A generator that plants nothing identifiable needs no extra code."""

    class Plain(VariableGenerator):
        def generate(self, n_periods, period, rng):
            return np.zeros(n_periods)

    assert Plain().events() == []


@pytest.mark.parametrize("name", ["flat", "linear_trend", "seasonal_smooth"])
def test_featureless_generators_report_nothing(name):
    generator, _ = _build(name)
    assert generator.events() == []


def test_events_is_callable_before_generate():
    """Nothing has been placed yet, so there is nothing to report."""
    assert get_generator("outbreak_shocks")().events() == []


# ------------------------------------------------------------ seasonal_spike


def test_seasonal_spike_reports_one_peak_per_year():
    generator, _ = _build("seasonal_spike", spike_center=10)
    # 104 weekly periods is two years.
    assert len(generator.events()) == 2


def test_seasonal_spike_peaks_land_on_the_declared_centre():
    generator, _ = _build("seasonal_spike", spike_center=10)
    ppy = periods_per_year("weekly")
    assert [e["index"] for e in generator.events()] == [10, 10 + ppy]


def test_seasonal_spike_peaks_match_the_actual_maximum():
    """The reported period is where the series really peaks."""
    generator, values = _build("seasonal_spike", spike_center=20, noise=0.0)
    first_year = values[: periods_per_year("weekly")]
    assert generator.events()[0]["index"] == int(np.argmax(first_year))


def test_seasonal_spike_reports_its_kind_and_magnitude():
    generator, _ = _build("seasonal_spike", spike_height=30.0)
    event = generator.events()[0]
    assert event["kind"] == "seasonal_spike"
    assert event["magnitude"] == pytest.approx(30.0)


def test_seasonal_spike_default_centre_is_reported():
    generator, _ = _build("seasonal_spike")
    assert generator.events()[0]["index"] == periods_per_year("weekly") // 2


# ----------------------------------------------------------- outbreak_shocks


def test_outbreak_shocks_reports_the_shocks_it_drew():
    generator, values = _build(
        "outbreak_shocks", baseline=5.0, magnitude=20.0, rate=3.0, noise=0.0
    )
    events = generator.events()
    assert events
    # Every reported period is genuinely elevated in the output.
    for event in events:
        assert values[event["index"]] > 5.0


def test_outbreak_shocks_reports_every_elevated_window():
    generator, values = _build(
        "outbreak_shocks", baseline=0.0, magnitude=10.0, rate=4.0,
        duration=3, noise=0.0,
    )
    reported = {e["index"] for e in generator.events()}
    elevated = set(np.flatnonzero(values > 0.0))
    # A shock's reported start must be the first period of its window.
    assert reported <= elevated
    assert min(reported) == min(elevated)


def test_outbreak_shocks_reports_duration():
    generator, _ = _build(
        "outbreak_shocks", magnitude=10.0, rate=4.0, duration=3, noise=0.0
    )
    assert all(e["duration"] == 3 for e in generator.events())


def test_outbreak_shocks_with_zero_rate_reports_nothing():
    generator, _ = _build("outbreak_shocks", rate=0.0)
    assert generator.events() == []


def test_outbreak_shocks_events_are_reproducible():
    first, _ = _build("outbreak_shocks", rate=3.0)
    second, _ = _build("outbreak_shocks", rate=3.0)
    assert first.events() == second.events()


# ------------------------------------------------------- collected by the run


def test_run_collects_events_per_series_and_location():
    config = parse_config(
        make_config_dict(
            series=[
                series_dict(
                    "rainfall",
                    generate="seasonal_spike",
                    params={"spike_center": 10},
                ),
                series_dict(
                    "cases",
                    counts={"population": 100_000},
                    depends_on=[{"series": "rainfall", "lag": 1}],
                ),
            ],
            locations=["north", "south"],
        )
    )
    events = series_events(config)
    assert {e["series"] for e in events} == {"rainfall"}
    assert {e["location"] for e in events} == {"north", "south"}


def test_collected_events_carry_the_generator_name():
    config = parse_config(make_config_dict())
    kinds = {e["kind"] for e in series_events(config)}
    assert "seasonal_spike" in kinds


def test_scenario_events_are_collected_too():
    """A declared shock is ground truth just as much as a generator's."""
    config = parse_config(
        make_config_dict(
            events={"storm": {"at": [4, 9]}},
            series=[
                series_dict(
                    "rainfall",
                    generate="flat",
                    events={"storm": {"multiplier": 2.0}},
                ),
            ],
        )
    )
    storms = [e for e in series_events(config) if e["kind"] == "storm"]
    assert sorted(e["index"] for e in storms) == [4, 9]


def test_a_scenario_with_no_features_collects_nothing():
    config = parse_config(
        make_config_dict(series=[series_dict("x", generate="flat")])
    )
    assert series_events(config) == []


def test_collecting_events_does_not_disturb_the_data():
    """Reporting must be a read, not a second run with its own draws."""
    config = parse_config(make_config_dict())
    before = run(config)["rainfall"].to_numpy()
    series_events(config)
    np.testing.assert_array_equal(before, run(config)["rainfall"].to_numpy())
