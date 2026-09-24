"""Tests for named shared events.

An event is a one-off shock that fires in specific periods and can hit
several series at once — a storm raising rainfall and dropping temperature in
the same month. That same-period coupling is what a generator cannot express,
since a generator sees only its own series.

Events are regional: one draw, shared by every location, so a storm hits the
whole region in the same periods.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict


def _scenario(events, series, **overrides):
    return make_config_dict(events=events, series=series, **overrides)


def _flat(name, **fields):
    """A constant series, so an event's effect is unambiguous."""
    return series_dict(
        name, generate="flat", params={"level": 10.0, "noise": 0.0}, **fields
    )


# ---------------------------------------------------------------- the schema


def test_events_are_optional():
    assert parse_config(make_config_dict()).events == {}


def test_event_with_explicit_periods_parses():
    config = parse_config(
        _scenario(
            {"storm": {"at": [3, 7]}},
            [_flat("rain", events={"storm": {"multiplier": 2.0}})],
        )
    )
    assert config.events["storm"].at == [3, 7]


def test_event_with_a_rate_parses():
    config = parse_config(
        _scenario(
            {"storm": {"rate": 0.1}},
            [_flat("rain", events={"storm": {"multiplier": 2.0}})],
        )
    )
    assert config.events["storm"].rate == 0.1


def test_event_needs_either_rate_or_at():
    with pytest.raises(ValidationError, match="rate"):
        parse_config(_scenario({"storm": {}}, [_flat("rain")]))


def test_event_rejects_both_rate_and_at():
    """Two sources of truth for when it fires would be ambiguous."""
    with pytest.raises(ValidationError, match="rate"):
        parse_config(
            _scenario({"storm": {"rate": 0.1, "at": [3]}}, [_flat("rain")])
        )


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_event_rate_must_be_a_probability(rate):
    with pytest.raises(ValidationError, match="rate"):
        parse_config(_scenario({"storm": {"rate": rate}}, [_flat("rain")]))


def test_event_period_beyond_the_series_rejected():
    with pytest.raises(ValidationError, match="n_total"):
        parse_config(
            _scenario(
                {"storm": {"at": [500]}},
                [_flat("rain", events={"storm": {"multiplier": 2.0}})],
                n_total=24,
            )
        )


def test_negative_event_period_rejected():
    with pytest.raises(ValidationError, match="at"):
        parse_config(_scenario({"storm": {"at": [-1]}}, [_flat("rain")]))


def test_series_referencing_an_undeclared_event_rejected():
    with pytest.raises(ValidationError, match="monsoon"):
        parse_config(
            _scenario(
                {"storm": {"at": [3]}},
                [_flat("rain", events={"monsoon": {"multiplier": 2.0}})],
            )
        )


def test_effect_needs_either_multiplier_or_add():
    with pytest.raises(ValidationError, match="multiplier"):
        parse_config(
            _scenario({"storm": {"at": [3]}}, [_flat("rain", events={"storm": {}})])
        )


def test_effect_rejects_both_multiplier_and_add():
    with pytest.raises(ValidationError, match="multiplier"):
        parse_config(
            _scenario(
                {"storm": {"at": [3]}},
                [_flat("rain", events={"storm": {"multiplier": 2.0, "add": 1.0}})],
            )
        )


def test_declared_event_nobody_uses_warns():
    from dsl.core.config.schema import validate_scenario

    config = parse_config(
        _scenario({"storm": {"at": [3]}}, [_flat("rain")])
    )
    assert any("storm" in w for w in validate_scenario(config))


# -------------------------------------------------------------- the effects


def test_multiplier_scales_only_the_event_periods():
    config = parse_config(
        _scenario(
            {"storm": {"at": [5]}},
            [_flat("rain", events={"storm": {"multiplier": 2.0}})],
        )
    )
    rain = run(config)["rain"].to_numpy()
    assert rain[5] == pytest.approx(20.0)
    assert rain[4] == pytest.approx(10.0)
    assert rain[6] == pytest.approx(10.0)


def test_multiplier_below_one_scales_down():
    config = parse_config(
        _scenario(
            {"storm": {"at": [5]}},
            [_flat("rain", events={"storm": {"multiplier": 0.5}})],
        )
    )
    assert run(config)["rain"].to_numpy()[5] == pytest.approx(5.0)


def test_add_shifts_by_a_fixed_amount():
    config = parse_config(
        _scenario(
            {"storm": {"at": [5]}},
            [_flat("temp", events={"storm": {"add": -4.0}})],
        )
    )
    temp = run(config)["temp"].to_numpy()
    assert temp[5] == pytest.approx(6.0)
    assert temp[4] == pytest.approx(10.0)


def test_add_can_push_a_series_negative():
    """The difference from multiplier: an absolute shift ignores the level."""
    config = parse_config(
        _scenario(
            {"storm": {"at": [5]}},
            [_flat("temp", events={"storm": {"add": -25.0}})],
        )
    )
    assert run(config)["temp"].to_numpy()[5] < 0


def test_several_periods_all_fire():
    config = parse_config(
        _scenario(
            {"storm": {"at": [2, 9, 14]}},
            [_flat("rain", events={"storm": {"multiplier": 3.0}})],
        )
    )
    rain = run(config)["rain"].to_numpy()
    assert rain[[2, 9, 14]] == pytest.approx(30.0)
    assert rain[3] == pytest.approx(10.0)


def test_a_series_may_react_to_several_events():
    config = parse_config(
        _scenario(
            {"storm": {"at": [4]}, "drought": {"at": [8]}},
            [
                _flat(
                    "rain",
                    events={
                        "storm": {"multiplier": 2.0},
                        "drought": {"multiplier": 0.1},
                    },
                )
            ],
        )
    )
    rain = run(config)["rain"].to_numpy()
    assert rain[4] == pytest.approx(20.0)
    assert rain[8] == pytest.approx(1.0)


# --------------------------------------------------- the point: same-period


def test_one_event_hits_several_series_in_the_same_period():
    """The coupling a generator cannot express: a storm raises rainfall and
    drops temperature in the SAME period."""
    config = parse_config(
        _scenario(
            {"storm": {"at": [6]}},
            [
                _flat("rain", events={"storm": {"multiplier": 2.1}}),
                _flat("temp", events={"storm": {"add": -4.0}}),
            ],
        )
    )
    df = run(config)
    assert df["rain"].to_numpy()[6] == pytest.approx(21.0)
    assert df["temp"].to_numpy()[6] == pytest.approx(6.0)


def test_a_series_can_opt_out_of_an_event():
    config = parse_config(
        _scenario(
            {"storm": {"at": [6]}},
            [
                _flat("rain", events={"storm": {"multiplier": 2.0}}),
                _flat("humidity"),
            ],
        )
    )
    df = run(config)
    assert df["rain"].to_numpy()[6] == pytest.approx(20.0)
    assert df["humidity"].to_numpy()[6] == pytest.approx(10.0)


def test_events_reach_a_count_series():
    config = parse_config(
        _scenario(
            {"outbreak": {"at": [10, 11, 12]}},
            [
                _flat("rain"),
                series_dict(
                    "cases",
                    counts={"population": 100_000},
                    depends_on=[{"series": "rain", "lag": 1}],
                    events={"outbreak": {"add": 3.0}},
                ),
            ],
            n_total=40,
        )
    )
    cases = run(config)["cases"].to_numpy()
    assert np.nanmean(cases[10:13]) > np.nanmean(cases[20:30])


# --------------------------------------------------------------- regionality


def test_an_event_fires_in_the_same_periods_at_every_location():
    """Regional by design: correlation across locations with a named cause."""
    config = parse_config(
        _scenario(
            {"storm": {"rate": 0.2}},
            [_flat("rain", events={"storm": {"multiplier": 5.0}})],
            locations=["north", "south"],
            n_total=60,
        )
    )
    df = run(config)
    north = df[df["location"] == "north"]["rain"].to_numpy()
    south = df[df["location"] == "south"]["rain"].to_numpy()
    # Same periods struck in both, since the draw is shared.
    np.testing.assert_array_equal(north > 10.0, south > 10.0)


def test_a_rate_event_actually_fires():
    config = parse_config(
        _scenario(
            {"storm": {"rate": 0.5}},
            [_flat("rain", events={"storm": {"multiplier": 5.0}})],
            n_total=200,
        )
    )
    rain = run(config)["rain"].to_numpy()
    struck = np.sum(rain > 10.0)
    assert 50 < struck < 150  # ~100 expected, wide band for the draw


def test_rate_event_draws_are_reproducible():
    config = parse_config(
        _scenario(
            {"storm": {"rate": 0.3}},
            [_flat("rain", events={"storm": {"multiplier": 5.0}})],
            n_total=100,
        )
    )
    np.testing.assert_array_equal(
        run(config)["rain"].to_numpy(), run(config)["rain"].to_numpy()
    )


def test_a_different_seed_moves_a_rate_event():
    def struck(seed):
        config = parse_config(
            _scenario(
                {"storm": {"rate": 0.3}},
                [_flat("rain", events={"storm": {"multiplier": 5.0}})],
                n_total=100,
                seed=seed,
            )
        )
        return run(config)["rain"].to_numpy() > 10.0

    assert not np.array_equal(struck(1), struck(2))


def test_explicit_periods_need_no_seed_stability():
    """`at:` is fully deterministic: the same periods whatever the seed."""
    def struck(seed):
        config = parse_config(
            _scenario(
                {"storm": {"at": [3, 11]}},
                [_flat("rain", events={"storm": {"multiplier": 5.0}})],
                seed=seed,
            )
        )
        return run(config)["rain"].to_numpy() > 10.0

    np.testing.assert_array_equal(struck(1), struck(99))


# --------------------------------------------------------------- interaction


def test_an_event_propagates_through_a_dependency():
    """A struck parent carries the shock into its children."""
    config = parse_config(
        _scenario(
            {"storm": {"at": [5]}},
            [
                _flat("rain", events={"storm": {"multiplier": 10.0}}),
                series_dict("dam", depends_on=[{"series": "rain", "lag": 1}]),
            ],
        )
    )
    dam = run(config)["dam"].to_numpy()
    assert dam[6] > np.nanmean(dam[10:20])


def test_events_apply_before_dependencies_read_the_parent():
    """Order matters: a child must see its parent's post-event values."""
    without = parse_config(
        _scenario(
            {"storm": {"at": [5]}},
            [
                _flat("rain"),
                series_dict("dam", depends_on=[{"series": "rain", "lag": 0}]),
            ],
        )
    )
    with_event = parse_config(
        _scenario(
            {"storm": {"at": [5]}},
            [
                _flat("rain", events={"storm": {"multiplier": 10.0}}),
                series_dict("dam", depends_on=[{"series": "rain", "lag": 0}]),
            ],
        )
    )
    assert not np.allclose(
        run(without)["dam"].to_numpy(), run(with_event)["dam"].to_numpy()
    )
