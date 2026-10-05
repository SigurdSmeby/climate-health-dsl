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
                    counts={},
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
            locations={"north": {"population": 100_000}, "south": {"population": 100_000}},
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


# ------------------------------------------------- one shock per fold


def _split_scenario(event, **overrides):
    return make_config_dict(
        events={"storm": event},
        split={"kind": "time", "k": 4},
        n_total=120,
        period="monthly",
        series=[
            _flat("rain", events={"storm": {"multiplier": 3.0}}),
            series_dict(
                "cases", counts={}, depends_on=[{"series": "rain", "lag": 1}]
            ),
        ],
        **overrides,
    )


def test_per_fold_puts_one_event_in_every_test_half():
    """Guaranteeing a shock per fold by hand means computing the fold
    boundaries yourself, and redoing it whenever k or n_total changes."""
    from dsl.core.pipeline.engine import run as run_engine
    from dsl.core.pipeline.folds import build_report

    config = parse_config(_split_scenario({"per_fold": 1}))
    report = build_report(config, run_engine(config))
    for fold in report["folds"]:
        entry = fold["test"]["events"].get("rain", {}).get("storm")
        assert entry and entry["count"] == 1, fold["index"]
    assert report["warnings"] == []


def test_per_fold_adapts_when_k_changes():
    """The whole point: the periods are derived, not written down."""
    from dsl.core.pipeline.engine import run as run_engine
    from dsl.core.pipeline.folds import build_report

    for k in (2, 3, 5, 6):
        data = _split_scenario({"per_fold": 1})
        data["split"]["k"] = k
        config = parse_config(data)
        report = build_report(config, run_engine(config))
        assert len(report["folds"]) == k
        assert report["warnings"] == [], (k, report["warnings"])


def test_per_fold_can_place_several():
    config = parse_config(_split_scenario({"per_fold": 2}))
    assert len(config.events["storm"].at or []) == 8  # 2 per fold, 4 folds


def test_per_fold_needs_a_split():
    with pytest.raises(ValidationError, match="split"):
        parse_config(
            make_config_dict(
                events={"storm": {"per_fold": 1}},
                series=[_flat("rain", events={"storm": {"multiplier": 2.0}})],
            )
        )


def test_per_fold_rejects_a_location_split():
    """A location split has no period boundaries to place a shock between."""
    data = _split_scenario(
        {"per_fold": 1},
        locations={"a": {"population": 1000}, "b": {"population": 1000}},
    )
    data["split"] = {"kind": "location"}
    with pytest.raises(ValidationError, match="per_fold"):
        parse_config(data)


def test_per_fold_excludes_rate_and_at():
    with pytest.raises(ValidationError, match="per_fold"):
        parse_config(_split_scenario({"per_fold": 1, "at": [3]}))


# ------------------------------------------------- variable effect strength


def _strength_scenario(effect, n_total=200):
    return make_config_dict(
        events={"storm": {"rate": 0.3}},
        n_total=n_total,
        period="monthly",
        series=[
            _flat("rain", events={"storm": effect}),
            series_dict(
                "cases", counts={}, depends_on=[{"series": "rain", "lag": 1}]
            ),
        ],
    )


def test_a_fixed_multiplier_is_identical_every_time():
    """The baseline: every storm scales by exactly the same factor, so a model
    could memorise the constant rather than learn the mechanism."""
    config = parse_config(_strength_scenario({"multiplier": 2.5}))
    rain = run(config)["rain"].to_numpy()
    struck = rain[rain > 10.0]
    assert struck.size > 10
    assert np.allclose(struck, 25.0)


def test_a_multiplier_range_varies_each_event():
    config = parse_config(
        _strength_scenario({"multiplier": {"min": 2.0, "max": 4.0}})
    )
    rain = run(config)["rain"].to_numpy()
    struck = rain[rain > 10.0]
    assert struck.size > 10
    assert np.unique(struck).size > 5, "every storm got the same factor"
    # A level of 10 scaled by 2-4 lands in 20-40.
    assert struck.min() >= 20.0 - 1e-9
    assert struck.max() <= 40.0 + 1e-9


def test_an_add_range_varies_each_event():
    config = parse_config(
        _strength_scenario({"add": {"min": -8.0, "max": -3.0}})
    )
    rain = run(config)["rain"].to_numpy()
    struck = rain[rain < 10.0]
    assert struck.size > 10
    assert np.unique(struck).size > 5
    assert struck.min() >= 2.0 - 1e-9  # 10 - 8
    assert struck.max() <= 7.0 + 1e-9  # 10 - 3


def test_a_strength_range_is_reproducible():
    config = parse_config(
        _strength_scenario({"multiplier": {"min": 2.0, "max": 4.0}})
    )
    np.testing.assert_array_equal(
        run(config)["rain"].to_numpy(), run(config)["rain"].to_numpy()
    )


def test_a_range_needs_min_below_max():
    with pytest.raises(ValidationError, match="min"):
        parse_config(_strength_scenario({"multiplier": {"min": 4.0, "max": 2.0}}))


def test_a_range_rejects_a_missing_bound():
    with pytest.raises(ValidationError, match="max"):
        parse_config(_strength_scenario({"multiplier": {"min": 2.0}}))


# ------------------------------------------------- a count per fold


def test_per_fold_accepts_a_count_per_fold():
    """Giving one fold more than the others is a deliberate experiment — you
    know fold 2 has two, so a score difference there is interpretable."""
    from dsl.core.pipeline.engine import run as run_engine
    from dsl.core.pipeline.folds import build_report

    config = parse_config(_split_scenario({"per_fold": [1, 1, 2, 1]}))
    report = build_report(config, run_engine(config))
    assert [f["summary"]["storm"]["test"] for f in report["folds"]] == [1, 1, 2, 1]


def test_a_per_fold_list_must_match_the_fold_count():
    """Otherwise changing k silently leaves some folds unspecified."""
    with pytest.raises(ValidationError, match="per_fold"):
        parse_config(_split_scenario({"per_fold": [1, 1, 2]}))  # k is 4


def test_a_per_fold_list_may_hold_zero():
    """Leaving one fold empty is a legitimate thing to test. The FOLD REPORT
    is what warns about it — validate_scenario runs before generation and
    does not know what landed where."""
    from dsl.core.pipeline.engine import run as run_engine
    from dsl.core.pipeline.folds import build_report

    config = parse_config(_split_scenario({"per_fold": [1, 0, 1, 1]}))
    assert len(config.events["storm"].at) == 3
    report = build_report(config, run_engine(config))
    assert [f["summary"]["storm"]["test"] for f in report["folds"]] == [1, 0, 1, 1]
    assert any("fold 1" in w and "storm" in w for w in report["warnings"])


def test_a_per_fold_list_rejects_negative_counts():
    with pytest.raises(ValidationError, match="per_fold"):
        parse_config(_split_scenario({"per_fold": [1, -1, 1, 1]}))


def test_a_plain_per_fold_number_still_works():
    config = parse_config(_split_scenario({"per_fold": 2}))
    assert len(config.events["storm"].at) == 8  # 2 per fold, 4 folds
