"""Tests for the ``split:`` block — how a dataset is divided for evaluation.

Two kinds. A time split cuts the periods, so a model is asked to forecast
forward; a location split holds whole places out, so it is asked to generalise
sideways. The schema decides the fold boundaries here; writing them to disk
comes later.
"""
import pytest
from pydantic import ValidationError

from dsl.core.config.schema import parse_config
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict


def _scenario(split, **overrides):
    return make_config_dict(split=split, **overrides)


def _locations(*names):
    return {name: {"population": 100_000} for name in names}


# ---------------------------------------------------------------- the schema


def test_split_is_optional():
    assert parse_config(make_config_dict()).split is None


def test_time_split_parses():
    split = parse_config(_scenario({"kind": "time", "k": 5})).split
    assert split.kind == "time"
    assert split.k == 5


def test_scheme_defaults_to_expanding():
    """The forecasting default: never train on the future."""
    assert parse_config(_scenario({"kind": "time", "k": 3})).split.scheme == (
        "expanding"
    )


def test_blocked_scheme_parses():
    split = parse_config(
        _scenario({"kind": "time", "k": 3, "scheme": "blocked"})
    ).split
    assert split.scheme == "blocked"


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError, match="kind"):
        parse_config(_scenario({"kind": "sideways", "k": 3}))


def test_unknown_scheme_rejected():
    with pytest.raises(ValidationError, match="scheme"):
        parse_config(_scenario({"kind": "time", "k": 3, "scheme": "random"}))


def test_k_below_one_rejected():
    with pytest.raises(ValidationError, match="k"):
        parse_config(_scenario({"kind": "time", "k": 0}))


def test_k_of_one_is_a_plain_holdout():
    """One fold is the single train/test split, not a special case."""
    folds = parse_config(_scenario({"kind": "time", "k": 1})).folds()
    assert len(folds) == 1


def test_scheme_rejected_on_a_location_split():
    """A scheme orders periods; holding a location out has no ordering."""
    with pytest.raises(ValidationError, match="scheme"):
        parse_config(
            _scenario(
                {"kind": "location", "scheme": "blocked"},
                locations=_locations("north", "south"),
            )
        )


# ------------------------------------------------------------- time, expanding


def test_expanding_folds_grow_the_training_set():
    folds = parse_config(
        _scenario({"kind": "time", "k": 4}, n_total=100)
    ).folds()
    sizes = [len(f.train_periods) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_expanding_train_always_precedes_its_test():
    folds = parse_config(
        _scenario({"kind": "time", "k": 4}, n_total=100)
    ).folds()
    for fold in folds:
        assert max(fold.train_periods) < min(fold.test_periods)


def test_expanding_test_blocks_do_not_overlap():
    folds = parse_config(
        _scenario({"kind": "time", "k": 4}, n_total=100)
    ).folds()
    seen: set[int] = set()
    for fold in folds:
        assert not seen & set(fold.test_periods)
        seen |= set(fold.test_periods)


def test_every_fold_has_a_non_empty_train_and_test():
    folds = parse_config(
        _scenario({"kind": "time", "k": 5}, n_total=37)
    ).folds()
    assert all(f.train_periods and f.test_periods for f in folds)


def test_min_train_sets_a_floor_on_the_first_fold():
    folds = parse_config(
        _scenario({"kind": "time", "k": 3, "min_train": 40}, n_total=100)
    ).folds()
    assert len(folds[0].train_periods) >= 40


def test_min_train_larger_than_the_series_rejected():
    with pytest.raises(ValidationError, match="min_train"):
        parse_config(
            _scenario({"kind": "time", "k": 3, "min_train": 500}, n_total=100)
        )


def test_too_many_time_folds_for_the_series_rejected():
    with pytest.raises(ValidationError, match="k"):
        parse_config(_scenario({"kind": "time", "k": 50}, n_total=10))


# -------------------------------------------------------------- time, blocked


def test_blocked_folds_cover_every_period_exactly_once():
    folds = parse_config(
        _scenario({"kind": "time", "k": 4, "scheme": "blocked"}, n_total=100)
    ).folds()
    covered: list[int] = []
    for fold in folds:
        covered.extend(fold.test_periods)
    assert sorted(covered) == list(range(100))


def test_blocked_train_is_everything_outside_the_test_block():
    folds = parse_config(
        _scenario({"kind": "time", "k": 4, "scheme": "blocked"}, n_total=100)
    ).folds()
    for fold in folds:
        assert set(fold.train_periods) | set(fold.test_periods) == set(range(100))


def test_blocked_scheme_warns_about_leaking_the_future():
    from dsl.core.config.schema import validate_scenario

    config = parse_config(
        _scenario({"kind": "time", "k": 4, "scheme": "blocked"}, n_total=100)
    )
    assert any("future" in w or "leak" in w for w in validate_scenario(config))


def test_expanding_scheme_does_not_warn():
    from dsl.core.config.schema import validate_scenario

    config = parse_config(_scenario({"kind": "time", "k": 4}, n_total=100))
    assert not any("leak" in w for w in validate_scenario(config))


# ------------------------------------------------------------------ location


def test_location_folds_hold_one_location_out_each():
    folds = parse_config(
        _scenario(
            {"kind": "location"}, locations=_locations("north", "south", "east")
        )
    ).folds()
    assert len(folds) == 3
    assert [f.test_locations for f in folds] == [["north"], ["south"], ["east"]]


def test_location_fold_trains_on_the_others():
    folds = parse_config(
        _scenario(
            {"kind": "location"}, locations=_locations("north", "south", "east")
        )
    ).folds()
    assert folds[0].train_locations == ["south", "east"]


def test_location_k_defaults_to_the_location_count():
    config = parse_config(
        _scenario({"kind": "location"}, locations=_locations("a", "b", "c", "d"))
    )
    assert len(config.folds()) == 4


def test_location_split_with_one_location_rejected():
    with pytest.raises(ValidationError, match="location"):
        parse_config(_scenario({"kind": "location"}))


def test_location_k_above_the_location_count_rejected():
    with pytest.raises(ValidationError, match="k"):
        parse_config(
            _scenario({"kind": "location", "k": 5}, locations=_locations("a", "b"))
        )


def test_fewer_location_folds_than_locations_groups_them():
    """k below the location count splits them into k groups."""
    folds = parse_config(
        _scenario({"kind": "location", "k": 2}, locations=_locations("a", "b", "c", "d"))
    ).folds()
    assert len(folds) == 2
    held_out = [loc for f in folds for loc in f.test_locations]
    assert sorted(held_out) == ["a", "b", "c", "d"]


def test_a_time_split_uses_every_location_on_both_sides():
    folds = parse_config(
        _scenario(
            {"kind": "time", "k": 3},
            locations=_locations("north", "south"),
            n_total=60,
        )
    ).folds()
    for fold in folds:
        assert fold.train_locations == ["north", "south"]
        assert fold.test_locations == ["north", "south"]


# ------------------------------------------------------------------ warm-up


def test_a_lag_covering_the_first_train_split_warns():
    """Every training target would be warm-up NaN — nothing to learn from."""
    from dsl.core.config.schema import validate_scenario

    data = _scenario({"kind": "time", "k": 5}, n_total=60)
    data["series"][2]["depends_on"][0]["lag"] = 11
    assert any("lag" in w for w in validate_scenario(parse_config(data)))


def test_a_short_lag_does_not_warn():
    from dsl.core.config.schema import validate_scenario

    data = _scenario({"kind": "time", "k": 5}, n_total=60)
    data["series"][2]["depends_on"][0]["lag"] = 1
    assert not any("lag" in w for w in validate_scenario(parse_config(data)))


def test_folds_are_none_without_a_split():
    assert parse_config(make_config_dict()).folds() == []


def test_a_scenario_with_no_counts_may_still_split():
    """Splitting is about rows, not about whether anything is counted."""
    config = parse_config(
        _scenario(
            {"kind": "time", "k": 2},
            locations=["loc"],
            series=[series_dict("rain", generate="flat")],
        )
    )
    assert len(config.folds()) == 2


# ------------------------------------------------- degenerate fold rejection


def test_k_equal_to_n_total_rejected():
    """Every period would be tested on, leaving fold 0 nothing to train on."""
    with pytest.raises(ValidationError, match="k"):
        parse_config(_scenario({"kind": "time", "k": 3}, n_total=3))


def test_k_leaving_no_reserved_training_block_rejected():
    """An expanding split reserves n_total // (k+1) periods; that must be >= 1."""
    with pytest.raises(ValidationError, match="k"):
        parse_config(_scenario({"kind": "time", "k": 12}, n_total=12))


def test_the_smallest_workable_expanding_split_is_accepted():
    """n_total = 4, k = 2 reserves 1 period, leaving 3 to test over."""
    folds = parse_config(
        _scenario({"kind": "time", "k": 2}, n_total=4)
    ).folds()
    assert all(f.train_periods and f.test_periods for f in folds)


def test_every_expanding_fold_has_something_to_train_on():
    for n_total, k in ((10, 3), (12, 5), (24, 7), (100, 20)):
        folds = parse_config(
            _scenario({"kind": "time", "k": k}, n_total=n_total)
        ).folds()
        assert all(f.train_periods for f in folds), (n_total, k)
        assert all(f.test_periods for f in folds), (n_total, k)


def test_location_split_holding_out_every_location_rejected():
    """k=1 over all locations leaves no location to train on."""
    with pytest.raises(ValidationError, match="k"):
        parse_config(
            _scenario(
                {"kind": "location", "k": 1}, locations=_locations("a", "b")
            )
        )


def test_location_split_keeps_a_training_location_in_every_fold():
    for names, k in ((("a", "b"), 2), (("a", "b", "c"), 2), (("a", "b", "c", "d"), 3)):
        folds = parse_config(
            _scenario({"kind": "location", "k": k}, locations=_locations(*names))
        ).folds()
        assert all(f.train_locations for f in folds), (names, k)
        assert all(f.test_locations for f in folds), (names, k)
