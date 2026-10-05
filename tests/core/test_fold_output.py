"""Tests for writing a scenario's folds to disk.

Each fold becomes a directory holding its own train.csv and test.csv, ready to
hand to a model without any further slicing. A scenario with no ``split:``
writes only the full dataset.
"""
import pandas as pd
import pytest

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
from dsl.core.pipeline.output import write_output
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict


def _write(tmp_path, split=None, **overrides):
    """Run a scenario and write it, returning the output directory."""
    data = make_config_dict(**overrides)
    if split is not None:
        data["split"] = split
    config = parse_config(data)
    write_output(run(config), config, tmp_path)
    return tmp_path


def _locations(*names):
    return {name: {"population": 100_000} for name in names}


def _read(path):
    return pd.read_csv(path)


# ------------------------------------------------------------- without split


def test_full_dataset_is_always_written(tmp_path):
    out = _write(tmp_path)
    assert (out / "simulated_data.csv").is_file()


def test_no_split_writes_no_folds(tmp_path):
    out = _write(tmp_path)
    assert not (out / "folds").exists()


def test_stale_folds_are_cleared(tmp_path):
    """A rerun without a split must not leave the previous run's folds."""
    _write(tmp_path, split={"kind": "time", "k": 3})
    assert (tmp_path / "folds").is_dir()
    _write(tmp_path)
    assert not (tmp_path / "folds").exists()


# ---------------------------------------------------------------- the layout


def test_one_directory_per_fold(tmp_path):
    out = _write(tmp_path, split={"kind": "time", "k": 3}, n_total=60)
    assert sorted(p.name for p in (out / "folds").iterdir()) == [
        "fold_0",
        "fold_1",
        "fold_2",
    ]


def test_each_fold_holds_a_train_and_a_test_file(tmp_path):
    out = _write(tmp_path, split={"kind": "time", "k": 2}, n_total=60)
    for index in range(2):
        fold = out / "folds" / f"fold_{index}"
        assert (fold / "train.csv").is_file()
        assert (fold / "test.csv").is_file()


def test_fold_directories_are_zero_padded_in_order(tmp_path):
    out = _write(tmp_path, split={"kind": "time", "k": 12}, n_total=120)
    names = sorted(p.name for p in (out / "folds").iterdir())
    assert names[:2] == ["fold_00", "fold_01"]
    assert names[-1] == "fold_11"


def test_stale_fold_directories_from_a_larger_k_are_cleared(tmp_path):
    _write(tmp_path, split={"kind": "time", "k": 5}, n_total=60)
    _write(tmp_path, split={"kind": "time", "k": 2}, n_total=60)
    assert len(list((tmp_path / "folds").iterdir())) == 2


# ------------------------------------------------------------- time contents


def test_time_fold_train_precedes_its_test(tmp_path):
    out = _write(tmp_path, split={"kind": "time", "k": 3}, n_total=60)
    fold = out / "folds" / "fold_1"
    train, test = _read(fold / "train.csv"), _read(fold / "test.csv")
    assert train["time_period"].max() < test["time_period"].min()


def test_time_fold_rows_match_the_declared_periods(tmp_path):
    data = make_config_dict(n_total=60, split={"kind": "time", "k": 3})
    config = parse_config(data)
    write_output(run(config), config, tmp_path)
    for fold in config.folds():
        directory = tmp_path / "folds" / f"fold_{fold.index}"
        assert len(_read(directory / "train.csv")) == len(fold.train_periods)
        assert len(_read(directory / "test.csv")) == len(fold.test_periods)


def test_time_fold_keeps_every_location_on_both_sides(tmp_path):
    out = _write(
        tmp_path,
        split={"kind": "time", "k": 2},
        locations=_locations("north", "south"),
        n_total=60,
    )
    fold = out / "folds" / "fold_0"
    for name in ("train.csv", "test.csv"):
        assert set(_read(fold / name)["location"]) == {"north", "south"}


def test_time_fold_splits_each_location_at_the_same_period(tmp_path):
    """A time split must cut the calendar, not hand whole locations over."""
    out = _write(
        tmp_path,
        split={"kind": "time", "k": 2},
        locations=_locations("north", "south"),
        n_total=60,
    )
    train = _read(out / "folds" / "fold_0" / "train.csv")
    per_location = train.groupby("location")["time_period"].agg(["min", "max"])
    assert per_location["min"].nunique() == 1
    assert per_location["max"].nunique() == 1


# --------------------------------------------------------- location contents


def test_location_fold_holds_its_location_out(tmp_path):
    out = _write(
        tmp_path,
        split={"kind": "location"},
        locations=_locations("north", "south", "east"),
        n_total=24,
    )
    fold = out / "folds" / "fold_0"
    assert set(_read(fold / "test.csv")["location"]) == {"north"}
    assert set(_read(fold / "train.csv")["location"]) == {"south", "east"}


def test_location_fold_keeps_every_period_on_both_sides(tmp_path):
    out = _write(
        tmp_path,
        split={"kind": "location"},
        locations=_locations("north", "south"),
        n_total=24,
    )
    fold = out / "folds" / "fold_0"
    train, test = _read(fold / "train.csv"), _read(fold / "test.csv")
    assert train["time_period"].nunique() == 24
    assert test["time_period"].nunique() == 24


def test_location_folds_hold_each_location_out_once(tmp_path):
    out = _write(
        tmp_path,
        split={"kind": "location"},
        locations=_locations("north", "south", "east"),
        n_total=24,
    )
    held = [
        set(_read(out / "folds" / f"fold_{i}" / "test.csv")["location"])
        for i in range(3)
    ]
    assert held == [{"north"}, {"south"}, {"east"}]


# ------------------------------------------------------------------ contents


def test_fold_files_carry_every_column(tmp_path):
    out = _write(tmp_path, split={"kind": "time", "k": 2}, n_total=60)
    full = _read(out / "simulated_data.csv")
    train = _read(out / "folds" / "fold_0" / "train.csv")
    assert list(train.columns) == list(full.columns)


def test_fold_rows_are_unchanged_from_the_full_dataset(tmp_path):
    """Splitting selects rows; it must not regenerate or alter them."""
    out = _write(tmp_path, split={"kind": "time", "k": 2}, n_total=60)
    full = _read(out / "simulated_data.csv")
    train = _read(out / "folds" / "fold_0" / "train.csv")
    merged = train.merge(full, on=["time_period", "location"], suffixes=("", "_full"))
    assert (merged["rainfall"] == merged["rainfall_full"]).all()


def test_a_holdout_is_just_one_fold(tmp_path):
    out = _write(tmp_path, split={"kind": "time", "k": 1}, n_total=60)
    assert [p.name for p in (out / "folds").iterdir()] == ["fold_0"]


def test_top_level_train_test_files_are_not_written(tmp_path):
    """Folds are the one mechanism; there is no second set of split files."""
    out = _write(tmp_path, split={"kind": "time", "k": 2}, n_total=60)
    assert not (out / "train.csv").exists()
    assert not (out / "test.csv").exists()


def test_train_fraction_is_no_longer_accepted():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="train_fraction"):
        parse_config(make_config_dict(train_fraction=0.8))


def test_a_scenario_with_no_counts_still_writes_folds(tmp_path):
    out = _write(
        tmp_path,
        split={"kind": "time", "k": 2},
        locations=["loc"],
        series=[series_dict("rain", generate="flat")],
    )
    assert (out / "folds" / "fold_0" / "train.csv").is_file()


def test_plot_marker_is_the_first_test_period_not_a_train_count(tmp_path):
    """The marker shows where evaluation begins. A train-period COUNT only
    coincides with that for an expanding split; for blocked it lands in the
    wrong place, and for a location split there is no period boundary."""
    from dsl.core.pipeline.folds import evaluation_boundary

    expanding = parse_config(
        make_config_dict(n_total=48, split={"kind": "time", "k": 4})
    )
    assert evaluation_boundary(expanding) == expanding.folds()[0].test_periods[0]

    blocked = parse_config(
        make_config_dict(
            n_total=48, split={"kind": "time", "k": 4, "scheme": "blocked"}
        )
    )
    # Fold 0 tests periods 0-11 while training on 12-47, so a train count
    # (36) would put the marker nowhere near the boundary.
    assert evaluation_boundary(blocked) == 0

    location = parse_config(
        make_config_dict(
            n_total=24,
            locations={"a": {"population": 1000}, "b": {"population": 1000}},
            split={"kind": "location"},
        )
    )
    # Holding a location out has no period boundary to draw.
    assert evaluation_boundary(location) is None

    assert evaluation_boundary(parse_config(make_config_dict())) is None
