"""Tests for CHAP-format CSV output and the optional train/test split."""
import pandas as pd

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
from dsl.core.pipeline.output import write_output
from tests.conftest import scenario_dict


def make_config(train_fraction=None):
    overrides = {} if train_fraction is None else {"train_fraction": train_fraction}
    return parse_config(scenario_dict(**overrides))


def test_writes_only_full_csv_without_train_fraction(tmp_path):
    config = make_config()
    write_output(run(config), config, tmp_path)
    assert (tmp_path / "simulated_data.csv").is_file()
    assert not (tmp_path / "train.csv").exists()
    assert not (tmp_path / "test.csv").exists()


def test_full_csv_has_chap_columns_and_no_index(tmp_path):
    config = make_config()
    write_output(run(config), config, tmp_path)
    # Read the raw header line: an index column would show up as a leading
    # empty name, which CHAP would choke on.
    header = (tmp_path / "simulated_data.csv").read_text().splitlines()[0]
    assert header == (
        "time_period,location,rainfall,mean_temperature,disease_cases,population"
    )


def test_full_csv_round_trips(tmp_path):
    config = make_config()
    df = run(config)
    write_output(df, config, tmp_path)
    loaded = pd.read_csv(tmp_path / "simulated_data.csv")
    assert len(loaded) == 78
    assert list(loaded.columns) == list(df.columns)
    # disease_cases is present for the whole series apart from the blanked
    # warm-up — CHAP does its own train/test hiding.
    assert loaded["disease_cases"].iloc[:3].isna().all()
    assert loaded["disease_cases"].iloc[3:].notna().all()


def test_train_fraction_also_writes_split(tmp_path):
    config = make_config(train_fraction=0.8)
    write_output(run(config), config, tmp_path)
    assert (tmp_path / "simulated_data.csv").is_file()
    train = pd.read_csv(tmp_path / "train.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    # floor(78 * 0.8) = 62 train rows, the remaining 16 are test.
    assert len(train) == 62
    assert len(test) == 16


def test_split_keeps_all_columns_and_is_a_row_slice(tmp_path):
    config = make_config(train_fraction=0.8)
    df = run(config)
    write_output(df, config, tmp_path)
    train = pd.read_csv(tmp_path / "train.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    full = pd.read_csv(tmp_path / "simulated_data.csv")
    assert list(train.columns) == list(full.columns)
    assert list(test.columns) == list(full.columns)
    # train + test stacked back together must equal the full file.
    recombined = pd.concat([train, test], ignore_index=True)
    assert recombined.equals(full)


def test_output_dir_is_created_if_missing(tmp_path):
    config = make_config()
    out = tmp_path / "does" / "not" / "exist"
    write_output(run(config), config, out)
    assert (out / "simulated_data.csv").is_file()


def test_multi_location_split_is_per_location(tmp_path):
    # With several locations, a plain row split would put whole locations in
    # train and others in test. The split must instead take the first
    # floor(n_total * fraction) periods of EACH location.
    config = parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "seed": 1,
            "train_fraction": 0.75,
            "locations": ["oslo", "bergen"],
            "variables": [{"name": "rainfall", "generate": "seasonal_spike"}],
            "disease_cases": {
                "population": 10_000,
                "depends_on": [{"variable": "rainfall", "lag": 2}],
            },
        }
    )
    write_output(run(config), config, tmp_path)
    train = pd.read_csv(tmp_path / "train.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    # floor(24 * 0.75) = 18 periods per location.
    for loc in ("oslo", "bergen"):
        assert len(train[train["location"] == loc]) == 18
        assert len(test[test["location"] == loc]) == 6
    # Test rows continue where train rows stopped, for every location.
    assert (train[train["location"] == "oslo"]["time_period"].iloc[-1]) == "2001-06"
    assert (test[test["location"] == "oslo"]["time_period"].iloc[0]) == "2001-07"


def test_rerun_without_split_removes_stale_train_test(tmp_path):
    # Bug #25: a directory first used with train_fraction, then reused without
    # it, must not keep the old train.csv/test.csv beside the new full dataset.
    split = make_config(train_fraction=0.8)
    write_output(run(split), split, tmp_path)
    assert (tmp_path / "train.csv").is_file()

    plain = make_config()  # no train_fraction, same dir
    write_output(run(plain), plain, tmp_path)
    assert not (tmp_path / "train.csv").exists()
    assert not (tmp_path / "test.csv").exists()
    assert (tmp_path / "simulated_data.csv").is_file()
