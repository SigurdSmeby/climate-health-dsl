"""Tests for the output CSV and the fold files."""
import pandas as pd

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
from dsl.core.pipeline.output import write_output
from tests.conftest import scenario_dict


def make_config(**overrides):
    return parse_config(scenario_dict(**overrides))


def test_writes_only_full_csv_without_train_fraction(tmp_path):
    config = make_config()
    write_output(run(config), config, tmp_path)
    assert (tmp_path / "simulated_data.csv").is_file()
    assert not (tmp_path / "train.csv").exists()
    assert not (tmp_path / "test.csv").exists()


def test_full_csv_has_required_columns_and_no_index(tmp_path):
    config = make_config()
    write_output(run(config), config, tmp_path)
    # Read the raw header line: an index column would show up as a leading
    # empty name, which a consumer would choke on.
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
    # warm-up — a consumer does its own train/test hiding.
    assert loaded["disease_cases"].iloc[:3].isna().all()
    assert loaded["disease_cases"].iloc[3:].notna().all()


def test_output_dir_is_created_if_missing(tmp_path):
    config = make_config()
    out = tmp_path / "does" / "not" / "exist"
    write_output(run(config), config, out)
    assert (out / "simulated_data.csv").is_file()