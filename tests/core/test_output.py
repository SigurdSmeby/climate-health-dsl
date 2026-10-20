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

def test_a_symlinked_metadata_names_the_folder_the_user_pointed_at(tmp_path):
    """resolve() follows symlinks, so a linked metadata.json named the run
    after the link's TARGET rather than the folder actually given."""
    import os

    from dsl._output import _run_name

    real = tmp_path / "real"
    real.mkdir()
    (real / "metadata.json").write_text("{}")
    link = tmp_path / "link"
    link.mkdir()
    os.symlink(real / "metadata.json", link / "metadata.json")

    assert _run_name(link / "metadata.json") == "link"


def test_a_users_own_folds_directory_is_not_deleted(tmp_path):
    """write_output cleared out_dir/folds unconditionally, so `-o .` on a
    working directory holding a folds/ destroyed it — with no split: block
    anywhere in the scenario. Same family as the --replicates data loss,
    on the single-run path."""
    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run
    from dsl.core.pipeline.output import write_output
    from tests.conftest import scenario_dict as make_config_dict
    from tests.conftest import series_dict

    mine = tmp_path / "folds"
    mine.mkdir()
    (mine / "important.txt").write_text("my work")

    config = parse_config(
        make_config_dict(
            series=[
                series_dict("r", generate="flat", params={"level": 5.0}),
                series_dict(
                    "disease_cases", counts={}, depends_on=[{"series": "r", "lag": 1}]
                ),
            ]
        )
    )
    write_output(run(config), config, tmp_path)

    assert (mine / "important.txt").read_text() == "my work"


def test_a_previous_runs_folds_are_still_cleared(tmp_path):
    """The guard must not stop a rerun with fewer folds from tidying up,
    or stale fold_N would be read as part of the new run."""
    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run
    from dsl.core.pipeline.output import write_output
    from tests.conftest import scenario_dict as make_config_dict
    from tests.conftest import series_dict

    def built(k):
        return parse_config(
            make_config_dict(
                n_total=48,
                period="monthly",
                split={"kind": "time", "k": k},
                series=[
                    series_dict("r", generate="flat", params={"level": 5.0}),
                    series_dict(
                        "disease_cases",
                        counts={},
                        depends_on=[{"series": "r", "lag": 1}],
                    ),
                ],
            )
        )

    five = built(5)
    write_output(run(five), five, tmp_path)
    assert len(list((tmp_path / "folds").glob("fold_*"))) == 5

    two = built(2)
    write_output(run(two), two, tmp_path)
    assert len(list((tmp_path / "folds").glob("fold_*"))) == 2
