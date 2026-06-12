"""Smoke tests for the `dsl run` command-line interface."""
import pandas as pd
import yaml

from dsl.cli import main
from tests.conftest import scenario_dict as base_scenario

EXAMPLE = "examples/basic_scenario.yaml"


def write_scenario(tmp_path, data):
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_example_scenario_runs_and_writes_files(tmp_path, capsys):
    out = tmp_path / "out"
    code = main(["run", EXAMPLE, "-o", str(out)])
    assert code == 0
    # The example sets train_fraction: 0.8, so all three files appear.
    assert (out / "simulated_data.csv").is_file()
    assert (out / "train.csv").is_file()
    assert (out / "test.csv").is_file()


def test_orphan_variable_warns_but_succeeds(tmp_path, capsys):
    data = base_scenario()
    data["variables"].append({"name": "wind", "generate": "seasonal_smooth"})
    path = write_scenario(tmp_path, data)
    out = tmp_path / "out"
    code = main(["run", str(path), "-o", str(out)])
    assert code == 0
    assert (out / "simulated_data.csv").is_file()
    captured = capsys.readouterr()
    # Warnings go to stderr, prefixed, and mention the orphan variable.
    assert "warning:" in captured.err
    assert "wind" in captured.err


def test_dangling_reference_fails_before_writing(tmp_path, capsys):
    data = base_scenario()
    data["disease_cases"]["depends_on"] = [{"variable": "rainfal", "lag": 3}]
    path = write_scenario(tmp_path, data)
    out = tmp_path / "out"
    code = main(["run", str(path), "-o", str(out)])
    assert code != 0
    # A hard error must stop the run before anything is generated.
    assert not out.exists()
    captured = capsys.readouterr()
    assert "rainfal" in captured.err


def test_missing_file_fails_cleanly(tmp_path, capsys):
    code = main(["run", str(tmp_path / "nope.yaml"), "-o", str(tmp_path / "out")])
    assert code != 0
    assert "nope.yaml" in capsys.readouterr().err


def nan_covariate_scenario(tmp_path):
    """A scenario whose covariate carries a NaN (a real CHAP finding).

    A from_csv covariate reads a column containing a NaN, which the CHAP
    check flags — CHAP requires complete covariates. Engine output is
    otherwise always CHAP-valid, so this is how we exercise the strict path.
    """
    csv = tmp_path / "real.csv"
    periods = [f"2010-{m + 1:02d}" for m in range(12)]
    rain = [1.0] * 12
    rain[4] = float("nan")  # one missing covariate value
    pd.DataFrame({"time_period": periods, "rainfall": rain}).to_csv(csv, index=False)
    return {
        "period": "monthly",
        "n_total": 12,
        "start_period": "2010-01",
        "variables": [
            {
                "name": "rainfall",
                "generate": "from_csv",
                "params": {"file": str(csv), "column": "rainfall"},
            }
        ],
        "disease_cases": {
            "population": 1000,
            "depends_on": [{"variable": "rainfall", "lag": 1}],
        },
    }


def test_chap_finding_warns_but_succeeds(tmp_path, capsys):
    # A CHAP-compatibility finding (here: a NaN in real covariate data) is
    # advisory — it prints a warning but the run still writes output.
    path = write_scenario(tmp_path, nan_covariate_scenario(tmp_path))
    out = tmp_path / "out"
    code = main(["run", str(path), "-o", str(out)])
    assert code == 0
    assert (out / "simulated_data.csv").is_file()
    assert "rainfall" in capsys.readouterr().err  # CHAP finding, as warning


def test_relative_from_csv_path_resolves_to_scenario_dir(tmp_path, monkeypatch):
    # Bug #29: a from_csv path relative to the scenario file must work even
    # when dsl is launched from a different directory.
    exp = tmp_path / "experiment"
    exp.mkdir()
    pd.DataFrame(
        {"time_period": [f"2010-{m:02d}" for m in range(1, 7)],
         "rainfall": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    ).to_csv(exp / "data.csv", index=False)
    scenario = {
        "period": "monthly", "n_total": 6, "start_period": "2010-01",
        "variables": [{"name": "rainfall", "generate": "from_csv",
                       "params": {"file": "data.csv", "column": "rainfall"}}],
        "disease_cases": {"population": 100, "depends_on": [{"variable": "rainfall"}]},
    }
    (exp / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    # Launch from a DIFFERENT directory.
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    code = main(["run", str(exp / "scenario.yaml"), "-o", str(out)])
    assert code == 0
    assert (out / "simulated_data.csv").is_file()


def test_generation_error_is_clean_cli_error(tmp_path, capsys):
    # Bug #19: an invalid generator param surfaces during generation, after
    # the schema passes. The CLI must print 'error:' and exit 1, not dump a
    # traceback.
    data = {
        "period": "monthly", "n_total": 3,
        "variables": [
            {"name": "x", "generate": "seasonal_spike", "params": {"bogus": 1}}
        ],
        "disease_cases": {"population": 100, "depends_on": [{"variable": "x"}]},
    }
    path = write_scenario(tmp_path, data)
    out = tmp_path / "out"
    code = main(["run", str(path), "-o", str(out)])
    assert code == 1
    assert "error:" in capsys.readouterr().err
    assert not (out / "simulated_data.csv").exists()


def test_daily_scenario_has_no_chap_warning(tmp_path, capsys):
    # Daily output is valid CHAP (TimePeriod.parse accepts YYYYMMDD); it must
    # not produce a CHAP period-format warning.
    data = {
        "period": "daily",
        "n_total": 400,
        "variables": [
            {"name": "rainfall", "generate": "seasonal_spike"},
            {"name": "mean_temperature", "generate": "seasonal_smooth"},
        ],
        "disease_cases": {
            "population": 1000,
            "depends_on": [
                {"variable": "rainfall", "lag": 1},
                {"variable": "mean_temperature", "lag": 1},
            ],
        },
    }
    path = write_scenario(tmp_path, data)
    out = tmp_path / "out"
    code = main(["run", str(path), "-o", str(out)])
    assert code == 0
    assert "time_period" not in capsys.readouterr().err  # no format complaint


def test_metadata_sidecar_written(tmp_path):
    out = tmp_path / "out"
    code = main(["run", EXAMPLE, "-o", str(out)])
    assert code == 0
    assert (out / "metadata.json").is_file()


def test_plot_flag_writes_plot(tmp_path):
    out = tmp_path / "out"
    code = main(["run", EXAMPLE, "-o", str(out), "--plot"])
    assert code == 0
    assert (out / "plot.html").is_file()


def test_plot_format_png(tmp_path):
    out = tmp_path / "out"
    code = main(["run", EXAMPLE, "-o", str(out), "--plot", "--plot-format", "png"])
    assert code == 0
    assert (out / "plot.png").is_file()


def test_no_plot_by_default(tmp_path):
    out = tmp_path / "out"
    main(["run", EXAMPLE, "-o", str(out)])
    assert not (out / "plot.html").exists()


def test_reproduce_from_metadata(tmp_path):
    # Run a scenario, then re-run pointing at the produced metadata.json;
    # the regenerated dataset must be byte-identical.
    out_a = tmp_path / "a"
    assert main(["run", EXAMPLE, "-o", str(out_a)]) == 0
    out_b = tmp_path / "b"
    assert main(["run", str(out_a / "metadata.json"), "-o", str(out_b)]) == 0
    a = (out_a / "simulated_data.csv").read_text()
    b = (out_b / "simulated_data.csv").read_text()
    assert a == b


def test_no_out_dir_creates_named_folder_then_numbers(tmp_path, monkeypatch):
    # Without -o, write into out/<scenario-stem>/; a second run must not
    # overwrite it but go to out/<stem>_1/.
    monkeypatch.chdir(tmp_path)  # so out/ lands in the temp dir, not the repo
    scenario = write_scenario(tmp_path, base_scenario())  # stem "scenario"

    assert main(["run", str(scenario)]) == 0
    assert (tmp_path / "out" / "scenario" / "simulated_data.csv").is_file()

    assert main(["run", str(scenario)]) == 0
    assert (tmp_path / "out" / "scenario_1" / "simulated_data.csv").is_file()
    # The first folder is untouched.
    assert (tmp_path / "out" / "scenario" / "simulated_data.csv").is_file()

    assert main(["run", str(scenario)]) == 0
    assert (tmp_path / "out" / "scenario_2" / "simulated_data.csv").is_file()


def test_explicit_out_dir_used_directly(tmp_path):
    out = tmp_path / "exact"
    assert main(["run", EXAMPLE, "-o", str(out)]) == 0
    # Files land directly in the given dir, no auto-subfolder.
    assert (out / "simulated_data.csv").is_file()
    assert not (out / "basic_scenario").exists()


def test_reproduce_without_out_dir_uses_source_folder_name(tmp_path, monkeypatch):
    # Reproducing from out/foo/metadata.json with no -o should derive the
    # folder name from the parent ("foo"), not the file stem ("metadata").
    monkeypatch.chdir(tmp_path)
    scenario = write_scenario(tmp_path, base_scenario())
    main(["run", str(scenario)])  # → out/scenario/
    meta = tmp_path / "out" / "scenario" / "metadata.json"
    assert main(["run", str(meta)]) == 0
    assert (tmp_path / "out" / "scenario_1" / "simulated_data.csv").is_file()


def test_output_is_reproducible(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert main(["run", EXAMPLE, "-o", str(out_a)]) == 0
    assert main(["run", EXAMPLE, "-o", str(out_b)]) == 0
    a = (out_a / "simulated_data.csv").read_text()
    b = (out_b / "simulated_data.csv").read_text()
    assert a == b  # same seed → byte-identical files
