"""Smoke tests for the `dsl run` command-line interface."""
import numpy as np
import pandas as pd
import pytest
import yaml

from dsl.cli import main

EXAMPLE = "examples/basic_scenario.yaml"


def write_scenario(tmp_path, data):
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def base_scenario():
    return {
        "period": "weekly",
        "n_total": 78,
        "seed": 42,
        "variables": [
            {"name": "rainfall", "generate": "seasonal_spike"},
            {"name": "mean_temperature", "generate": "seasonal_smooth"},
        ],
        "disease_cases": {
            "population": 100_000,
            "depends_on": [
                {"variable": "rainfall", "lag": 3, "weight": 2.0},
                {"variable": "mean_temperature", "lag": 3, "weight": 1.0},
            ],
        },
    }


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


def test_output_is_reproducible(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert main(["run", EXAMPLE, "-o", str(out_a)]) == 0
    assert main(["run", EXAMPLE, "-o", str(out_b)]) == 0
    a = (out_a / "simulated_data.csv").read_text()
    b = (out_b / "simulated_data.csv").read_text()
    assert a == b  # same seed → byte-identical files
