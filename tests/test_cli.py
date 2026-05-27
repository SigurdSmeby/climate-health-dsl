"""Smoke tests for the `dsl run` command-line interface."""
import numpy as np
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


def test_output_is_reproducible(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert main(["run", EXAMPLE, "-o", str(out_a)]) == 0
    assert main(["run", EXAMPLE, "-o", str(out_b)]) == 0
    a = (out_a / "simulated_data.csv").read_text()
    b = (out_b / "simulated_data.csv").read_text()
    assert a == b  # same seed → byte-identical files
