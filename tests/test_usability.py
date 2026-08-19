"""Usability polish: friendly unknown-field errors, teaching starter, dsl list.

TDD: written before the CLI changes.
"""
import yaml

from dsl.cli import main


def write_scenario(tmp_path, data):
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# --- 1. Friendly "did you mean" for a misspelled field (extra_forbidden) ---

def test_misspelled_field_suggests_correction(tmp_path, capsys):
    data = {
        "period": "weekly", "n_total": 52,
        "variables": [{"name": "rain", "generate": "seasonal_spike"}],
        "disease_cases": {
            "populaton": 1000,  # typo of population
            "depends_on": [{"variable": "rain", "weight": 1}],
        },
    }
    path = write_scenario(tmp_path, data)
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "populaton" in err            # names the offending key
    assert "population" in err           # suggests the near-miss
    assert "did you mean" in err.lower()
    # The raw pydantic noise should be gone.
    assert "extra_forbidden" not in err
    assert "errors.pydantic.dev" not in err


def test_unknown_field_with_no_close_match_still_clear(tmp_path, capsys):
    data = {
        "period": "weekly", "n_total": 52,
        "variables": [{"name": "rain", "generate": "seasonal_spike"}],
        "disease_cases": {
            "population": 1000, "zzzzz": 5,  # not close to any real field
            "depends_on": [{"variable": "rain", "weight": 1}],
        },
    }
    path = write_scenario(tmp_path, data)
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "zzzzz" in err                # still names it
    assert "unknown field" in err.lower()


def test_misspelled_location_override_field_suggests_from_location_spec(
    tmp_path, capsys
):
    # Regression: the suggestion pool must come from LocationSpec's own
    # fields (just 'population'), not get derailed by the location's NAME
    # ('north') looking like a plausible parent-model key too.
    data = {
        "period": "weekly", "n_total": 52,
        "locations": {"north": {"populaton": 5}},  # typo, inside an override
        "variables": [{"name": "rain", "generate": "seasonal_spike"}],
        "disease_cases": {
            "depends_on": [{"variable": "rain", "weight": 1}],
        },
    }
    path = write_scenario(tmp_path, data)
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "populaton" in err
    assert "did you mean 'population'?" in err


# --- 2. Starter template teaches the newer features ---

def test_starter_mentions_transforms_and_shared(tmp_path, capsys):
    path = tmp_path / "s.yaml"
    assert main(["new", str(path)]) == 0
    text = path.read_text()
    assert "transforms" in text
    assert "shared" in text


def test_new_message_does_not_assume_uv(tmp_path, capsys):
    # A no-uv user must not be told to run `uv run …`; the hint should be the
    # bare `dsl run` (with uv mentioned only as an option).
    main(["new", str(tmp_path / "s.yaml")])
    out = capsys.readouterr().out
    assert "uv run dsl run" not in out
    assert "dsl run" in out


def test_starter_still_runs_clean(tmp_path, capsys):
    # The teaching comments must not break the scaffold: it still parses/runs
    # with no warnings (the commented blocks are inert).
    path = tmp_path / "s.yaml"
    main(["new", str(path)])
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 0
    err = capsys.readouterr().err
    assert "warning:" not in err


# --- 3. `dsl list` shows registered generators and transforms ---

def test_dsl_list_shows_generators_and_transforms(capsys):
    code = main(["list"])
    assert code == 0
    out = capsys.readouterr().out
    # A sample of each kind must appear.
    assert "seasonal_spike" in out
    assert "lag" in out
    assert "missing" in out
