"""Smoke tests for the `dsl run` command-line interface."""
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dsl.cli import (
    _changed,
    _inject_reload,
    _prompt_continue,
    _run_once,
    _scenario_runs,
    _serve,
    main,
)
from tests.conftest import scenario_dict as base_scenario
from tests.conftest import write_csv

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
    # A from_csv path relative to the scenario file must work even
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


def test_relative_from_csv_population_path_resolves(tmp_path, monkeypatch):
    # A relative from_csv path also resolves for a top-level generated
    # POPULATION when launched from elsewhere.
    exp = tmp_path / "experiment"
    exp.mkdir()
    write_csv(exp / "clim.csv", ["2010-01", "2010-02", "2010-03"],
              rainfall=[1.0, 2.0, 3.0])
    write_csv(exp / "pop.csv", ["2010-01", "2010-02", "2010-03"],
              pop=[1000, 1100, 1200])
    scenario = {
        "period": "monthly", "n_total": 3, "start_period": "2010-01",
        "variables": [{"name": "rainfall", "generate": "from_csv",
                       "params": {"file": "clim.csv", "column": "rainfall"}}],
        "disease_cases": {
            "population": {"generate": "from_csv",
                           "params": {"file": "pop.csv", "column": "pop"}},
            "depends_on": [{"variable": "rainfall"}],
        },
    }
    (exp / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    code = main(["run", str(exp / "scenario.yaml"), "-o", str(out)])
    assert code == 0
    df = pd.read_csv(out / "simulated_data.csv")
    assert df["population"].tolist() == [1000, 1100, 1200]


def test_relative_from_csv_per_location_population_path_resolves(tmp_path, monkeypatch):
    # A relative from_csv path resolves for a PER-LOCATION population too,
    # when the scenario uses the locations mapping form.
    exp = tmp_path / "experiment"
    exp.mkdir()
    write_csv(exp / "clim.csv", ["2010-01", "2010-02", "2010-03"],
              rainfall=[1.0, 2.0, 3.0])
    write_csv(exp / "pop.csv", ["2010-01", "2010-02", "2010-03"],
              pop=[2000, 2100, 2200])
    scenario = {
        "period": "monthly", "n_total": 3, "start_period": "2010-01",
        "locations": {
            "north": {"population": {"generate": "from_csv",
                                     "params": {"file": "pop.csv", "column": "pop"}}},
        },
        "variables": [{"name": "rainfall", "generate": "from_csv",
                       "params": {"file": "clim.csv", "column": "rainfall"}}],
        "disease_cases": {"depends_on": [{"variable": "rainfall"}]},
    }
    (exp / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    code = main(["run", str(exp / "scenario.yaml"), "-o", str(out)])
    assert code == 0
    df = pd.read_csv(out / "simulated_data.csv")
    assert df["population"].tolist() == [2000, 2100, 2200]


def test_generation_error_is_clean_cli_error(tmp_path, capsys):
    # An invalid generator param surfaces during generation, after
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


def test_new_writes_runnable_starter(tmp_path):
    # `dsl new` writes a scenario that `dsl run` accepts end-to-end — guards
    # the template against drifting out of sync with the schema.
    scenario = tmp_path / "starter.yaml"
    assert main(["new", str(scenario)]) == 0
    assert scenario.is_file()
    out = tmp_path / "out"
    assert main(["run", str(scenario), "-o", str(out)]) == 0
    assert (out / "simulated_data.csv").is_file()


def test_new_default_path_and_message(tmp_path, monkeypatch, capsys):
    # With no path it writes scenario.yaml in the cwd and names it in the hint.
    monkeypatch.chdir(tmp_path)
    assert main(["new"]) == 0
    assert (tmp_path / "scenario.yaml").is_file()
    assert "scenario.yaml" in capsys.readouterr().out


def test_new_refuses_overwrite_without_force(tmp_path, capsys):
    scenario = tmp_path / "starter.yaml"
    scenario.write_text("period: monthly\n")  # pre-existing, distinct content
    assert main(["new", str(scenario)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert scenario.read_text() == "period: monthly\n"  # untouched
    # --force overwrites it with the template.
    assert main(["new", str(scenario), "--force"]) == 0
    assert "period: monthly\n" not in scenario.read_text()


def test_changed_detects_mtime_advance(tmp_path):
    # The watch loop's change check: a newer mtime is a change, a missing
    # file is treated as unchanged (some editors unlink mid-save).
    f = tmp_path / "s.yaml"
    f.write_text("a")
    mtime = f.stat().st_mtime
    assert _changed(str(f), mtime) == (False, mtime)
    assert _changed(str(f), mtime - 1)[0] is True
    assert _changed(str(tmp_path / "gone.yaml"), mtime) == (False, mtime)


def test_run_once_writes_files(tmp_path):
    # The shared single-run path (used by both one-shot and --watch).
    scenario = write_scenario(tmp_path, base_scenario())
    out = tmp_path / "out"
    assert _run_once(str(scenario), out, plot=False, plot_format="html") == 0
    assert (out / "simulated_data.csv").is_file()
    assert (out / "metadata.json").is_file()


def test_run_watch_flag_is_accepted(tmp_path):
    # A first-run failure under --watch returns the error code, never looping —
    # confirms the flag parses and the loop only starts after a clean first run.
    bad = tmp_path / "missing.yaml"
    assert main(["run", str(bad), "-o", str(tmp_path / "o"), "--watch"]) == 1


def test_inject_reload_adds_script_once(tmp_path):
    # The live-reload script is injected before </body> and is idempotent.
    html = tmp_path / "plot.html"
    html.write_text("<html><body>plot</body></html>")
    _inject_reload(html)
    once = html.read_text()
    assert once.count("__plot_version__") == 1
    assert once.index("__plot_version__") < once.index("</body>")
    _inject_reload(html)  # second call must not double-inject
    assert html.read_text().count("__plot_version__") == 1


def test_serve_serves_files_and_version(tmp_path):
    # The watch server serves the out dir and reports the shared version.
    (tmp_path / "plot.html").write_text("<body>hi</body>")
    version = [0]
    server = _serve(tmp_path, version)
    try:
        port = server.server_address[1]

        def get(path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
                return r.read().decode()

        assert get("/plot.html") == "<body>hi</body>"
        assert get("/__plot_version__") == "0"
        version[0] += 1  # a re-run bumps it; the handler reads it live
        assert get("/__plot_version__") == "1"
    finally:
        server.shutdown()


def test_scenario_runs_finds_and_orders_siblings(tmp_path, monkeypatch):
    # out/<name>, then _1, _2, _10 numerically; unrelated folders excluded.
    monkeypatch.chdir(tmp_path)
    for d in ("scenario", "scenario_1", "scenario_2", "scenario_10", "other"):
        (tmp_path / "out" / d).mkdir(parents=True)
    runs = [p.name for p in _scenario_runs("scenario.yaml")]
    assert runs == ["scenario", "scenario_1", "scenario_2", "scenario_10"]


def test_prompt_continue_non_interactive_returns_none(tmp_path, monkeypatch):
    # No TTY (the test runner) → no prompt, fall back to a new run.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "scenario").mkdir(parents=True)
    assert _prompt_continue("scenario.yaml") is None


def test_prompt_continue_picks_folder(tmp_path, monkeypatch):
    # Faking a TTY + the user typing "2" continues the second listed run.
    monkeypatch.chdir(tmp_path)
    for d in ("scenario", "scenario_1"):
        (tmp_path / "out" / d).mkdir(parents=True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _="": "2")
    assert _prompt_continue("scenario.yaml") == Path("out") / "scenario_1"


def test_prompt_continue_new_choice_returns_none(tmp_path, monkeypatch):
    # Typing "n" (or anything non-numeric) means a new run.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "scenario").mkdir(parents=True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _="": "n")
    assert _prompt_continue("scenario.yaml") is None


def test_prompt_continue_no_existing_runs_returns_none(tmp_path, monkeypatch):
    # Nothing to continue → None even with a TTY (no prompt shown).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert _prompt_continue("scenario.yaml") is None


def test_run_new_flag_skips_picker_and_autonumbers(tmp_path, monkeypatch):
    # --new must NOT prompt, even with an existing run folder and a TTY — it
    # auto-numbers a fresh folder. We fake a TTY and make input() blow up so
    # any prompt attempt would fail the test.
    monkeypatch.chdir(tmp_path)
    scenario = write_scenario(tmp_path, base_scenario())
    (tmp_path / "out" / "scenario").mkdir(parents=True)  # an existing run
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda *_: pytest.fail("--new must not prompt")
    )
    assert main(["run", str(scenario), "--new"]) == 0
    # Wrote a NEW numbered folder, leaving the existing out/scenario alone.
    assert (tmp_path / "out" / "scenario_1" / "simulated_data.csv").is_file()


def test_run_picker_applies_without_watch(tmp_path, monkeypatch):
    # The continue prompt now fires on a plain `dsl run` too (not just --watch):
    # a faked TTY choosing "1" continues the existing folder in place.
    monkeypatch.chdir(tmp_path)
    scenario = write_scenario(tmp_path, base_scenario())
    existing = tmp_path / "out" / "scenario"
    existing.mkdir(parents=True)
    (existing / "stale.txt").write_text("x")  # marker to prove same folder reused
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "1")
    assert main(["run", str(scenario)]) == 0
    assert (existing / "simulated_data.csv").is_file()  # wrote into the chosen one
    assert not (tmp_path / "out" / "scenario_1").exists()  # no new folder spawned
