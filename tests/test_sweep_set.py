"""The guarantees a multi-run set makes about the output folder.

A swept scenario writes N datasets into one directory, so it has to answer
the questions a single run never faces: what happens to results already
there, what a half-finished run leaves behind, and which folders it may
clear at all. They are pointed at a sweep, which is the only
thing that writes such a set; several describe bugs found before sweeps
existed, which is why the comments name failures the current code cannot
reach.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dsl.cli import main

# Two combinations, from the example the fold tests use, so a sweep of it
# exercises split/folds output as well as the CSV.
_BASE = yaml.safe_load(Path("examples/cross_validation.yaml").read_text())


def _sweep_scenario(tmp_path, values=(42, 43), name="s.yaml") -> str:
    """The example, sweeping its seed over ``values``."""
    scenario = {**_BASE, "seed": {"sweep": list(values)}}
    path = tmp_path / name
    path.write_text(yaml.safe_dump(scenario))
    return str(path)


def _labels(values) -> list[str]:
    return [f"seed_{v}" for v in values]


def _index(out) -> dict:
    """``sweep.json``, parsed."""
    return json.loads((out / "sweep.json").read_text())


def _run_dirs(out) -> list[str]:
    """The run directories in ``out``, sorted.

    Not everything in a sweep's folder is a run: ``sweep.json`` sits beside
    them, so these tests compare the directories rather than the listing.
    """
    return sorted(p.name for p in out.iterdir() if p.is_dir())


def test_each_combination_writes_a_full_dataset(tmp_path):
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 0
    for label in _labels((42, 43)):
        assert (out / label / "simulated_data.csv").is_file()
        assert (out / label / "metadata.json").is_file()


def test_a_reused_folder_clears_a_previous_single_run(tmp_path):
    """A folder that held one run's output must not leave its CSV and
    folds/ beside the new run dirs, where they read as part of this run."""
    out = tmp_path / "out"
    assert main(["run", "examples/cross_validation.yaml", "-o", str(out)]) == 0
    assert (out / "simulated_data.csv").is_file()

    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 0
    assert not (out / "simulated_data.csv").exists()
    assert _run_dirs(out) == _labels((42, 43))


def test_a_folder_holding_other_files_is_refused_not_cleaned(tmp_path, capsys):
    """The cleanup deletes what it will not overwrite, so it must not run
    in a folder it does not own. `-o .` in a working directory is the case
    that bites: it took the scenario file, and everything beside it, with
    no warning and no prompt.
    """
    out = tmp_path / "work"
    (out / "src").mkdir(parents=True)
    (out / "src" / "main.py").write_text("code")
    scenario = _sweep_scenario(out)
    thesis = out / "thesis.tex"
    thesis.write_text("my thesis")

    assert main(["run", scenario, "-o", str(out)]) == 1
    assert Path(scenario).is_file(), "deleted the scenario it was running"
    assert thesis.read_text() == "my thesis"
    assert (out / "src" / "main.py").read_text() == "code"

    err = capsys.readouterr().err
    # The message has to name what stopped it, or the user cannot act on it.
    assert "thesis.tex" in err and "src" in err


def test_the_refusal_lists_at_most_a_handful_of_names(tmp_path, capsys):
    """A home directory would otherwise print hundreds of names."""
    out = tmp_path / "many"
    out.mkdir()
    for i in range(12):
        (out / f"file_{i}.txt").write_text("x")

    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.count("file_") == 5
    assert "…" in err


def test_output_of_a_previous_run_is_not_foreign(tmp_path):
    """Every name the tool itself writes must still be cleanable, or
    re-running into a used folder would refuse instead of refreshing.
    """
    from dsl._run import _foreign_entries

    out = tmp_path / "out"
    out.mkdir()
    for name in ("simulated_data.csv", "metadata.json", "plot.html", "plot.png"):
        (out / name).write_text("x")
    (out / "folds").mkdir()
    # A run directory left by a previous, differently-shaped sweep. It is
    # cleanable only because sweep.json says this folder is a sweep's own;
    # the same directory sitting in a folder without one belongs to another
    # scenario's run and is protected.
    (out / "sweep.json").write_text("{}")
    stale = out / "seed_99"
    stale.mkdir()
    (stale / "metadata.json").write_text("{}")

    assert _foreign_entries(out, keep={"seed_42"}) == []


def test_another_scenarios_run_is_foreign_without_a_sweep_index(tmp_path):
    """The same leftover directory, in a folder no sweep built: this is
    `out/`, where `dsl run` auto-names every run, so those directories are
    other scenarios' datasets rather than a set's own leftovers."""
    from dsl._run import _foreign_entries

    out = tmp_path / "out"
    out.mkdir()
    other = out / "important_study"
    other.mkdir()
    (other / "metadata.json").write_text("{}")

    assert _foreign_entries(out, keep={"seed_42"}) == ["important_study"]


def test_stale_run_folders_are_cleared(tmp_path):
    """A reused -o folder from a LARGER sweep must not leave stale run
    dirs behind when re-run over fewer values."""
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path, range(42, 47)), "-o", str(out)]) == 0
    assert (out / "seed_46").is_dir()

    assert main([
        "run", _sweep_scenario(tmp_path, (42, 43), "narrow.yaml"), "-o", str(out),
    ]) == 0
    assert _run_dirs(out) == _labels((42, 43))


def test_a_failed_run_keeps_the_previous_set(tmp_path, monkeypatch):
    """The cleanup ran before the first run succeeded, so a set that
    failed part-way had already deleted the results it was replacing."""
    import dsl._run as run_module

    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path, (42, 43, 44)), "-o", str(out)]) == 0
    before = _run_dirs(out)
    assert before == _labels((42, 43, 44))

    monkeypatch.setattr(run_module, "_run_once", lambda *a, **k: 1)
    args = argparse.Namespace(
        scenario=_sweep_scenario(tmp_path), plot=False, plot_format="html",
        watch=False, plot_ticks=None,
    )
    assert run_module._run_sweep(args, out) == 1
    assert _run_dirs(out) == before


def test_a_failed_run_never_mixes_two_sets(tmp_path):
    """Writing in place meant a set that failed part-way left its new
    first run among an older set's remaining ones — a folder that still
    looks like one valid set, with two generations in it. The set is built
    aside and swapped in only once it is complete."""
    import dsl._run as run_module

    out = tmp_path / "out"
    wide = _sweep_scenario(tmp_path, range(42, 47))
    assert main(["run", wide, "-o", str(out)]) == 0
    before = {
        name: (out / name / "simulated_data.csv").read_bytes()
        for name in _run_dirs(out)
    }
    assert len(before) == 5

    # A DIFFERENT seed, so a run this set writes is distinguishable from
    # the one it would replace — otherwise the first is byte-identical
    # either way and the mixing is invisible.
    calls = {"n": 0}
    real = run_module._run_once

    def fail_at_second(scenario, run_dir, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            return 1
        config = kwargs.pop("config")
        return real(
            scenario, run_dir, *args,
            config=config.model_copy(update={"seed": config.seed + 1000}),
            **kwargs,
        )

    monkey = pytest.MonkeyPatch()
    monkey.setattr(run_module, "_run_once", fail_at_second)
    try:
        args = argparse.Namespace(
            scenario=wide, plot=False, plot_format="html",
            watch=False, plot_ticks=None,
        )
        assert run_module._run_sweep(args, out) == 1
    finally:
        monkey.undo()

    after = {
        name: (out / name / "simulated_data.csv").read_bytes()
        for name in _run_dirs(out)
    }
    assert after == before, (
        "a failed run left its own output among the previous set's — one "
        "folder, two generations, looking like a valid set"
    )


def test_a_successful_run_still_clears_a_stale_set(tmp_path):
    """Writing before cleaning must not mean never cleaning."""
    out = tmp_path / "out"
    main(["run", _sweep_scenario(tmp_path, (42, 43, 44)), "-o", str(out)])
    main(["run", _sweep_scenario(tmp_path, (42, 43), "narrow.yaml"), "-o", str(out)])
    assert _run_dirs(out) == _labels((42, 43))


def test_a_reused_run_folder_does_not_keep_its_old_plot(tmp_path):
    """write_output rewrites the data and clears folds/, but a plot left
    from a run made with different --plot flags would ship beside data it
    does not describe."""
    out = tmp_path / "out"
    scenario = _sweep_scenario(tmp_path)
    assert main(["run", scenario, "-o", str(out), "--plot"]) == 0
    assert (out / "seed_42" / "plot.html").is_file()

    assert main(["run", scenario, "-o", str(out)]) == 0
    assert not (out / "seed_42" / "plot.html").exists()
    assert (out / "seed_42" / "simulated_data.csv").is_file()


def test_a_leftover_staging_folder_does_not_block_the_next_run(tmp_path):
    """A run killed mid-write leaves .partial behind; the next one must
    clear it rather than refuse the folder as foreign."""
    out = tmp_path / "out"
    scenario = _sweep_scenario(tmp_path)
    assert main(["run", scenario, "-o", str(out)]) == 0
    (out / ".partial" / "seed_42").mkdir(parents=True)

    assert main(["run", scenario, "-o", str(out)]) == 0
    assert _run_dirs(out) == _labels((42, 43))


def test_the_messages_name_where_the_files_actually_are(tmp_path, capsys):
    """A set is built in a staging folder and swapped in, so naming the
    real write path pointed the reader at one deleted moments later."""
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out), "--plot"]) == 0
    printed = capsys.readouterr().out

    assert ".partial" not in printed
    for line in printed.splitlines():
        if line.startswith(("Wrote 1", "Wrote 2", "Wrote plot")):
            named = Path(line.rsplit(" ", 1)[-1].rstrip("/"))
            assert named.exists(), f"named a path that does not exist: {named}"


def test_the_new_set_lands_before_the_old_one_is_removed(tmp_path):
    """A true atomic swap would need to exchange two directories, which no
    portable call offers. Moving each new run into place first means a kill
    partway leaves complete runs rather than a gap."""
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path, range(42, 46)), "-o", str(out)]) == 0
    assert main([
        "run", _sweep_scenario(tmp_path, (42, 43), "narrow.yaml"), "-o", str(out),
    ]) == 0

    assert _run_dirs(out) == _labels((42, 43))
    for name in _run_dirs(out):
        assert (out / name / "simulated_data.csv").is_file()
        assert (out / name / "metadata.json").is_file()


def test_scenario_warnings_print_once_not_once_per_run(tmp_path, capsys):
    """A sweep varies scalar values, not the shape of the scenario, so a
    soft warning about its structure is the same every time — printing it
    N times buries the per-run output."""
    path = tmp_path / "s.yaml"
    path.write_text(
        "period: monthly\nn_total: 24\nseed: { sweep: [1, 2, 3] }\n"
        "locations: { loc: { population: 100000 } }\n"
        "series:\n"
        "  - { name: orphan, generate: flat, params: { level: 5.0 } }\n"
        "  - { name: r, generate: flat, params: { level: 5.0 } }\n"
        "  - { name: disease_cases, counts: {}, "
        "depends_on: [{ series: r, lag: 1 }] }\n"
    )
    assert main(["run", str(path), "-o", str(tmp_path / "out")]) == 0

    err = capsys.readouterr().err
    assert err.count("nothing depends on it") == 1


def test_a_file_named_like_a_run_folder_is_refused_not_overwritten(tmp_path, capsys):
    """`keep` exempted any entry with a run-dir name, file or not, so the
    foreign check passed and the rename then failed mid-swap — after every
    run had already been generated."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "seed_42").write_text("my notes")

    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 1
    assert (out / "seed_42").read_text() == "my notes"
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "seed_42" in err


def test_each_run_is_reproducible_from_its_own_metadata(tmp_path):
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 0
    repro = tmp_path / "repro"
    assert main(["run", str(out / "seed_43" / "metadata.json"), "-o", str(repro)]) == 0
    orig = pd.read_csv(out / "seed_43" / "simulated_data.csv")
    again = pd.read_csv(repro / "simulated_data.csv")
    assert orig.equals(again)


# --- sweep.json as the marker of a folder this tool built as a set. ---------
#
# Its CONTENTS are a later change: `chap export-metrics` identifies rows by
# filename, so something has to map a directory name back to the values that
# made it (parsing the name cannot, since a float's '.' and a negative's '-'
# are slugged away). Here it only has to exist, because that is what tells
# the cleanup which folders are a sweep's own.


def test_a_sweep_writes_an_index_file(tmp_path):
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 0
    assert (out / "sweep.json").is_file()
    # Its presence is what marks the folder as a set's own, which the
    # cleanup relies on; its contents are asserted below.
    assert set(_index(out)) == {"dsl_version", "swept", "runs"}


def test_a_plain_run_writes_no_index(tmp_path):
    out = tmp_path / "out"
    assert main(["run", "examples/minimal.yaml", "-o", str(out)]) == 0
    assert not (out / "sweep.json").exists()


def test_a_sweep_does_not_delete_another_scenarios_run(tmp_path, capsys):
    """`out/` is where `dsl run` auto-names every run, so it is exactly the
    folder most likely to hold datasets a sweep must not touch. Treating
    any metadata.json-bearing directory as this tool's own to delete lost
    a complete unrelated dataset, silently."""
    out = tmp_path / "out"
    assert main(["run", "examples/minimal.yaml", "-o", str(out / "important")]) == 0
    before = (out / "important" / "simulated_data.csv").read_bytes()

    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 1

    assert (out / "important" / "simulated_data.csv").read_bytes() == before
    err = capsys.readouterr().err
    assert "important" in err


def test_a_sweep_replaces_a_plain_run_in_the_same_folder(tmp_path):
    """A plain run's own files (simulated_data.csv, metadata.json, folds/)
    are this tool's output by name, so pointing -o at a folder you already
    ran into keeps working — the single dataset is replaced by the set.

    What a sweep must not touch is a *sibling run directory*; that is
    `test_a_sweep_does_not_delete_another_scenarios_run`.
    """
    out = tmp_path / "out"
    assert main(["run", "examples/minimal.yaml", "-o", str(out)]) == 0
    assert (out / "simulated_data.csv").is_file()

    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 0
    assert not (out / "simulated_data.csv").exists()
    assert _run_dirs(out) == _labels((42, 43))


def test_re_sweeping_a_sweeps_own_folder_clears_its_leftovers(tmp_path):
    """A folder holding sweep.json is one this tool built as a set, so a
    narrower sweep into it may clear what it no longer covers."""
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path, (42, 43, 44)), "-o", str(out)]) == 0
    assert (out / "sweep.json").is_file()

    assert main([
        "run", _sweep_scenario(tmp_path, (42, 43), "narrow.yaml"), "-o", str(out),
    ]) == 0
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == _labels((42, 43))
    assert (out / "sweep.json").is_file()


# --- sweep.json's contents: which directory is which run. -------------------
#
# `chap export-metrics` identifies rows by filename, so joining a model's
# scores back to the parameters that produced them needs a translation from
# directory name to values. Parsing the name cannot do it: a float's '.' and
# a negative's '-' are slugged away, and two sweeps on one field name are
# qualified by their parent.


def test_the_index_lists_every_run_with_its_values(tmp_path):
    scenario = {
        **_BASE,
        "series": [
            {"name": "rainfall", "generate": "seasonal_spike",
             "params": {"spike_height": {"sweep": [10, 40]}}},
            {"name": "disease_cases", "counts": {},
             "depends_on": [{"series": "rainfall", "lag": {"sweep": [1, 2]},
                             "weight": 1.0}]},
        ],
    }
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(scenario))
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0

    index = _index(out)
    assert index["swept"] == ["spike_height", "lag"]
    assert [run["dir"] for run in index["runs"]] == [
        "spike_height_10__lag_1",
        "spike_height_10__lag_2",
        "spike_height_40__lag_1",
        "spike_height_40__lag_2",
    ]
    assert index["runs"][0]["params"] == {"spike_height": 10, "lag": 1}
    assert index["runs"][3]["params"] == {"spike_height": 40, "lag": 2}
    # Every directory on disk is described, and nothing that is not there.
    assert {run["dir"] for run in index["runs"]} == set(_run_dirs(out))


def test_the_index_records_the_seed_of_each_run(tmp_path):
    """The seed is what makes a run reproducible, and a seed sweep is the
    case where it is the thing that varies."""
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path, (42, 43)), "-o", str(out)]) == 0

    index = _index(out)
    assert [run["seed"] for run in index["runs"]] == [42, 43]
    assert [run["params"]["seed"] for run in index["runs"]] == [42, 43]


def test_the_index_records_a_float_value_not_its_slug(tmp_path):
    """The directory is called weight_0p5; the join needs 0.5."""
    scenario = {**_BASE, "series": [
        {"name": "rainfall", "generate": "seasonal_spike"},
        {"name": "disease_cases", "counts": {}, "depends_on": [
            {"series": "rainfall", "lag": 1, "weight": {"sweep": [0.5, -1.5]}},
        ]},
    ]}
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(scenario))
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0

    index = _index(out)
    assert [run["dir"] for run in index["runs"]] == ["weight_0p5", "weight_m1p5"]
    assert [run["params"]["weight"] for run in index["runs"]] == [0.5, -1.5]


def test_the_index_names_the_dataset_to_read(tmp_path):
    """A consumer joining scores to parameters needs the CSV path, and
    building it by hand means knowing this tool's layout."""
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 0

    for run in _index(out)["runs"]:
        assert (out / run["dataset"]).is_file()
        assert run["dataset"].endswith("simulated_data.csv")


def test_the_index_is_rewritten_not_merged(tmp_path):
    """A stale entry describing a directory that is gone is worse than no
    index at all."""
    out = tmp_path / "out"
    main(["run", _sweep_scenario(tmp_path, (42, 43, 44)), "-o", str(out)])
    main(["run", _sweep_scenario(tmp_path, (42, 43), "narrow.yaml"), "-o", str(out)])

    assert [run["dir"] for run in _index(out)["runs"]] == _labels((42, 43))


def test_the_index_says_which_version_wrote_it(tmp_path):
    from dsl import __version__

    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path), "-o", str(out)]) == 0
    assert _index(out)["dsl_version"] == __version__
