"""The guarantees a multi-run set makes about the output folder.

A swept scenario writes N datasets into one directory, so it has to answer
the questions a single run never faces: what happens to results already
there, what a half-finished run leaves behind, and which folders it may
clear at all. These were written for ``--replicates`` and are kept here,
pointed at a sweep, because the sweep now shares that machinery
(``_run_set``) and is the only thing left using it.
"""
import argparse
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
    assert sorted(p.name for p in out.iterdir()) == _labels((42, 43))


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
    # A run directory from a previous, differently-shaped sweep: named
    # nothing in particular, recognised by the metadata.json in it.
    stale = out / "seed_99"
    stale.mkdir()
    (stale / "metadata.json").write_text("{}")

    assert _foreign_entries(out, keep={"seed_42"}) == []


def test_stale_run_folders_are_cleared(tmp_path):
    """A reused -o folder from a LARGER sweep must not leave stale run
    dirs behind when re-run over fewer values."""
    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path, range(42, 47)), "-o", str(out)]) == 0
    assert (out / "seed_46").is_dir()

    assert main([
        "run", _sweep_scenario(tmp_path, (42, 43), "narrow.yaml"), "-o", str(out),
    ]) == 0
    assert sorted(p.name for p in out.iterdir()) == _labels((42, 43))


def test_a_failed_run_keeps_the_previous_set(tmp_path, monkeypatch):
    """The cleanup ran before the first run succeeded, so a set that
    failed part-way had already deleted the results it was replacing."""
    import dsl._run as run_module

    out = tmp_path / "out"
    assert main(["run", _sweep_scenario(tmp_path, (42, 43, 44)), "-o", str(out)]) == 0
    before = sorted(p.name for p in out.iterdir())
    assert before == _labels((42, 43, 44))

    monkeypatch.setattr(run_module, "_run_once", lambda *a, **k: 1)
    args = argparse.Namespace(
        scenario=_sweep_scenario(tmp_path), plot=False, plot_format="html",
        watch=False, plot_ticks=None,
    )
    assert run_module._run_sweep(args, out) == 1
    assert sorted(p.name for p in out.iterdir()) == before


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
        d.name: (d / "simulated_data.csv").read_bytes() for d in sorted(out.iterdir())
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
        d.name: (d / "simulated_data.csv").read_bytes() for d in sorted(out.iterdir())
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
    assert sorted(p.name for p in out.iterdir()) == _labels((42, 43))


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
    assert sorted(p.name for p in out.iterdir()) == _labels((42, 43))


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

    assert sorted(p.name for p in out.iterdir()) == _labels((42, 43))
    for run_dir in out.iterdir():
        assert (run_dir / "simulated_data.csv").is_file()
        assert (run_dir / "metadata.json").is_file()


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
