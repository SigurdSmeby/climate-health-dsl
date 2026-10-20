"""Feature 14: multiple seeded replicates via `dsl run --replicates N`.

Each replicate i uses seed = base_seed + i and writes its own subfolder with a
full dataset + metadata (independently reproducible). N=1 is the single-run
default. TDD: written before the CLI flag exists.
"""
import argparse
from pathlib import Path

import pandas as pd
import pytest

from dsl.cli import main

EXAMPLE = "examples/cross_validation.yaml"


def test_replicates_write_one_folder_each(tmp_path):
    out = tmp_path / "out"
    code = main(["run", EXAMPLE, "-o", str(out), "--replicates", "3"])
    assert code == 0
    for i in range(3):
        rep = out / f"rep_{i:02d}"
        assert (rep / "simulated_data.csv").is_file()
        assert (rep / "metadata.json").is_file()


def test_replicates_use_consecutive_seeds(tmp_path):
    out = tmp_path / "out"
    main(["run", EXAMPLE, "-o", str(out), "--replicates", "3"])
    import json

    seeds = [
        json.loads((out / f"rep_{i:02d}" / "metadata.json").read_text())["scenario"][
            "seed"
        ]
        for i in range(3)
    ]
    # cross_validation.yaml has seed 5 → replicates use 5, 6, 7.
    assert seeds == [seeds[0], seeds[0] + 1, seeds[0] + 2]


def test_replicates_differ_from_each_other(tmp_path):
    out = tmp_path / "out"
    main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"])
    a = pd.read_csv(out / "rep_00" / "simulated_data.csv")["disease_cases"]
    b = pd.read_csv(out / "rep_01" / "simulated_data.csv")["disease_cases"]
    # Different seeds → different draws.
    assert not a.equals(b)


def test_single_replicate_is_the_plain_run(tmp_path):
    # --replicates 1 (or omitted) writes directly to out/, no rep_ subfolder —
    # byte-identical to a normal run.
    plain = tmp_path / "plain"
    one = tmp_path / "one"
    main(["run", EXAMPLE, "-o", str(plain)])
    main(["run", EXAMPLE, "-o", str(one), "--replicates", "1"])
    p = pd.read_csv(plain / "simulated_data.csv")
    o = pd.read_csv(one / "simulated_data.csv")
    assert p.equals(o)
    assert not (one / "rep_00").exists()


def test_replicate_is_reproducible_from_its_metadata(tmp_path):
    out = tmp_path / "out"
    main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"])
    repro = tmp_path / "repro"
    main(["run", str(out / "rep_01" / "metadata.json"), "-o", str(repro)])
    orig = pd.read_csv(out / "rep_01" / "simulated_data.csv")
    again = pd.read_csv(repro / "simulated_data.csv")
    assert orig.equals(again)


@pytest.mark.parametrize("bad", ["0", "-1", "two"])
def test_a_bad_replicate_count_is_rejected_before_anything_runs(
    tmp_path, capsys, bad
):
    """An argument error, so argparse rejects it (exit 2) before the
    dataset is generated — rather than paying for the parse first."""
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        main(["run", EXAMPLE, "-o", str(out), "--replicates", bad])
    assert excinfo.value.code == 2
    assert not out.exists()
    assert "Traceback" not in capsys.readouterr().err


def test_stale_replicate_folders_are_cleared(tmp_path):
    # Regression: a reused -o folder from a LARGER --replicates run must not
    # leave stale rep_NN/ dirs behind when re-run with fewer replicates.
    out = tmp_path / "out"
    main(["run", EXAMPLE, "-o", str(out), "--replicates", "5"])
    assert (out / "rep_04").is_dir()
    main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"])
    assert (out / "rep_00").is_dir()
    assert (out / "rep_01").is_dir()
    assert not (out / "rep_02").exists()
    assert not (out / "rep_04").exists()


def test_reusing_a_folder_clears_a_previous_single_run(tmp_path):
    """A folder that held one run's output must not leave its CSVs and
    folds/ beside the new rep_NN dirs, where they read as part of this run."""
    out = tmp_path / "out"
    assert main(["run", EXAMPLE, "-o", str(out)]) == 0
    assert (out / "simulated_data.csv").is_file()

    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"]) == 0
    assert not (out / "simulated_data.csv").exists()
    assert sorted(p.name for p in out.iterdir()) == ["rep_00", "rep_01"]


def test_a_folder_holding_other_files_is_refused_not_cleaned(tmp_path, capsys):
    """The cleanup deletes what it will not overwrite, so it must not run
    in a folder it does not own. `-o .` in a working directory is the case
    that bites: it took the scenario file, and everything beside it, with
    no warning and no prompt.
    """
    out = tmp_path / "work"
    (out / "src").mkdir(parents=True)
    (out / "src" / "main.py").write_text("code")
    scenario = out / "scenario.yaml"
    scenario.write_text(Path(EXAMPLE).read_text())
    thesis = out / "thesis.tex"
    thesis.write_text("my thesis")

    code = main(["run", str(scenario), "-o", str(out), "--replicates", "2"])

    assert code == 1
    assert scenario.is_file(), "deleted the scenario it was running"
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

    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"]) == 1
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
    (out / "rep_07").mkdir()

    assert _foreign_entries(out, keep={"rep_00"}) == []


def test_a_failed_run_keeps_the_previous_replicates(tmp_path, monkeypatch):
    """The cleanup ran before the first replicate succeeded, so a run that
    failed part-way had already deleted the results it was replacing —
    three good replicates plus a failing run asking for one left rep_00
    half written and the rest gone."""
    import dsl._run as run_module

    out = tmp_path / "out"
    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "3"]) == 0
    before = sorted(p.name for p in out.iterdir())
    assert before == ["rep_00", "rep_01", "rep_02"]

    monkeypatch.setattr(run_module, "_run_once", lambda *a, **k: 1)
    args = argparse.Namespace(
        scenario=EXAMPLE, replicates=1, plot=False, plot_format="html",
        watch=False, plot_ticks=None,
    )
    assert run_module._run_replicates(args, out) == 1
    assert sorted(p.name for p in out.iterdir()) == before


def test_a_successful_run_still_clears_stale_replicates(tmp_path):
    """Writing before cleaning must not mean never cleaning."""
    out = tmp_path / "out"
    main(["run", EXAMPLE, "-o", str(out), "--replicates", "3"])
    main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"])
    assert sorted(p.name for p in out.iterdir()) == ["rep_00", "rep_01"]


def test_a_reused_replicate_folder_does_not_keep_its_old_plot(tmp_path):
    """write_output rewrites the data and clears folds/, but a plot left
    from a run made with different --plot flags would ship beside data it
    does not describe."""
    out = tmp_path / "out"
    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2", "--plot"]) == 0
    assert (out / "rep_00" / "plot.html").is_file()

    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"]) == 0
    assert not (out / "rep_00" / "plot.html").exists()
    assert (out / "rep_00" / "simulated_data.csv").is_file()


def test_a_failed_run_never_mixes_two_runs_replicates(tmp_path):
    """Writing in place meant a run that failed part-way left its new
    rep_00 among an older run's rep_01..04 — a folder that still looks
    like one valid five-replicate set, with two seeds in it. The set is
    built aside and swapped in only once it is complete."""
    import dsl._run as run_module

    out = tmp_path / "out"
    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "5"]) == 0
    before = {
        d.name: (d / "simulated_data.csv").read_bytes() for d in sorted(out.iterdir())
    }
    assert len(before) == 5

    # A DIFFERENT seed, so a replicate this run writes is distinguishable
    # from the one it would replace — otherwise rep_00 is byte-identical
    # either way and the mixing is invisible.
    calls = {"n": 0}
    real = run_module._run_once

    def fail_at_second(scenario, rep_dir, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            return 1
        config = kwargs.pop("config")
        return real(
            scenario, rep_dir, *args,
            config=config.model_copy(update={"seed": config.seed + 1000}),
            **kwargs,
        )

    run_module._run_once = fail_at_second
    try:
        args = argparse.Namespace(
            scenario=EXAMPLE, replicates=3, plot=False, plot_format="html",
            watch=False, plot_ticks=None,
        )
        assert run_module._run_replicates(args, out) == 1
    finally:
        run_module._run_once = real

    after = {
        d.name: (d / "simulated_data.csv").read_bytes() for d in sorted(out.iterdir())
    }
    assert after == before, (
        "a failed run left its own rep_00 among the previous run's "
        "replicates — one folder, two seeds, looking like a valid set"
    )


def test_a_leftover_staging_folder_does_not_block_the_next_run(tmp_path):
    """A run killed mid-write leaves .partial behind; the next one must
    clear it rather than refuse the folder as foreign."""
    out = tmp_path / "out"
    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"]) == 0
    (out / ".partial" / "rep_00").mkdir(parents=True)

    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["rep_00", "rep_01"]


def test_the_messages_name_where_the_files_actually_are(tmp_path, capsys):
    """Replicates are built in a staging folder and swapped in, so naming
    the real write path pointed the reader at one deleted moments later."""
    out = tmp_path / "out"
    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2",
                 "--plot"]) == 0
    printed = capsys.readouterr().out

    assert ".partial" not in printed
    for line in printed.splitlines():
        if line.startswith(("Wrote 1", "Wrote 2", "Wrote plot")):
            named = Path(line.rsplit(" ", 1)[-1].rstrip("/"))
            assert named.exists(), f"named a path that does not exist: {named}"


def test_the_new_set_lands_before_the_old_one_is_removed(tmp_path):
    """A true atomic swap would need to exchange two directories, which no
    portable call offers. Moving each new rep_NN into place first means a
    kill partway leaves complete replicates rather than a gap."""
    out = tmp_path / "out"
    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "4"]) == 0
    assert main(["run", EXAMPLE, "-o", str(out), "--replicates", "2"]) == 0

    assert sorted(p.name for p in out.iterdir()) == ["rep_00", "rep_01"]
    for rep in out.iterdir():
        assert (rep / "simulated_data.csv").is_file()
        assert (rep / "metadata.json").is_file()


def test_scenario_warnings_print_once_not_once_per_replicate(tmp_path, capsys):
    """Replicates share one parsed scenario, differing only by seed, so a
    soft warning about the scenario is the same every time — printing it N
    times buries the per-replicate output."""
    path = tmp_path / "s.yaml"
    path.write_text(
        "period: monthly\nn_total: 24\nseed: 1\n"
        "locations: { loc: { population: 100000 } }\n"
        "series:\n"
        "  - { name: orphan, generate: flat, params: { level: 5.0 } }\n"
        "  - { name: r, generate: flat, params: { level: 5.0 } }\n"
        "  - { name: disease_cases, counts: {}, "
        "depends_on: [{ series: r, lag: 1 }] }\n"
    )
    assert main(["run", str(path), "-o", str(tmp_path / "out"),
                 "--replicates", "3"]) == 0

    err = capsys.readouterr().err
    assert err.count("nothing depends on it") == 1
