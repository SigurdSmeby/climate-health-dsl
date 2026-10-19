"""Data-driven end-to-end tests: run the real `dsl` CLI as a subprocess
against every case in tests/e2e/cases/ and check the declared outcome.

Add a new end-to-end test by creating a new directory under cases/ with a
scenario.yaml in it — no Python needed. Any sibling files in that directory
(e.g. a CSV a from_csv generator points at with a relative path) are copied
alongside it, so relative paths in the scenario resolve correctly.

scenario.yaml is a normal scenario YAML plus two optional extra top-level
keys:

    _run:
      extra_args: ["--plot"]       # extra flags after `dsl run <scenario> -o <out>`
    _expect:
      exit_code: 0                 # default 0
      files_exist: [simulated_data.csv, metadata.json]
      files_absent: [train.csv]
      stderr_contains: "warning:"
      stderr_not_contains: "time_period"
      reproduce: true              # re-run from metadata.json, byte-identical
      columns:                     # exact column values in simulated_data.csv
        population: [1000, 1100, 1200]
      column_exists: [location]
      column_all:                  # every value in the column satisfies this
        disease_cases: non_negative_int   # non_negative_int | non_negative
      report:                      # assertions against folds/report.json
        folds: 5                   # how many folds the report describes
        total_events: 12           # planted features the scenario declares
        warnings: 0                # or a substring every warning must match
        fold_events:               # exact counts: fold -> series -> kind
          0: {rainfall: {storm: {train: 1, test: 0}}}

Both keys (and all their sub-keys) are optional; an empty ``_expect: {}``
just means "must run and exit 0". ``columns`` is for deterministic (seeded)
values; ``column_all`` is for a property that must hold on every row when
exact values aren't practical (e.g. random counts).
"""
import io
import json
import os
import shutil
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest
import yaml

from dsl.cli import main

CASES_DIR = Path(__file__).parent / "cases"
CASE_DIRS = sorted(p for p in CASES_DIR.iterdir() if (p / "scenario.yaml").is_file())

COLUMN_PROPERTIES = {
    # NaN is allowed (e.g. the lag warm-up rows) — only real values are checked.
    "non_negative_int": lambda s: ((s.dropna() >= 0) & (s.dropna() % 1 == 0)).all(),
    "non_negative": lambda s: (s.dropna() >= 0).all(),
}


class Result(NamedTuple):
    """What a run produced, shaped like subprocess.CompletedProcess."""

    returncode: int
    stdout: str
    stderr: str


def run_dsl(*args, cwd):
    """Run the CLI the way a user would, in this interpreter.

    Spawning `dsl` per call costs ~2.8s of interpreter and pandas import
    each time, which across these cases is minutes of startup for
    milliseconds of work. Calling main() keeps the same entry point and
    the same argv parsing, and only skips paying for a new process.

    `test_the_console_script_entry_point_works` still spawns the real
    binary once, so a broken [project.scripts] entry is still caught.

    Args:
        *args: argv for the CLI, without the program name.
        cwd: Directory to run from — a scenario's relative from_csv paths
            are resolved against it.

    Returns:
        A Result with the exit code and whatever was printed.
    """
    out, err = io.StringIO(), io.StringIO()
    before = Path.cwd()
    os.chdir(cwd)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = main([*args])
    except SystemExit as exc:
        # argparse exits rather than returning. A bare sys.exit() carries
        # None, which the shell reads as success, so it must not become 1.
        if exc.code is None:
            code = 0
        elif isinstance(exc.code, int):
            code = exc.code
        else:
            code = 1  # a message, which the interpreter prints and exits 1
    except Exception:  # noqa: BLE001 — any crash must keep its output
        # Attach whatever was printed before the crash: without this the
        # traceback arrives with no sign of what the run was doing.
        raise AssertionError(
            f"dsl {' '.join(args)} raised\n"
            f"stdout:\n{out.getvalue()}\nstderr:\n{err.getvalue()}"
        ) from None
    finally:
        os.chdir(before)
    return Result(code, out.getvalue(), err.getvalue())


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
def test_e2e_case(case_dir, tmp_path):
    raw = yaml.safe_load((case_dir / "scenario.yaml").read_text())
    expect = raw.pop("_expect", {})
    run_opts = raw.pop("_run", {})

    # Copy the whole case directory (scenario + any sibling fixture files).
    work = tmp_path / "work"
    shutil.copytree(case_dir, work)
    scenario = work / "scenario.yaml"
    scenario.write_text(yaml.safe_dump(raw))

    out = tmp_path / "out"
    extra_args = run_opts.get("extra_args", [])
    result = run_dsl("run", str(scenario), "-o", str(out), *extra_args, cwd=work)

    assert result.returncode == expect.get("exit_code", 0), (
        f"exit code {result.returncode}, expected {expect.get('exit_code', 0)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    for name in expect.get("files_exist", []):
        assert (out / name).is_file(), f"expected {name} to exist in {out}"
    for name in expect.get("files_absent", []):
        assert not (out / name).exists(), f"expected {name} to NOT exist in {out}"

    if "stderr_contains" in expect:
        assert expect["stderr_contains"] in result.stderr
    if "stderr_not_contains" in expect:
        assert expect["stderr_not_contains"] not in result.stderr

    df = None
    if (out / "simulated_data.csv").is_file():
        df = pd.read_csv(out / "simulated_data.csv")

    for col in expect.get("column_exists", []):
        assert df is not None and col in df.columns, f"missing column {col}"
    for col, values in expect.get("columns", {}).items():
        assert df is not None
        assert df[col].tolist() == values, f"column {col} mismatch"
    for col, prop in expect.get("column_all", {}).items():
        assert df is not None and col in df.columns, f"missing column {col}"
        check = COLUMN_PROPERTIES[prop]
        assert check(df[col]), f"column {col} fails property {prop!r}: {df[col].tolist()}"

    report_expect = expect.get("report")
    if report_expect is not None:
        report_path = out / "folds" / "report.json"
        assert report_path.is_file(), "expected folds/report.json"
        report = json.loads(report_path.read_text())

        if "folds" in report_expect:
            assert len(report["folds"]) == report_expect["folds"]
        if "total_events" in report_expect:
            assert report["total_events"] == report_expect["total_events"]
        if "warnings" in report_expect:
            wanted = report_expect["warnings"]
            if isinstance(wanted, int):
                assert len(report["warnings"]) == wanted, report["warnings"]
            else:
                assert any(wanted in w for w in report["warnings"]), (
                    f"no warning containing {wanted!r}: {report['warnings']}"
                )
        for index, series_counts in report_expect.get("fold_events", {}).items():
            fold = report["folds"][int(index)]
            for series, kinds in series_counts.items():
                for kind, sides in kinds.items():
                    for side, count in sides.items():
                        entry = fold[side]["events"].get(series, {}).get(kind)
                        actual = entry["count"] if entry else 0
                        assert actual == count, (
                            f"fold {index} {side} {series}/{kind}: "
                            f"{actual}, expected {count}"
                        )

    if expect.get("reproduce"):
        repro = tmp_path / "repro"
        result2 = run_dsl("run", str(out / "metadata.json"), "-o", str(repro), cwd=work)
        assert result2.returncode == 0, result2.stderr
        a = (out / "simulated_data.csv").read_text()
        b = (repro / "simulated_data.csv").read_text()
        assert a == b, "reproduction from metadata.json was not byte-identical"


@pytest.mark.parametrize(
    "raised, expected",
    [
        (SystemExit(None), 0),  # a bare sys.exit() is success
        (SystemExit(0), 0),
        (SystemExit(2), 2),  # argparse's usage error
        (SystemExit("a message"), 1),  # printed, then exit 1
    ],
)
def test_run_dsl_reads_exit_codes_the_way_a_shell_does(
    raised, expected, monkeypatch, tmp_path
):
    """A run reported as failing when it succeeded would be worse than slow."""
    monkeypatch.setitem(
        globals(), "main", lambda argv: (_ for _ in ()).throw(raised)
    )
    assert run_dsl("run", "x", cwd=tmp_path).returncode == expected


def test_a_crash_reports_what_the_run_had_printed(monkeypatch, tmp_path):
    """Captured output must not be swallowed with the traceback."""
    def boom(argv):
        print("got this far")
        raise RuntimeError("unexpected")

    monkeypatch.setitem(globals(), "main", boom)
    with pytest.raises(AssertionError, match="got this far"):
        run_dsl("run", "x", cwd=tmp_path)


def test_the_console_script_entry_point_works(tmp_path):
    """`dsl` must run as an installed command, not just as an import.

    Every other case calls main() directly for speed, so this is the only
    thing standing between a broken [project.scripts] entry and a release
    nobody can run.
    """
    if shutil.which("dsl") is None:
        pytest.skip("dsl is not on PATH — run pytest inside the venv")

    # Copy the whole case directory, so a scenario with sibling data files
    # still resolves them if this ever points at a different case.
    work = tmp_path / "work"
    shutil.copytree(CASES_DIR / "basic_run", work)
    scenario = work / "scenario.yaml"
    raw = yaml.safe_load(scenario.read_text())
    raw.pop("_expect", None)
    raw.pop("_run", None)
    scenario.write_text(yaml.safe_dump(raw))
    result = subprocess.run(
        ["dsl", "run", str(scenario), "-o", str(tmp_path / "out")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "simulated_data.csv").is_file()
