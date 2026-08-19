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

Both keys (and all their sub-keys) are optional; an empty ``_expect: {}``
just means "must run and exit 0". ``columns`` is for deterministic (seeded)
values; ``column_all`` is for a property that must hold on every row when
exact values aren't practical (e.g. random counts).
"""
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

CASES_DIR = Path(__file__).parent / "cases"
CASE_DIRS = sorted(p for p in CASES_DIR.iterdir() if (p / "scenario.yaml").is_file())

COLUMN_PROPERTIES = {
    # NaN is allowed (e.g. the lag warm-up rows) — only real values are checked.
    "non_negative_int": lambda s: ((s.dropna() >= 0) & (s.dropna() % 1 == 0)).all(),
    "non_negative": lambda s: (s.dropna() >= 0).all(),
}


def run_dsl(*args, cwd):
    # Calls the installed `dsl` console-script entry point (not `python -m
    # dsl.cli`), so these tests also catch a broken pyproject.toml
    # [project.scripts] entry or packaging misconfiguration.
    return subprocess.run(
        ["dsl", *args],
        cwd=cwd, capture_output=True, text=True, check=False,
    )


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

    if expect.get("reproduce"):
        repro = tmp_path / "repro"
        result2 = run_dsl("run", str(out / "metadata.json"), "-o", str(repro), cwd=work)
        assert result2.returncode == 0, result2.stderr
        a = (out / "simulated_data.csv").read_text()
        b = (repro / "simulated_data.csv").read_text()
        assert a == b, "reproduction from metadata.json was not byte-identical"
