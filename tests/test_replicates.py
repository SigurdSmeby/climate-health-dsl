"""Feature 14: multiple seeded replicates via `dsl run --replicates N`.

Each replicate i uses seed = base_seed + i and writes its own subfolder with a
full dataset + metadata (independently reproducible). N=1 is the single-run
default. TDD: written before the CLI flag exists.
"""
import pandas as pd
import yaml

from dsl.cli import main

EXAMPLE = "examples/basic_scenario.yaml"


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
        json.loads((out / f"rep_{i:02d}" / "metadata.json").read_text())["seed"]
        for i in range(3)
    ]
    # basic_scenario.yaml has seed 0 → replicates use 0, 1, 2.
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
