"""A scenario can start from a bundled preset and say only what differs.

`preset: climate` loads src/dsl/presets/climate.yaml and the scenario is
merged over it. The merge is the whole design: a plain dict merge would
replace the `series:` list wholesale and silently delete the series a
scenario did not mention, so lists of named entries merge per entry.
"""
import pytest
import yaml

from dsl._presets import PRESET_KEY, available, merge, resolve_preset
from dsl.cli import main

# --- The merge rule, in isolation. ------------------------------------------


def test_a_scalar_in_the_scenario_wins():
    assert merge({"n_total": 60, "seed": 1}, {"n_total": 120}) == {
        "n_total": 120, "seed": 1,
    }


def test_a_key_only_the_preset_has_is_kept():
    assert merge({"period": "monthly", "seed": 1}, {"seed": 2}) == {
        "period": "monthly", "seed": 2,
    }


def test_nested_mappings_merge_rather_than_replace():
    base = {"locations": {"north": {"population": 300000},
                          "south": {"population": 180000}}}
    over = {"locations": {"north": {"population": 400000}}}
    assert merge(base, over) == {
        "locations": {"north": {"population": 400000},
                      "south": {"population": 180000}},
    }


def test_a_named_list_merges_per_entry_and_keeps_the_rest():
    """The reason this module exists: a dict merge would drop temperature
    and disease_cases because the scenario only mentioned rainfall."""
    base = {"series": [
        {"name": "rainfall", "generate": "seasonal_spike",
         "params": {"spike_height": 30, "clamp_min": 0}},
        {"name": "temperature", "generate": "seasonal_smooth"},
        {"name": "disease_cases", "counts": {}},
    ]}
    over = {"series": [{"name": "rainfall", "params": {"spike_height": 40}}]}

    merged = merge(base, over)
    assert [spec["name"] for spec in merged["series"]] == [
        "rainfall", "temperature", "disease_cases",
    ]
    # Only the one value changed; the preset's other params survive.
    assert merged["series"][0]["params"] == {"spike_height": 40, "clamp_min": 0}
    assert merged["series"][0]["generate"] == "seasonal_spike"


def test_a_new_name_is_appended_in_scenario_order():
    base = {"series": [{"name": "rainfall"}]}
    over = {"series": [{"name": "humidity"}, {"name": "rainfall", "unit": "mm"}]}
    merged = merge(base, over)
    assert [spec["name"] for spec in merged["series"]] == ["rainfall", "humidity"]
    assert merged["series"][0]["unit"] == "mm"


def test_depends_on_merges_on_its_series_key():
    base = {"series": [{"name": "disease_cases", "depends_on": [
        {"series": "rainfall", "lag": 2, "weight": 1.0},
        {"series": "temperature", "lag": 1, "weight": 0.4},
    ]}]}
    over = {"series": [{"name": "disease_cases", "depends_on": [
        {"series": "rainfall", "lag": 3},
    ]}]}
    merged = merge(base, over)["series"][0]["depends_on"]
    assert merged == [
        {"series": "rainfall", "lag": 3, "weight": 1.0},
        {"series": "temperature", "lag": 1, "weight": 0.4},
    ]


def test_a_list_without_identifying_keys_is_replaced():
    # Nothing to match entries on, so the scenario's list is the list.
    base = {"series": [{"name": "d", "transforms": [{"name": "lag"}]}],
            "xs": [1, 2, 3]}
    over = {"xs": [9]}
    assert merge(base, over)["xs"] == [9]


def test_a_transform_list_merges_on_name_too():
    base = {"series": [{"name": "d", "transforms": [
        {"name": "threshold", "params": {"at": 5}},
        {"name": "lag", "params": {"n": 2}},
    ]}]}
    over = {"series": [{"name": "d", "transforms": [
        {"name": "lag", "params": {"n": 4}},
    ]}]}
    merged = merge(base, over)["series"][0]["transforms"]
    assert merged == [
        {"name": "threshold", "params": {"at": 5}},
        {"name": "lag", "params": {"n": 4}},
    ]


def test_merging_does_not_mutate_either_input():
    base = {"series": [{"name": "r", "params": {"a": 1}}]}
    over = {"series": [{"name": "r", "params": {"a": 2}}]}
    merge(base, over)
    assert base["series"][0]["params"] == {"a": 1}
    assert over["series"][0]["params"] == {"a": 2}


def test_an_explicit_null_in_the_scenario_wins_over_a_preset_value():
    # `split: ~` is how a scenario turns a preset's block off; a dict merge
    # that skipped None would make it impossible to override.
    assert merge({"split": {"kind": "time"}}, {"split": None}) == {"split": None}


# --- Resolution: finding the preset and reporting what went wrong. ----------


def test_a_scenario_without_a_preset_is_returned_unchanged():
    scenario = {"n_total": 36}
    assert resolve_preset(scenario) == scenario


def test_an_unknown_preset_names_what_is_available():
    with pytest.raises(ValueError) as excinfo:
        resolve_preset({PRESET_KEY: "nope"})
    message = str(excinfo.value)
    assert "nope" in message
    # The list is the actionable part — a typo needs the real names.
    for name in available():
        assert name in message


def test_a_near_miss_is_suggested():
    with pytest.raises(ValueError, match="did you mean"):
        resolve_preset({PRESET_KEY: "climat"})


def test_the_key_is_removed_after_resolution():
    """`preset:` is not a scenario field, so the schema must never see it."""
    resolved = resolve_preset({PRESET_KEY: "climate", "n_total": 120})
    assert PRESET_KEY not in resolved


def test_a_preset_cannot_itself_use_a_preset(tmp_path, monkeypatch):
    import dsl._presets as presets_module

    (tmp_path / "chained.yaml").write_text(yaml.safe_dump(
        {PRESET_KEY: "climate", "n_total": 10},
    ))
    monkeypatch.setattr(presets_module, "PRESETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="cannot use a preset"):
        resolve_preset({PRESET_KEY: "chained"})


@pytest.mark.parametrize("bad", [3, [1], {"a": 1}, True],
                         ids=["int", "list", "dict", "bool"])
def test_a_preset_name_must_be_a_string(bad):
    with pytest.raises(ValueError, match="must be the name"):
        resolve_preset({PRESET_KEY: bad})


def test_at_least_one_preset_ships():
    assert available(), "no presets bundled"
    assert "climate" in available()


# --- End to end through the CLI. -------------------------------------------


def test_a_two_line_scenario_runs(tmp_path):
    """The pitch: the base lives in the preset, the scenario is the diff."""
    path = tmp_path / "s.yaml"
    path.write_text("preset: climate\nn_total: 120\n")
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0

    import pandas as pd
    df = pd.read_csv(out / "simulated_data.csv")
    assert len(df) == 120 * df["location"].nunique()


def test_overriding_one_param_keeps_the_rest_of_the_preset(tmp_path):
    import json

    plain = tmp_path / "plain.yaml"
    plain.write_text("preset: climate\n")
    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(
        "preset: climate\n"
        "series:\n"
        "  - name: rainfall\n"
        "    params: { spike_height: 99 }\n"
    )
    assert main(["run", str(plain), "-o", str(tmp_path / "a")]) == 0
    assert main(["run", str(tuned), "-o", str(tmp_path / "b")]) == 0

    a = json.loads((tmp_path / "a" / "metadata.json").read_text())["scenario"]
    b = json.loads((tmp_path / "b" / "metadata.json").read_text())["scenario"]
    assert [s["name"] for s in a["series"]] == [s["name"] for s in b["series"]]
    rain_a = next(s for s in a["series"] if s["name"] == "rainfall")
    rain_b = next(s for s in b["series"] if s["name"] == "rainfall")
    assert rain_b["params"]["spike_height"] == 99
    assert rain_a["params"]["spike_height"] != 99
    # Everything else about rainfall is the preset's.
    assert rain_a["generate"] == rain_b["generate"]


def test_the_metadata_records_the_resolved_scenario_not_the_preset_name(tmp_path):
    """A dataset has to be reproducible from its own metadata, without the
    preset — otherwise it depends on a file outside the folder."""
    import json

    path = tmp_path / "s.yaml"
    path.write_text("preset: climate\nn_total: 48\n")
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0

    scenario = json.loads((out / "metadata.json").read_text())["scenario"]
    assert "preset" not in scenario
    assert scenario["n_total"] == 48
    assert scenario["series"], "the preset's series did not survive into metadata"

    repro = tmp_path / "repro"
    assert main(["run", str(out / "metadata.json"), "-o", str(repro)]) == 0
    assert (repro / "simulated_data.csv").read_bytes() == (
        out / "simulated_data.csv"
    ).read_bytes()


def test_an_unknown_preset_fails_cleanly(tmp_path, capsys):
    path = tmp_path / "s.yaml"
    path.write_text("preset: tropcal\n")
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "tropcal" in err
    assert not out.exists()


def test_a_preset_can_be_swept(tmp_path):
    """Presets and sweeps are independent: resolution happens first, so a
    swept value in the scenario still expands."""
    path = tmp_path / "s.yaml"
    path.write_text("preset: climate\nn_total: 36\nseed: { sweep: [1, 2] }\n")
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["seed_1", "seed_2"]


def test_every_bundled_preset_runs_on_its_own(tmp_path):
    """A preset is a complete scenario, so `preset: <name>` alone must run."""
    for name in available():
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"preset: {name}\n")
        assert main(["run", str(path), "-o", str(tmp_path / name)]) == 0, name
        assert (tmp_path / name / "simulated_data.csv").is_file()


def test_a_list_mixing_scalars_and_mappings_is_replaced():
    # Nothing to match a bare scalar on, so per-entry merging cannot
    # apply to the list at all.
    base = {"xs": [{"name": "a"}, 7]}
    over = {"xs": [{"name": "a", "v": 1}]}
    assert merge(base, over)["xs"] == [{"name": "a", "v": 1}]


def test_a_list_of_unnamed_mappings_is_replaced():
    # Mappings, but with nothing that identifies them — matching by
    # position would make the scenario's meaning depend on the preset's
    # ordering.
    base = {"xs": [{"a": 1}, {"b": 2}]}
    over = {"xs": [{"c": 3}]}
    assert merge(base, over)["xs"] == [{"c": 3}]


def test_dsl_list_names_the_bundled_presets(capsys):
    """`dsl list` is the discovery command, and a preset is the one thing
    you can use without writing any of it."""
    assert main(["list"]) == 0
    printed = capsys.readouterr().out
    assert "presets" in printed
    for name in available():
        assert name in printed


def test_overriding_one_location_keeps_the_others(tmp_path):
    """A mapping merges key by key, so changing north's population must
    not drop the preset's south."""
    import pandas as pd

    path = tmp_path / "s.yaml"
    path.write_text(
        "preset: climate\nn_total: 24\n"
        "locations:\n  north: { population: 100000 }\n"
    )
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0

    df = pd.read_csv(out / "simulated_data.csv")
    assert sorted(df["location"].unique()) == ["north", "south"]
    populations = df.groupby("location")["population"].first().to_dict()
    assert populations == {"north": 100000, "south": 180000}


# --- The bundled set. -------------------------------------------------------


def test_the_bundled_presets_are_the_documented_three():
    assert available() == ["climate", "evaluation", "minimal"]


def test_the_minimal_preset_is_a_one_line_scenario(tmp_path):
    import pandas as pd

    path = tmp_path / "s.yaml"
    path.write_text("preset: minimal\n")
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0
    df = pd.read_csv(out / "simulated_data.csv")
    assert list(df.columns) == [
        "time_period", "location", "rainfall", "disease_cases", "population",
    ]


def test_the_evaluation_preset_writes_folds_and_a_report(tmp_path):
    """It exists so folds do not have to be retyped, so the folds and the
    report are the thing to assert."""
    import json

    path = tmp_path / "s.yaml"
    path.write_text("preset: evaluation\n")
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0

    assert sorted(p.name for p in (out / "folds").iterdir()) == [
        *[f"fold_{i}" for i in range(5)], "report.json", "report.md",
    ]
    report = json.loads((out / "folds" / "report.json").read_text())
    assert len(report["folds"]) == 5
    # The point of the preset: every fold's test half can actually measure
    # the storm the scenario plants.
    assert not report["warnings"], report["warnings"]


@pytest.mark.parametrize("name", available())
def test_every_shipped_preset_runs_as_is(tmp_path, name):
    """A preset exists to be a working starting point, so each one has to
    run with nothing added. This is the catch-all that would have caught
    `evaluation` failing on its first ordinary use."""
    path = tmp_path / "s.yaml"
    path.write_text(f"preset: {name}\n")
    assert main(["run", str(path), "-o", str(tmp_path / "out")]) == 0


@pytest.mark.parametrize("k", [2, 3, 5])
def test_the_evaluation_preset_follows_the_fold_count(tmp_path, k):
    """`per_fold` was a five-entry list, so any k other than 5 failed with
    an error naming `storm` — an event the user never wrote. The preset's
    own comment promised the storms move with k; now they do."""
    path = tmp_path / "s.yaml"
    path.write_text(f"preset: evaluation\nsplit: {{ k: {k} }}\n")
    out = tmp_path / "out"
    assert main(["run", str(path), "-o", str(out)]) == 0
    assert len(list((out / "folds").glob("fold_*"))) == k
