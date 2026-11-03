"""Sweep expansion: one scenario dict with {sweep: ...} values -> N dicts.

Expansion happens on the loaded dict, BEFORE the schema sees it, so a swept
value never has to be a legal value of its own field (`lag: {sweep: [1,2]}`
is not an int). These tests cover the two accepted spellings, the label the
output directory is named after, and the errors a malformed sweep gives.
"""
import json

import pytest
import yaml

from dsl._sweep import SWEEP_KEY, expand_sweeps, has_sweep
from dsl.cli import main


def test_no_sweep_is_one_unlabelled_run():
    scenario = {"n_total": 36, "seed": 42}
    assert expand_sweeps(scenario) == [("", scenario)]
    assert has_sweep(scenario) is False


def test_list_form_expands_to_one_run_per_value():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: [10, 20, 30]}})
    assert [label for label, _ in runs] == [
        "n_total_10", "n_total_20", "n_total_30",
    ]
    assert [cfg["n_total"] for _, cfg in runs] == [10, 20, 30]


def test_range_form_is_inclusive_of_the_end():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: {"from": 10, "to": 50, "step": 10}}})
    assert [cfg["n_total"] for _, cfg in runs] == [10, 20, 30, 40, 50]


def test_range_form_stops_before_overshooting_the_end():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: {"from": 1, "to": 10, "step": 4}}})
    assert [cfg["n_total"] for _, cfg in runs] == [1, 5, 9]


def test_range_form_step_defaults_to_one():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: {"from": 3, "to": 6}}})
    assert [cfg["n_total"] for _, cfg in runs] == [3, 4, 5, 6]


def test_two_sweeps_give_the_cartesian_product_in_document_order():
    runs = expand_sweeps({
        "series": [
            {"name": "rainfall", "params": {"spike_height": {SWEEP_KEY: [10, 30]}}},
            {"name": "disease_cases", "depends_on": [
                {"series": "rainfall", "lag": {SWEEP_KEY: [1, 2]}},
            ]},
        ],
    })
    assert [label for label, _ in runs] == [
        "spike_height_10__lag_1",
        "spike_height_10__lag_2",
        "spike_height_30__lag_1",
        "spike_height_30__lag_2",
    ]
    first = runs[0][1]
    assert first["series"][0]["params"]["spike_height"] == 10
    assert first["series"][1]["depends_on"][0]["lag"] == 1


def test_expansion_does_not_mutate_the_input():
    scenario = {"n_total": {SWEEP_KEY: [10, 20]}}
    expand_sweeps(scenario)
    assert scenario == {"n_total": {SWEEP_KEY: [10, 20]}}


def test_sweeping_the_seed_replaces_replicates():
    runs = expand_sweeps({"seed": {SWEEP_KEY: [42, 43, 44]}})
    assert [label for label, _ in runs] == ["seed_42", "seed_43", "seed_44"]
    assert [cfg["seed"] for _, cfg in runs] == [42, 43, 44]


def test_two_sweeps_on_the_same_field_name_are_qualified_by_their_parent():
    runs = expand_sweeps({
        "series": [
            {"name": "disease_cases", "depends_on": [
                {"series": "rainfall", "lag": {SWEEP_KEY: [1, 2]}},
                {"series": "temperature", "lag": {SWEEP_KEY: [5]}},
            ]},
        ],
    })
    assert [label for label, _ in runs] == [
        "rainfall_lag_1__temperature_lag_5",
        "rainfall_lag_2__temperature_lag_5",
    ]


def test_float_values_keep_their_dot_out_of_the_label():
    # A directory name with a '.' or '-' in it is what sweep.json exists to
    # avoid parsing; the label itself stays filesystem-plain regardless.
    runs = expand_sweeps({"series": [
        {"name": "x", "params": {"weight": {SWEEP_KEY: [0.5, -1.5]}}},
    ]})
    assert [label for label, _ in runs] == ["weight_0p5", "weight_m1p5"]
    assert [cfg["series"][0]["params"]["weight"] for _, cfg in runs] == [0.5, -1.5]


def test_string_values_are_slugified():
    runs = expand_sweeps({"period": {SWEEP_KEY: ["monthly", "weekly"]}})
    assert [label for label, _ in runs] == ["period_monthly", "period_weekly"]


@pytest.mark.parametrize("bad,message", [
    ({SWEEP_KEY: []}, "at least one value"),
    ({SWEEP_KEY: {"to": 5}}, "'from'"),
    ({SWEEP_KEY: {"from": 5}}, "'to'"),
    ({SWEEP_KEY: {"from": 1, "to": 5, "step": 0}}, "step"),
    ({SWEEP_KEY: {"from": 5, "to": 1, "step": 1}}, "'to'"),
    ({SWEEP_KEY: "10,20"}, "a list"),
    ({SWEEP_KEY: {"from": 1, "to": 5, "stepp": 1}}, "unknown"),
], ids=["empty", "no-from", "no-to", "zero-step", "backwards", "scalar", "typo"])
def test_malformed_sweeps_are_rejected_by_name(bad, message):
    with pytest.raises(ValueError, match=message):
        expand_sweeps({"n_total": bad})


def test_a_sweep_alongside_other_keys_is_not_a_sweep():
    # {sweep: [...], other: 1} is a generator's param dict that happens to
    # hold a key called sweep, not a sweep spec. Left alone for the schema.
    scenario = {"params": {SWEEP_KEY: [1, 2], "other": 1}}
    assert expand_sweeps(scenario) == [("", scenario)]


def test_has_sweep_finds_a_nested_one():
    assert has_sweep({"series": [{"params": {"x": {SWEEP_KEY: [1]}}}]}) is True


# --- The run path: a sweep in the YAML is the trigger, no flag. ---------------

_SCENARIO = {
    "period": "monthly",
    "n_total": 24,
    "seed": 42,
    "locations": {"loc": {"population": 100000}},
    "series": [
        {"name": "rainfall", "generate": "seasonal_spike",
         "params": {"spike_height": {SWEEP_KEY: [10, 30]}}},
        {"name": "disease_cases", "counts": {},
         "depends_on": [{"series": "rainfall", "lag": {SWEEP_KEY: [1, 2]}}]},
    ],
}


def _write(tmp_path, scenario) -> str:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(scenario))
    return str(path)


def test_run_writes_one_directory_per_combination(tmp_path):
    out = tmp_path / "out"
    assert main(["run", _write(tmp_path, _SCENARIO), "-o", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
        "spike_height_10__lag_1",
        "spike_height_10__lag_2",
        "spike_height_30__lag_1",
        "spike_height_30__lag_2",
    ]
    for child in out.iterdir():
        if child.is_dir():
            assert (child / "simulated_data.csv").is_file()
            assert (child / "metadata.json").is_file()


def test_each_run_records_its_own_swept_value_in_metadata(tmp_path):
    out = tmp_path / "out"
    main(["run", _write(tmp_path, _SCENARIO), "-o", str(out)])
    meta = json.loads((out / "spike_height_30__lag_2" / "metadata.json").read_text())
    series = meta["scenario"]["series"]
    assert series[0]["params"]["spike_height"] == 30
    assert series[1]["depends_on"][0]["lag"] == 2


def test_swept_runs_differ_from_each_other(tmp_path):
    out = tmp_path / "out"
    main(["run", _write(tmp_path, _SCENARIO), "-o", str(out)])
    a = (out / "spike_height_10__lag_1" / "simulated_data.csv").read_text()
    b = (out / "spike_height_30__lag_1" / "simulated_data.csv").read_text()
    assert a != b


def test_a_scenario_without_a_sweep_still_writes_one_run_in_place(tmp_path):
    plain = {**_SCENARIO, "series": [
        {"name": "rainfall", "generate": "seasonal_spike"},
        {"name": "disease_cases", "counts": {},
         "depends_on": [{"series": "rainfall", "lag": 1}]},
    ]}
    out = tmp_path / "out"
    assert main(["run", _write(tmp_path, plain), "-o", str(out)]) == 0
    assert (out / "simulated_data.csv").is_file()


def test_a_swept_run_reproduces_from_its_own_metadata(tmp_path):
    out = tmp_path / "out"
    main(["run", _write(tmp_path, _SCENARIO), "-o", str(out)])
    original = out / "spike_height_30__lag_2"
    again = tmp_path / "again"
    assert main(["run", str(original / "metadata.json"), "-o", str(again)]) == 0
    assert (again / "simulated_data.csv").read_text() == (
        original / "simulated_data.csv"
    ).read_text()


def test_sweep_with_replicates_is_refused(tmp_path):
    out = tmp_path / "out"
    assert main([
        "run", _write(tmp_path, _SCENARIO), "-o", str(out), "-n", "3",
    ]) == 1


def test_a_malformed_sweep_fails_before_anything_is_written(tmp_path):
    bad = {**_SCENARIO, "n_total": {SWEEP_KEY: []}}
    out = tmp_path / "out"
    assert main(["run", _write(tmp_path, bad), "-o", str(out)]) == 1
    assert not out.exists() or not any(out.iterdir())


def test_a_stale_run_directory_is_cleared_by_the_next_sweep(tmp_path):
    out = tmp_path / "out"
    main(["run", _write(tmp_path, _SCENARIO), "-o", str(out)])
    # A second sweep over fewer values must not leave the first one's extras.
    narrower = {**_SCENARIO, "series": [
        {"name": "rainfall", "generate": "seasonal_spike",
         "params": {"spike_height": {SWEEP_KEY: [10]}}},
        {"name": "disease_cases", "counts": {},
         "depends_on": [{"series": "rainfall", "lag": {SWEEP_KEY: [1]}}]},
    ]}
    main(["run", _write(tmp_path, narrower), "-o", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["spike_height_10__lag_1"]


def test_a_sweep_refuses_a_directory_holding_foreign_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "thesis.tex").write_text("mine")
    assert main(["run", _write(tmp_path, _SCENARIO), "-o", str(out)]) == 1
    assert (out / "thesis.tex").is_file()
