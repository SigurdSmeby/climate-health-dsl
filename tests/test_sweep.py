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
    assert expand_sweeps(scenario) == [("", scenario, {})]
    assert has_sweep(scenario) is False


def test_list_form_expands_to_one_run_per_value():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: [10, 20, 30]}})
    assert [label for label, _, _ in runs] == [
        "n_total_10", "n_total_20", "n_total_30",
    ]
    assert [cfg["n_total"] for _, cfg, _ in runs] == [10, 20, 30]


def test_range_form_is_inclusive_of_the_end():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: {"from": 10, "to": 50, "step": 10}}})
    assert [cfg["n_total"] for _, cfg, _ in runs] == [10, 20, 30, 40, 50]


def test_range_form_stops_before_overshooting_the_end():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: {"from": 1, "to": 10, "step": 4}}})
    assert [cfg["n_total"] for _, cfg, _ in runs] == [1, 5, 9]


def test_range_form_step_defaults_to_one():
    runs = expand_sweeps({"n_total": {SWEEP_KEY: {"from": 3, "to": 6}}})
    assert [cfg["n_total"] for _, cfg, _ in runs] == [3, 4, 5, 6]


def test_two_sweeps_give_the_cartesian_product_in_document_order():
    runs = expand_sweeps({
        "series": [
            {"name": "rainfall", "params": {"spike_height": {SWEEP_KEY: [10, 30]}}},
            {"name": "disease_cases", "depends_on": [
                {"series": "rainfall", "lag": {SWEEP_KEY: [1, 2]}},
            ]},
        ],
    })
    assert [label for label, _, _ in runs] == [
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
    assert [label for label, _, _ in runs] == ["seed_42", "seed_43", "seed_44"]
    assert [cfg["seed"] for _, cfg, _ in runs] == [42, 43, 44]


def test_two_sweeps_on_the_same_field_name_are_qualified_by_their_parent():
    runs = expand_sweeps({
        "series": [
            {"name": "disease_cases", "depends_on": [
                {"series": "rainfall", "lag": {SWEEP_KEY: [1, 2]}},
                {"series": "temperature", "lag": {SWEEP_KEY: [5]}},
            ]},
        ],
    })
    assert [label for label, _, _ in runs] == [
        "rainfall_lag_1__temperature_lag_5",
        "rainfall_lag_2__temperature_lag_5",
    ]


def test_float_values_keep_their_dot_out_of_the_label():
    # A directory name with a '.' or '-' in it is what sweep.json exists to
    # avoid parsing; the label stays filesystem-plain regardless.
    runs = expand_sweeps({"series": [
        {"name": "x", "params": {"weight": {SWEEP_KEY: [0.5, -1.5]}}},
    ]})
    assert [label for label, _, _ in runs] == ["weight_0p5", "weight_m1p5"]
    assert [cfg["series"][0]["params"]["weight"] for _, cfg, _ in runs] == [0.5, -1.5]


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
    assert expand_sweeps(scenario) == [("", scenario, {})]


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


def test_the_replicates_flag_is_gone(tmp_path):
    # Sweeping the seed replaced it; an unrecognised flag is an argparse
    # error (exit 2), so nothing is generated before it is reported.
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        main(["run", _write(tmp_path, _SCENARIO), "-o", str(out), "-n", "3"])
    assert excinfo.value.code == 2
    assert not out.exists()


def test_sweeping_the_seed_writes_one_run_per_seed(tmp_path):
    scenario = {
        **_SCENARIO,
        "seed": {SWEEP_KEY: [42, 43, 44]},
        "series": [
            {"name": "rainfall", "generate": "seasonal_spike"},
            {"name": "disease_cases", "counts": {},
             "depends_on": [{"series": "rainfall", "lag": 1}]},
        ],
    }
    out = tmp_path / "out"
    assert main(["run", _write(tmp_path, scenario), "-o", str(out)]) == 0
    assert sorted(
        p.name for p in out.iterdir() if p.is_dir()
    ) == ["seed_42", "seed_43", "seed_44"]
    # Different seeds, different draws — what --replicates was for.
    data = [
        (out / f"seed_{s}" / "simulated_data.csv").read_text() for s in (42, 43, 44)
    ]
    assert len(set(data)) == 3


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
    assert sorted(
        p.name for p in out.iterdir() if p.is_dir()
    ) == ["spike_height_10__lag_1"]


def test_a_sweep_refuses_a_directory_holding_foreign_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "thesis.tex").write_text("mine")
    assert main(["run", _write(tmp_path, _SCENARIO), "-o", str(out)]) == 1
    assert (out / "thesis.tex").is_file()


# --- Review follow-ups: numeric-only values, exact float ranges, unique labels. ---


def test_a_float_range_reaches_its_endpoint():
    # (0.3 - 0) / 0.1 is 2.9999… in binary floating point, so truncating
    # dropped the endpoint and a weight sweep silently lost its last value.
    runs = expand_sweeps({"w": {SWEEP_KEY: {"from": 0, "to": 0.3, "step": 0.1}}})
    assert [round(cfg["w"], 10) for _, cfg, _ in runs] == [0.0, 0.1, 0.2, 0.3]


def test_a_float_range_is_inclusive_like_an_integer_one():
    runs = expand_sweeps({"w": {SWEEP_KEY: {"from": 0.5, "to": 2.0, "step": 0.5}}})
    assert [round(cfg["w"], 10) for _, cfg, _ in runs] == [0.5, 1.0, 1.5, 2.0]


def test_a_float_range_still_stops_before_overshooting():
    runs = expand_sweeps({"w": {SWEEP_KEY: {"from": 0, "to": 1, "step": 0.3}}})
    assert [round(cfg["w"], 10) for _, cfg, _ in runs] == [0.0, 0.3, 0.6, 0.9]


def test_a_negative_step_counts_down_inclusively():
    runs = expand_sweeps({"n": {SWEEP_KEY: {"from": 3, "to": 1, "step": -1}}})
    assert [cfg["n"] for _, cfg, _ in runs] == [3, 2, 1]


@pytest.mark.parametrize("value", ["monthly", None, True, [1], {"a": 1}],
                         ids=["str", "null", "bool", "list", "dict"])
def test_only_numbers_can_be_swept(value):
    with pytest.raises(ValueError, match="number"):
        expand_sweeps({"n_total": {SWEEP_KEY: [1, value]}})


@pytest.mark.parametrize("bound", ["from", "to", "step"])
def test_a_range_bound_must_be_a_number(bound):
    spec = {"from": 1, "to": 5, "step": 1, bound: "x"}
    with pytest.raises(ValueError, match="number"):
        expand_sweeps({"n_total": {SWEEP_KEY: spec}})


def test_a_repeated_value_in_a_sweep_is_rejected():
    # Two identical values would give two runs one directory name, so the
    # second silently overwrote the first: N combinations, fewer than N
    # datasets, no warning. It is a typo every time.
    with pytest.raises(ValueError, match="repeat"):
        expand_sweeps({"n_total": {SWEEP_KEY: [10, 20, 10]}})


def test_equal_values_of_different_numeric_types_also_repeat():
    with pytest.raises(ValueError, match="repeat"):
        expand_sweeps({"n_total": {SWEEP_KEY: [2, 2.0]}})


def test_every_combination_gets_its_own_directory_name():
    runs = expand_sweeps({
        "series": [
            {"name": "a", "params": {"p": {SWEEP_KEY: [1, 2]}}},
            {"name": "b", "params": {"q": {SWEEP_KEY: [0.5, 1.5]}}},
        ],
    })
    labels = [label for label, _, _ in runs]
    assert len(set(labels)) == len(labels) == 4


def test_a_sweep_in_a_bare_list_element_is_expanded():
    # Found nothing here before, so the run silently became a single
    # unswept one — a no-op is the worst way to decline a sweep.
    runs = expand_sweeps({"xs": [{SWEEP_KEY: [1, 2]}]})
    assert [cfg["xs"][0] for _, cfg, _ in runs] == [1, 2]
    assert [label for label, _, _ in runs] == ["xs_1", "xs_2"]


# --- What each run was, as data rather than as a directory name. ------------


def test_expansion_reports_the_values_behind_each_label():
    runs = expand_sweeps({
        "series": [
            {"name": "rainfall", "params": {"spike_height": {SWEEP_KEY: [10, 30]}}},
            {"name": "disease_cases", "depends_on": [
                {"series": "rainfall", "lag": {SWEEP_KEY: [1, 2]}},
            ]},
        ],
    })
    assert [(label, params) for label, _, params in runs] == [
        ("spike_height_10__lag_1", {"spike_height": 10, "lag": 1}),
        ("spike_height_10__lag_2", {"spike_height": 10, "lag": 2}),
        ("spike_height_30__lag_1", {"spike_height": 30, "lag": 1}),
        ("spike_height_30__lag_2", {"spike_height": 30, "lag": 2}),
    ]


def test_an_unswept_scenario_reports_no_values():
    scenario = {"n_total": 36}
    assert expand_sweeps(scenario) == [("", scenario, {})]


def test_reported_values_use_the_qualified_name_when_one_repeats():
    runs = expand_sweeps({
        "series": [
            {"name": "disease_cases", "depends_on": [
                {"series": "rainfall", "lag": {SWEEP_KEY: [1]}},
                {"series": "temperature", "lag": {SWEEP_KEY: [5]}},
            ]},
        ],
    })
    _, _, params = runs[0]
    assert params == {"rainfall_lag": 1, "temperature_lag": 5}


def test_reported_values_keep_their_real_type_not_the_slug():
    runs = expand_sweeps({"w": {SWEEP_KEY: [0.5, -1.5]}})
    assert [params["w"] for _, _, params in runs] == [0.5, -1.5]
    assert [label for label, _, _ in runs] == ["w_0p5", "w_m1p5"]


# --- The index has to describe every swept value, exactly. ------------------


def test_two_unqualifiable_sweeps_on_one_field_are_rejected():
    """The label shows both values (xs_1__xs_7) but a dict keyed by field
    name can hold only one, so the index silently described a run by the
    wrong parameters — a join built on it would be wrong rather than
    broken. Neither sweep has a name/series sibling to qualify it, so
    there is no name to tell them apart by."""
    with pytest.raises(ValueError, match="cannot be told apart"):
        expand_sweeps({"xs": [{SWEEP_KEY: [1, 2]}, {SWEEP_KEY: [7, 8]}]})


def test_two_sweeps_on_one_field_are_fine_when_their_entries_are_named():
    runs = expand_sweeps({
        "series": [
            {"name": "disease_cases", "depends_on": [
                {"series": "rainfall", "lag": {SWEEP_KEY: [1, 2]}},
                {"series": "temperature", "lag": {SWEEP_KEY: [5]}},
            ]},
        ],
    })
    assert [params for _, _, params in runs] == [
        {"rainfall_lag": 1, "temperature_lag": 5},
        {"rainfall_lag": 2, "temperature_lag": 5},
    ]


def test_every_swept_value_appears_in_the_report():
    """One entry per sweep, always — the count is what makes the index
    trustworthy."""
    runs = expand_sweeps({
        "a": {SWEEP_KEY: [1, 2]},
        "b": {SWEEP_KEY: [3]},
        "c": {SWEEP_KEY: [4, 5]},
    })
    for label, _, params in runs:
        assert len(params) == 3, label
        assert label.count("__") == 2


@pytest.mark.parametrize("spec,expected", [
    ({"from": 0, "to": 0.3, "step": 0.1}, [0.0, 0.1, 0.2, 0.3]),
    ({"from": 0.1, "to": 0.7, "step": 0.2}, [0.1, 0.3, 0.5, 0.7]),
    ({"from": 0.05, "to": 0.2, "step": 0.05}, [0.05, 0.1, 0.15, 0.2]),
    ({"from": 1, "to": 3, "step": 1}, [1, 2, 3]),
], ids=["tenths", "offset-tenths", "hundredths", "ints-untouched"])
def test_a_float_range_gives_clean_values_not_binary_dust(spec, expected):
    """start + i*step is exact enough to compare against 'to' but stored
    0.30000000000000004, which then showed up in a directory name and in
    sweep.json."""
    runs = expand_sweeps({"w": {SWEEP_KEY: spec}})
    assert [params["w"] for _, _, params in runs] == expected


def test_a_drifting_float_does_not_reach_the_directory_name():
    runs = expand_sweeps({"w": {SWEEP_KEY: {"from": 0, "to": 0.3, "step": 0.1}}})
    assert [label for label, _, _ in runs] == ["w_0p0", "w_0p1", "w_0p2", "w_0p3"]


def test_an_int_range_keeps_integer_values():
    runs = expand_sweeps({"n": {SWEEP_KEY: {"from": 1, "to": 3}}})
    assert [params["n"] for _, _, params in runs] == [1, 2, 3]
    assert all(isinstance(params["n"], int) for _, _, params in runs)


def test_a_listed_float_is_left_exactly_as_written():
    # A value the author typed is theirs; only generated ones are rounded.
    runs = expand_sweeps({"w": {SWEEP_KEY: [0.1234567890123, 2.5]}})
    assert [params["w"] for _, _, params in runs] == [0.1234567890123, 2.5]


def test_a_range_written_in_exponent_form_is_left_unrounded():
    # repr(1e-05) has no literal decimals to count, so guessing a precision
    # would be worse than leaving these alone.
    runs = expand_sweeps({"w": {SWEEP_KEY: {"from": 0.0, "to": 2e-05, "step": 1e-05}}})
    assert [params["w"] for _, _, params in runs] == [0.0, 1e-05, 2e-05]


@pytest.mark.parametrize("endpoint", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_range_endpoint_is_rejected(endpoint):
    # Python counts inf and NaN as numbers, so they passed the isinstance
    # gate and reached the count of values as a raw OverflowError
    # ("cannot convert float infinity to integer") naming no field.
    with pytest.raises(ValueError, match="finite number"):
        expand_sweeps({"n_total": {SWEEP_KEY: {"from": 0, "to": endpoint}}})


def test_a_non_finite_step_is_rejected():
    with pytest.raises(ValueError, match="finite number"):
        expand_sweeps(
            {"n_total": {SWEEP_KEY: {"from": 0, "to": 10, "step": float("inf")}}}
        )


def test_a_non_finite_list_value_is_rejected():
    with pytest.raises(ValueError, match="finite number"):
        expand_sweeps({"n_total": {SWEEP_KEY: [1, float("nan"), 3]}})


def test_the_non_finite_message_names_the_field():
    with pytest.raises(ValueError, match="'weight'"):
        expand_sweeps({"weight": {SWEEP_KEY: {"from": 0, "to": float("inf")}}})
