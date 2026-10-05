"""Tests for the fold report.

The report answers the question a cross-validation run raises: is each fold
actually testing what it was meant to? It counts the features the scenario
planted — seasonal peaks, outbreak shocks, declared events — on each side of
every fold, and flags folds whose test half contains none of a kind.

The counts are ground truth read from the generators, not spikes detected in
the output, so "three outbreaks in this fold" is exact.
"""
import json
import re

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
from dsl.core.pipeline.folds import build_report, render_markdown, write_report
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict


def _config(split=None, **overrides):
    data = make_config_dict(**overrides)
    if split is not None:
        data["split"] = split
    return parse_config(data)


def _locations(*names):
    return {name: {"population": 100_000} for name in names}


def _spiky(split=None, **overrides):
    """A scenario whose generators plant countable features."""
    return _config(
        split=split or {"kind": "time", "k": 4},
        n_total=120,
        series=[
            series_dict(
                "rainfall",
                generate="seasonal_spike",
                params={"spike_center": 6},
            ),
            series_dict(
                "heatwaves",
                generate="outbreak_shocks",
                params={"rate": 2.0, "magnitude": 8},
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
        **overrides,
    )


# ------------------------------------------------------------- the structure


def test_no_split_means_no_report():
    assert build_report(_config(), run(_config())) is None


def test_report_has_one_entry_per_fold():
    config = _spiky()
    report = build_report(config, run(config))
    assert len(report["folds"]) == 4


def test_report_records_the_split_it_describes():
    config = _spiky(split={"kind": "time", "k": 3, "scheme": "blocked"})
    report = build_report(config, run(config))
    assert report["split"]["kind"] == "time"
    assert report["split"]["scheme"] == "blocked"
    assert report["split"]["k"] == 3


# -------------------------------------------------------- size and coverage


def test_each_fold_reports_both_sides():
    config = _spiky()
    fold = build_report(config, run(config))["folds"][0]
    assert set(fold) >= {"index", "train", "test"}


def test_row_counts_match_the_written_folds():
    config = _spiky()
    df = run(config)
    report = build_report(config, df)
    for fold, entry in zip(config.folds(), report["folds"]):
        assert entry["train"]["rows"] == len(fold.train_periods) * len(
            fold.train_locations
        )


def test_coverage_names_the_first_and_last_period():
    config = _spiky()
    fold = build_report(config, run(config))["folds"][0]
    assert fold["train"]["first_period"] < fold["train"]["last_period"]
    assert fold["train"]["last_period"] < fold["test"]["first_period"]


def test_location_counts_are_reported():
    config = _spiky(locations=_locations("north", "south"))
    fold = build_report(config, run(config))["folds"][0]
    assert fold["train"]["locations"] == 2


def test_a_location_split_reports_the_held_out_place():
    config = _spiky(
        split={"kind": "location"}, locations=_locations("north", "south", "east")
    )
    fold = build_report(config, run(config))["folds"][0]
    assert fold["test"]["location_names"] == ["north"]


# ----------------------------------------------------------- missing values


def test_missing_counts_are_reported_per_side():
    config = _spiky()
    fold = build_report(config, run(config))["folds"][0]
    assert "missing" in fold["train"]


def test_the_warm_up_shows_up_as_missing():
    """A lag blanks the opening periods, which the report must not hide."""
    config = _config(
        split={"kind": "time", "k": 2},
        n_total=60,
        series=[
            series_dict("rainfall", generate="seasonal_spike"),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 5}],
            ),
        ],
    )
    fold = build_report(config, run(config))["folds"][0]
    assert fold["train"]["missing"]["disease_cases"] == 5


def test_a_clean_fold_reports_no_missing():
    config = _config(
        split={"kind": "time", "k": 2},
        n_total=60,
        series=[
            series_dict("rainfall", generate="seasonal_spike"),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 0}],
            ),
        ],
    )
    fold = build_report(config, run(config))["folds"][1]
    assert fold["test"]["missing"]["disease_cases"] == 0


# ------------------------------------------------- the point: event counting


def test_events_are_counted_on_both_sides_of_each_fold():
    config = _spiky()
    fold = build_report(config, run(config))["folds"][0]
    assert fold["train"]["events"]
    assert isinstance(fold["train"]["events"], dict)


def test_seasonal_peaks_are_counted_exactly():
    """One peak per year, so a fold's count follows from its period span."""
    config = _config(
        split={"kind": "time", "k": 1},
        n_total=120,
        period="monthly",
        series=[
            series_dict(
                "rainfall", generate="seasonal_spike", params={"spike_center": 6}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    fold = build_report(config, run(config))["folds"][0]
    total = (
        fold["train"]["events"]["rainfall"]["seasonal_spike"]["count"]
        + fold["test"]["events"]["rainfall"]["seasonal_spike"]["count"]
    )
    assert total == 10  # 120 monthly periods is ten years


def test_every_planted_event_lands_in_exactly_one_side():
    """Nothing is double-counted or dropped between train and test."""
    config = _spiky(split={"kind": "time", "k": 1})
    report = build_report(config, run(config))
    fold = report["folds"][0]
    counted = sum(
        entry["count"]
        for side in ("train", "test")
        for series in fold[side]["events"].values()
        for entry in series.values()
    )
    assert counted == report["total_events"]


def test_declared_events_are_counted_too():
    config = _config(
        split={"kind": "time", "k": 1},
        n_total=60,
        events={"storm": {"at": [5, 40]}},
        series=[
            series_dict(
                "rainfall", generate="flat", events={"storm": {"multiplier": 2.0}}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    fold = build_report(config, run(config))["folds"][0]
    assert fold["train"]["events"]["rainfall"]["storm"]["count"] == 1
    assert fold["test"]["events"]["rainfall"]["storm"]["count"] == 1


def test_a_featureless_scenario_counts_nothing():
    config = _config(
        split={"kind": "time", "k": 2},
        n_total=60,
        series=[
            series_dict("flatline", generate="flat"),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "flatline", "lag": 1}],
            ),
        ],
    )
    assert build_report(config, run(config))["total_events"] == 0


# ------------------------------------------------------------ skew warnings


def test_a_fold_with_no_test_events_is_flagged():
    """The whole point: a fold that cannot measure what you planted."""
    config = _config(
        split={"kind": "time", "k": 2},
        n_total=60,
        events={"storm": {"at": [3, 8]}},  # both land in the opening periods
        series=[
            series_dict(
                "rainfall", generate="flat", events={"storm": {"multiplier": 2.0}}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    report = build_report(config, run(config))
    assert any("storm" in w for w in report["warnings"])


def test_an_even_spread_is_not_flagged():
    config = _config(
        split={"kind": "time", "k": 1},
        n_total=60,
        events={"storm": {"at": [5, 10, 40, 50]}},
        series=[
            series_dict(
                "rainfall", generate="flat", events={"storm": {"multiplier": 2.0}}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    report = build_report(config, run(config))
    assert not any("storm" in w for w in report["warnings"])


def test_warnings_name_the_fold_and_the_kind():
    config = _config(
        split={"kind": "time", "k": 2},
        n_total=60,
        events={"storm": {"at": [3, 8]}},
        series=[
            series_dict(
                "rainfall", generate="flat", events={"storm": {"multiplier": 2.0}}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    warning = next(
        w for w in build_report(config, run(config))["warnings"] if "storm" in w
    )
    assert "fold" in warning.lower()


# ------------------------------------------------------------- written files


def test_both_files_are_written(tmp_path):
    config = _spiky()
    write_report(config, run(config), tmp_path)
    assert (tmp_path / "folds" / "report.md").is_file()
    assert (tmp_path / "folds" / "report.json").is_file()


def test_no_files_without_a_split(tmp_path):
    config = _config()
    write_report(config, run(config), tmp_path)
    assert not (tmp_path / "folds" / "report.md").exists()


def test_the_json_is_the_report(tmp_path):
    config = _spiky()
    write_report(config, run(config), tmp_path)
    loaded = json.loads((tmp_path / "folds" / "report.json").read_text())
    assert len(loaded["folds"]) == 4


def test_the_markdown_has_a_csv_table(tmp_path):
    config = _spiky()
    write_report(config, run(config), tmp_path)
    text = (tmp_path / "folds" / "report.md").read_text()
    assert "| fold | split | rows |" in text


def test_the_markdown_lists_events_per_fold(tmp_path):
    config = _spiky()
    write_report(config, run(config), tmp_path)
    text = (tmp_path / "folds" / "report.md").read_text()
    assert "seasonal_spike" in text
    assert "outbreak" in text


def test_the_markdown_names_the_split(tmp_path):
    config = _spiky()
    write_report(config, run(config), tmp_path)
    assert "expanding" in (tmp_path / "folds" / "report.md").read_text()


def test_a_report_with_no_warnings_says_so(tmp_path):
    """A long enough span puts a seasonal peak on both sides of the split."""
    config = _config(
        split={"kind": "time", "k": 1},
        n_total=240,  # twenty years: ten peaks each side
        period="monthly",
        series=[
            series_dict("rainfall", generate="seasonal_spike"),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    write_report(config, run(config), tmp_path)
    text = (tmp_path / "folds" / "report.md").read_text()
    assert "None — every fold" in text


# ------------------------------------------------- where the features sit


def test_events_report_their_periods_not_just_a_count():
    """A count says a fold has a storm; it does not say which month, so you
    cannot line it up against the data."""
    config = _config(
        split={"kind": "time", "k": 2},
        period="monthly",
        n_total=60,
        start_period="2010-01",
        events={"storm": {"at": [5, 40]}},
        series=[
            series_dict(
                "rainfall", generate="flat", events={"storm": {"multiplier": 2.0}}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    fold = build_report(config, run(config))["folds"][0]
    storms = fold["train"]["events"]["rainfall"]["storm"]
    # Period 5 with a 2010-01 start is 2010-06.
    assert storms["periods"] == ["2010-06"]
    assert storms["count"] == 1


def test_reported_periods_use_the_scenario_calendar():
    """The labels are the scenario's own, so they line up with the CSV."""
    config = _spiky(
        split={"kind": "time", "k": 1}, period="monthly", start_period="2015-03"
    )
    fold = build_report(config, run(config))["folds"][0]
    peaks = fold["train"]["events"]["rainfall"]["seasonal_spike"]["periods"]
    assert peaks, "a ten-year monthly span should hold several peaks"
    assert all(re.fullmatch(r"20[12]\d-\d\d", p) for p in peaks), peaks


def test_markdown_renders_tables_not_code_blocks():
    config = _spiky()
    text = render_markdown(build_report(config, run(config)))
    assert "| fold | split |" in text
    assert "```" not in text


def test_markdown_lists_the_periods_a_feature_sits_at():
    config = _config(
        split={"kind": "time", "k": 1},
        period="monthly",
        n_total=60,
        start_period="2010-01",
        events={"storm": {"at": [5]}},
        series=[
            series_dict(
                "rainfall", generate="flat", events={"storm": {"multiplier": 2.0}}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    text = render_markdown(build_report(config, run(config)))
    assert "2010-06" in text


def test_the_period_list_matches_the_count():
    """Two shocks can start in the same period. Collapsing the duplicate
    would contradict the count beside it — and the count is the guarantee
    that overlapping shocks are still two."""
    config = _spiky(split={"kind": "time", "k": 3})
    for fold in build_report(config, run(config))["folds"]:
        for side in ("train", "test"):
            for kinds in fold[side]["events"].values():
                for entry in kinds.values():
                    assert len(entry["periods"]) == entry["count"]


def test_per_location_features_are_reported_per_location():
    """A generator draws per location, so a yearly peak in two places is two
    features. Reporting them under one row reads as two peaks in one place —
    and the periods then repeat with no explanation."""
    config = _spiky(
        split={"kind": "time", "k": 2},
        locations={"north": {"population": 1000}, "south": {"population": 1000}},
    )
    fold = build_report(config, run(config))["folds"][0]
    entry = fold["train"]["events"]["rainfall"]["seasonal_spike"]
    # Same periods in both places, so each period appears once per location.
    assert entry["locations"] == ["north", "south"]
    assert len(set(entry["periods"])) * 2 == entry["count"]


def test_a_regional_event_is_not_attributed_to_one_location():
    """An event is drawn once for the whole region, so it has no location."""
    config = _config(
        split={"kind": "time", "k": 1},
        period="monthly",
        n_total=60,
        locations={"north": {"population": 1000}, "south": {"population": 1000}},
        events={"storm": {"at": [5]}},
        series=[
            series_dict(
                "rainfall", generate="flat", events={"storm": {"multiplier": 2.0}}
            ),
            series_dict(
                "disease_cases",
                counts={},
                depends_on=[{"series": "rainfall", "lag": 1}],
            ),
        ],
    )
    fold = build_report(config, run(config))["folds"][0]
    entry = fold["train"]["events"]["rainfall"]["storm"]
    assert entry["count"] == 1  # once, not once per location
    assert entry["locations"] == []
