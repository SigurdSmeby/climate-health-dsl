"""Tests for the ``series:`` schema.

A scenario is one list of series. A series is a count series iff it carries a
``counts:`` block; any series may depend on any other, forming a DAG that
``series_order`` topologically sorts and that cycles are rejected in.

These tests cover the shape itself — parsing, validation and ordering — not
what the engine does with it.
"""
import pytest
from pydantic import ValidationError

from dsl.core.config.schema import ScenarioConfig, parse_config
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict

# ------------------------------------------------------------- the series list


def test_series_list_parses():
    config = parse_config(make_config_dict())
    assert isinstance(config, ScenarioConfig)
    assert [s.name for s in config.series] == [
        "rainfall",
        "mean_temperature",
        "disease_cases",
    ]


def test_variables_block_is_rejected():
    """``variables:`` is not an accepted spelling of ``series:``."""
    data = make_config_dict()
    data["variables"] = data.pop("series")
    with pytest.raises(ValidationError, match="variables"):
        parse_config(data)


def test_top_level_disease_cases_block_is_rejected():
    data = make_config_dict()
    data["disease_cases"] = {"population": 100, "depends_on": []}
    with pytest.raises(ValidationError, match="disease_cases"):
        parse_config(data)


def test_empty_series_list_rejected():
    with pytest.raises(ValidationError, match="series"):
        parse_config(make_config_dict(series=[]))


def test_duplicate_series_names_rejected():
    data = make_config_dict(
        series=[series_dict("rain"), series_dict("rain")],
    )
    with pytest.raises(ValidationError, match="duplicate"):
        parse_config(data)


@pytest.mark.parametrize("reserved", ["time_period", "location", "population"])
def test_series_named_like_reserved_column_rejected(reserved):
    data = make_config_dict(series=[series_dict(reserved)])
    with pytest.raises(ValidationError, match=reserved):
        parse_config(data)


def test_disease_cases_is_a_legal_series_name():
    """It is an ordinary name — the conventional one for a count series."""
    config = parse_config(make_config_dict())
    assert config.series[-1].name == "disease_cases"


# ------------------------------------------------------------- generate/depends


def test_generate_is_optional_when_depends_on_is_set():
    """A dependency-only series gets a flat 0 base — no `generate:` needed."""
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict("dam", depends_on=[{"series": "rain", "lag": 1}]),
        ]
    )
    config = parse_config(data)
    assert config.series[1].generate is None
    assert config.series[1].depends_on[0].series == "rain"


def test_series_with_neither_generate_nor_depends_on_rejected():
    data = make_config_dict(series=[{"name": "orphan"}])
    with pytest.raises(ValidationError, match="generate"):
        parse_config(data)


def test_depends_on_requires_a_series_key():
    data = make_config_dict(
        series=[
            series_dict("rain"),
            series_dict("dam", depends_on=[{"variable": "rain"}]),
        ]
    )
    with pytest.raises(ValidationError, match="variable"):
        parse_config(data)


def test_dangling_dependency_lists_known_names():
    data = make_config_dict(
        series=[
            series_dict("rain"),
            series_dict("dam", depends_on=[{"series": "nope"}]),
        ]
    )
    with pytest.raises(ValidationError, match="nope"):
        parse_config(data)


def test_series_may_have_several_parents():
    data = make_config_dict(
        series=[
            series_dict("rain"),
            series_dict("temp"),
            series_dict(
                "wind",
                depends_on=[
                    {"series": "rain", "lag": 1, "weight": 0.6},
                    {"series": "temp", "lag": 2, "weight": -0.3},
                ],
            ),
        ]
    )
    config = parse_config(data)
    assert [d.series for d in config.series[2].depends_on] == ["rain", "temp"]


# --------------------------------------------------------------- the DAG order


def test_topological_order_is_exposed():
    """Declaration order is YAML's; the engine needs dependency order."""
    data = make_config_dict(
        series=[
            series_dict("dam", depends_on=[{"series": "rain", "lag": 1}]),
            series_dict("rain"),
        ]
    )
    config = parse_config(data)
    assert config.series_order() == ["rain", "dam"]


def test_chain_of_three_sorts():
    data = make_config_dict(
        series=[
            series_dict("rain", depends_on=[{"series": "temp", "lag": 1}]),
            series_dict("temp", depends_on=[{"series": "wind", "lag": 1}]),
            series_dict("wind"),
        ]
    )
    assert parse_config(data).series_order() == ["wind", "temp", "rain"]


def test_confounder_shape_sorts():
    """wind drives both rain and temp — a DAG, not a cycle."""
    data = make_config_dict(
        series=[
            series_dict("wind"),
            series_dict("rain", depends_on=[{"series": "wind", "lag": 1}]),
            series_dict(
                "temp",
                depends_on=[
                    {"series": "rain", "lag": 1},
                    {"series": "wind", "lag": 1},
                ],
            ),
        ]
    )
    order = parse_config(data).series_order()
    assert order.index("wind") < order.index("rain") < order.index("temp")


def test_two_series_cycle_rejected():
    data = make_config_dict(
        series=[
            series_dict("rain", depends_on=[{"series": "temp", "lag": 1}]),
            series_dict("temp", depends_on=[{"series": "rain", "lag": 1}]),
        ]
    )
    with pytest.raises(ValidationError, match="cycle"):
        parse_config(data)


def test_three_series_cycle_rejected():
    """The whole graph is walked, so a longer loop is caught too."""
    data = make_config_dict(
        series=[
            series_dict("rain", depends_on=[{"series": "temp", "lag": 1}]),
            series_dict("temp", depends_on=[{"series": "wind", "lag": 1}]),
            series_dict("wind", depends_on=[{"series": "rain", "lag": 1}]),
        ]
    )
    with pytest.raises(ValidationError, match="cycle"):
        parse_config(data)


def test_cycle_error_names_the_members():
    data = make_config_dict(
        series=[
            series_dict("rain", depends_on=[{"series": "temp", "lag": 1}]),
            series_dict("temp", depends_on=[{"series": "rain", "lag": 1}]),
        ]
    )
    with pytest.raises(ValidationError) as excinfo:
        parse_config(data)
    message = str(excinfo.value)
    assert "rain" in message and "temp" in message


def test_self_dependency_rejected():
    data = make_config_dict(
        series=[series_dict("rain", depends_on=[{"series": "rain", "lag": 1}])]
    )
    with pytest.raises(ValidationError, match="cycle"):
        parse_config(data)


# ------------------------------------------------------------ the counts block


def test_counts_block_marks_a_count_series():
    config = parse_config(make_config_dict())
    rainfall, disease = config.series[0], config.series[2]
    assert rainfall.counts is None
    assert disease.counts is not None
    assert config.population_for("loc") == 100_000


def test_counts_defaults_are_applied():
    counts = parse_config(make_config_dict()).series[2].counts
    assert counts.max_rate == 0.3
    assert counts.median_rate == 0.1
    assert counts.distribution == "poisson"
    assert counts.overdispersion == 10.0


def test_count_fields_rejected_outside_the_block():
    """`extra="forbid"` does the work — no cross-field rule needed."""
    data = make_config_dict()
    data["series"][0]["max_rate"] = 0.3
    with pytest.raises(ValidationError, match="max_rate"):
        parse_config(data)


def test_count_distribution_key_rejected():
    data = make_config_dict()
    data["series"][2]["counts"]["count_distribution"] = "poisson"
    with pytest.raises(ValidationError, match="count_distribution"):
        parse_config(data)


def test_negative_binomial_accepted():
    data = make_config_dict()
    data["series"][2]["counts"].update(
        distribution="negative_binomial", overdispersion=3.0
    )
    counts = parse_config(data).series[2].counts
    assert counts.distribution == "negative_binomial"
    assert counts.overdispersion == 3.0


def test_unknown_distribution_rejected():
    data = make_config_dict()
    data["series"][2]["counts"]["distribution"] = "gaussian"
    with pytest.raises(ValidationError, match="distribution"):
        parse_config(data)


def test_median_rate_must_be_below_max_rate():
    data = make_config_dict()
    data["series"][2]["counts"].update(median_rate=0.4, max_rate=0.3)
    with pytest.raises(ValidationError, match="median_rate"):
        parse_config(data)


def test_several_count_series_allowed():
    """Multi-disease falls out of the unified list for free."""
    data = make_config_dict(
        series=[
            series_dict("rain"),
            series_dict(
                "dengue",
                counts={},
                depends_on=[{"series": "rain", "lag": 1}],
            ),
            series_dict(
                "malaria",
                counts={},
                depends_on=[{"series": "rain", "lag": 2}],
            ),
        ]
    )
    config = parse_config(data)
    assert [s.name for s in config.series if s.counts] == ["dengue", "malaria"]


def test_a_count_series_may_drive_another_series():
    """Co-infection: a disease is a series like any other."""
    data = make_config_dict(
        series=[
            series_dict("rain"),
            series_dict(
                "dengue",
                counts={},
                depends_on=[{"series": "rain", "lag": 1}],
            ),
            series_dict(
                "malaria",
                counts={},
                depends_on=[{"series": "dengue", "lag": 1}],
            ),
        ]
    )
    order = parse_config(data).series_order()
    assert order.index("dengue") < order.index("malaria")


# --------------------------------------------- fields that moved to any series


def test_missing_rate_allowed_on_a_plain_series():
    data = make_config_dict()
    data["series"][0]["missing_rate"] = 0.02
    assert parse_config(data).series[0].missing_rate == 0.02


def test_missing_rate_defaults_to_zero():
    assert parse_config(make_config_dict()).series[0].missing_rate == 0.0


def test_missing_rate_out_of_range_rejected():
    data = make_config_dict()
    data["series"][0]["missing_rate"] = 2.0
    with pytest.raises(ValidationError, match="missing_rate"):
        parse_config(data)


def test_autoregressive_is_an_ar1_block():
    data = make_config_dict()
    data["series"][0]["autoregressive"] = {"phi": 0.8, "noise": 0.3}
    series = parse_config(data).series[0]
    assert series.autoregressive.phi == 0.8
    assert series.autoregressive.noise == 0.3


def test_autoregressive_defaults_to_none():
    assert parse_config(make_config_dict()).series[0].autoregressive is None


def test_autoregressive_boolean_form_rejected():
    """A bare ``true`` would leave the persistence magnitude unstated."""
    data = make_config_dict()
    data["series"][0]["autoregressive"] = True
    with pytest.raises(ValidationError, match="autoregressive"):
        parse_config(data)


@pytest.mark.parametrize("phi", [1.0, 1.5, -0.1])
def test_autoregressive_phi_must_be_in_unit_range(phi):
    """phi == 1 is a random walk: non-stationary, so rejected."""
    data = make_config_dict()
    data["series"][0]["autoregressive"] = {"phi": phi}
    with pytest.raises(ValidationError, match="phi"):
        parse_config(data)


def test_autoregressive_noise_must_be_non_negative():
    data = make_config_dict()
    data["series"][0]["autoregressive"] = {"phi": 0.5, "noise": -1.0}
    with pytest.raises(ValidationError, match="noise"):
        parse_config(data)


# ------------------------------------------------------------------ warm-up


def test_lag_at_least_n_total_rejected():
    data = make_config_dict(n_total=5)
    data["series"][2]["depends_on"][0]["lag"] = 5
    with pytest.raises(ValidationError, match="lag"):
        parse_config(data)


def test_negative_lag_rejected():
    data = make_config_dict()
    data["series"][2]["depends_on"][0]["lag"] = -1
    with pytest.raises(ValidationError, match="lag"):
        parse_config(data)


def test_transform_warmup_counts_toward_n_total():
    data = make_config_dict(n_total=6)
    data["series"][2]["depends_on"][0].update(
        lag=4,
        transforms=[
            {"name": "distributed_lag", "params": {"weights": [0.5, 0.3, 0.2]}}
        ],
    )
    with pytest.raises(ValidationError, match="warm-up"):
        parse_config(data)


# ------------------------------------------------------------- population


def test_population_lives_on_the_location():
    config = parse_config(
        make_config_dict(
            locations={"north": {"population": 300_000}},
            series=[
                series_dict("rain"),
                series_dict(
                    "cases",
                    counts={},
                    depends_on=[{"series": "rain", "lag": 1}],
                ),
            ],
        )
    )
    assert config.population_for("north") == 300_000


def test_population_on_a_counts_block_is_rejected():
    """One place only: population is a property of a place, not a disease."""
    data = make_config_dict()
    data["series"][2]["counts"]["population"] = 100_000
    with pytest.raises(ValidationError, match="population"):
        parse_config(data)


def test_locations_are_required_when_a_series_counts():
    """A count series needs a population, which only a location can give."""
    data = make_config_dict(
        series=[
            series_dict("rain"),
            series_dict(
                "cases", counts={}, depends_on=[{"series": "rain", "lag": 1}]
            ),
        ]
    )
    data.pop("locations", None)
    with pytest.raises(ValidationError, match="locations"):
        parse_config(data)


def test_a_scenario_without_counts_needs_no_population():
    config = parse_config(
        make_config_dict(
            locations={"north": {"population": 100_000}}, series=[series_dict("rain")]
        )
    )
    assert config.locations == ["north"]


def test_every_location_must_declare_a_population():
    data = make_config_dict(
        locations={"north": {"population": 300_000}, "south": {}},
        series=[
            series_dict("rain"),
            series_dict(
                "cases", counts={}, depends_on=[{"series": "rain", "lag": 1}]
            ),
        ],
    )
    with pytest.raises(ValidationError, match="south"):
        parse_config(data)


def test_two_diseases_share_one_location_population():
    """The point of the move: they cannot disagree about the same place."""
    config = parse_config(
        make_config_dict(
            locations={"north": {"population": 300_000}},
            series=[
                series_dict("rain"),
                series_dict(
                    "dengue",
                    counts={},
                    depends_on=[{"series": "rain", "lag": 1}],
                ),
                series_dict(
                    "malaria",
                    counts={},
                    depends_on=[{"series": "rain", "lag": 2}],
                ),
            ],
        )
    )
    assert config.population_for("north") == 300_000


def test_location_population_may_be_a_generator():
    config = parse_config(
        make_config_dict(
            locations={
                "north": {
                    "population": {
                        "generate": "linear_trend",
                        "params": {"start": 70_000, "slope": 90},
                    }
                }
            },
            series=[
                series_dict("rain"),
                series_dict(
                    "cases",
                    counts={},
                    depends_on=[{"series": "rain", "lag": 1}],
                ),
            ],
        )
    )
    assert config.population_for("north").generate == "linear_trend"
