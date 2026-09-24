"""Tests for generating a scenario's series in dependency order.

The engine walks the series DAG parents-first, so a series can be built from
other series as well as from a generator. The same lag/transform/standardize/
weight pipeline feeds both a plain float series and a count series; only the
final step differs.
"""
import numpy as np
import pytest

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict

# ------------------------------------------------------------ frame assembly


def test_every_series_becomes_a_column():
    df = run(parse_config(make_config_dict()))
    for name in ("rainfall", "mean_temperature", "disease_cases"):
        assert name in df.columns


def test_columns_follow_declaration_order_not_dependency_order():
    """Output order is the author's; only generation order is the graph's."""
    data = make_config_dict(
        series=[
            series_dict("dam", depends_on=[{"series": "rain", "lag": 1}]),
            series_dict("rain", generate="seasonal_spike"),
        ]
    )
    df = run(parse_config(data))
    assert list(df.columns)[:4] == ["time_period", "location", "dam", "rain"]


def test_row_count_is_n_total_times_locations():
    data = make_config_dict(n_total=24, locations=["north", "south"])
    df = run(parse_config(data))
    assert len(df) == 48


def test_population_column_comes_from_the_count_series():
    df = run(parse_config(make_config_dict()))
    assert (df["population"] == 100_000).all()


def test_scenario_without_a_count_series_has_no_disease_column():
    data = make_config_dict(series=[series_dict("rain", generate="flat")])
    df = run(parse_config(data))
    assert "disease_cases" not in df.columns
    assert "rain" in df.columns


# ------------------------------------------------------- dependency ordering


def test_dependent_series_is_built_from_its_parent():
    """A dependency-only series is not flat: it carries its parent's shape."""
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict("dam", depends_on=[{"series": "rain", "lag": 1}]),
        ]
    )
    df = run(parse_config(data))
    dam = df["dam"].to_numpy()[1:]  # skip the lag warm-up
    assert np.std(dam) > 0


def test_declaration_order_does_not_change_the_result():
    """The graph decides generation order, so YAML order is cosmetic."""
    parent = series_dict("rain", generate="seasonal_spike")
    child = series_dict("dam", depends_on=[{"series": "rain", "lag": 1}])
    forward = run(parse_config(make_config_dict(series=[parent, child])))
    reverse = run(parse_config(make_config_dict(series=[child, parent])))
    np.testing.assert_array_equal(
        forward["dam"].to_numpy(), reverse["dam"].to_numpy()
    )


def test_chain_of_three_propagates():
    data = make_config_dict(
        series=[
            series_dict("wind", generate="seasonal_spike"),
            series_dict("rain", depends_on=[{"series": "wind", "lag": 1}]),
            series_dict("dam", depends_on=[{"series": "rain", "lag": 1}]),
        ]
    )
    df = run(parse_config(data))
    assert np.std(df["dam"].to_numpy()[2:]) > 0


def test_several_parents_both_contribute():
    """Dropping one parent changes the child, so neither is ignored."""
    def build(deps):
        return run(
            parse_config(
                make_config_dict(
                    series=[
                        series_dict("rain", generate="seasonal_spike"),
                        series_dict("temp", generate="linear_trend"),
                        series_dict("wind", depends_on=deps),
                    ]
                )
            )
        )["wind"].to_numpy()

    both = build(
        [
            {"series": "rain", "lag": 1, "weight": 1.0},
            {"series": "temp", "lag": 1, "weight": 1.0},
        ]
    )
    rain_only = build([{"series": "rain", "lag": 1, "weight": 1.0}])
    assert not np.allclose(both[1:], rain_only[1:])


def test_dependency_weight_scales_the_contribution():
    def build(weight):
        data = make_config_dict(
            series=[
                series_dict("rain", generate="seasonal_spike"),
                series_dict(
                    "dam",
                    depends_on=[{"series": "rain", "lag": 1, "weight": weight}],
                ),
            ]
        )
        return run(parse_config(data))["dam"].to_numpy()[1:]

    assert np.std(build(2.0)) > np.std(build(1.0))


def test_zero_weight_dependency_contributes_nothing():
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict(
                "dam", depends_on=[{"series": "rain", "lag": 1, "weight": 0.0}]
            ),
        ]
    )
    dam = run(parse_config(data))["dam"].to_numpy()
    assert np.allclose(dam, 0.0)


def test_dependency_lag_blanks_the_warm_up():
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict("dam", depends_on=[{"series": "rain", "lag": 3}]),
        ]
    )
    dam = run(parse_config(data))["dam"].to_numpy()
    assert np.isnan(dam[:3]).all()
    assert not np.isnan(dam[3:]).any()


def test_dependency_transforms_are_applied():
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict(
                "dam",
                depends_on=[
                    {
                        "series": "rain",
                        "lag": 1,
                        "transforms": [
                            {
                                "name": "distributed_lag",
                                "params": {"weights": [0.5, 0.3, 0.2]},
                            }
                        ],
                    }
                ],
            ),
        ]
    )
    dam = run(parse_config(data))["dam"].to_numpy()
    # lag 1 plus two extra warm-up periods from the three-weight kernel.
    assert np.isnan(dam[:3]).all()


def test_a_count_series_can_drive_another_series():
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict(
                "dengue",
                counts={"population": 100_000},
                depends_on=[{"series": "rain", "lag": 1}],
            ),
            series_dict(
                "malaria",
                counts={"population": 100_000},
                depends_on=[{"series": "dengue", "lag": 1}],
            ),
        ]
    )
    df = run(parse_config(data))
    assert df["malaria"].notna().any()


def test_generate_and_depends_on_combine():
    """A series may have both a base generator and parents."""
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict(
                "dam",
                generate="linear_trend",
                params={"start": 100.0, "slope": 1.0},
                depends_on=[{"series": "rain", "lag": 1, "weight": 1.0}],
            ),
        ]
    )
    dam = run(parse_config(data))["dam"].to_numpy()[1:]
    assert dam.mean() > 50  # the trend base survives


# ------------------------------------------------------------- count series


def test_count_series_are_whole_numbers():
    df = run(parse_config(make_config_dict()))
    observed = df["disease_cases"].dropna().to_numpy()
    assert np.all(observed == np.floor(observed))


def test_count_series_respects_max_rate():
    data = make_config_dict()
    data["series"][2]["counts"].update(max_rate=0.02, median_rate=0.001)
    df = run(parse_config(data))
    assert df["disease_cases"].dropna().max() <= 100_000 * 0.02


def test_several_count_series_are_independent():
    data = make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict(
                "dengue",
                counts={"population": 100_000},
                depends_on=[{"series": "rain", "lag": 1}],
            ),
            series_dict(
                "malaria",
                counts={"population": 100_000},
                depends_on=[{"series": "rain", "lag": 4}],
            ),
        ]
    )
    df = run(parse_config(data))
    assert not np.array_equal(
        df["dengue"].to_numpy(), df["malaria"].to_numpy()
    )


def test_no_built_in_seasonality_without_a_seasonal_driver():
    """A count series' seasonality comes from its drivers, nothing else."""
    data = make_config_dict(
        series=[
            series_dict("flatline", generate="flat", params={"level": 1.0}),
            series_dict(
                "cases",
                counts={"population": 100_000},
                depends_on=[{"series": "flatline", "lag": 0, "weight": 1.0}],
            ),
        ],
        n_total=104,
    )
    cases = run(parse_config(data))["cases"].dropna().to_numpy()
    # With a constant driver the rate is constant, so counts are Poisson
    # noise around one level — a seasonal term would show as a yearly swing.
    first_half = cases[: len(cases) // 2].mean()
    second_half = cases[len(cases) // 2 :].mean()
    assert abs(first_half - second_half) < 0.2 * cases.mean()


# ------------------------------------------------------- per-series features


def test_missing_rate_blanks_a_plain_series():
    data = make_config_dict()
    data["series"][0]["missing_rate"] = 0.5
    df = run(parse_config(data))
    assert df["rainfall"].isna().any()


def test_missing_rate_defaults_to_no_gaps():
    df = run(parse_config(make_config_dict()))
    assert df["rainfall"].notna().all()


def test_autoregressive_adds_persistence():
    """AR(1) output correlates with its own past; white noise does not."""
    def lag1_corr(phi):
        data = make_config_dict(
            series=[
                series_dict(
                    "store",
                    generate="flat",
                    params={"level": 0.0, "noise": 1.0},
                    **({"autoregressive": {"phi": phi, "noise": 1.0}} if phi else {}),
                )
            ],
            n_total=400,
        )
        x = run(parse_config(data))["store"].to_numpy()
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    assert lag1_corr(0.9) > lag1_corr(None) + 0.3


def test_autoregressive_stays_bounded():
    """AR(1) is stationary: its spread does not grow with series length."""
    def spread(n_total):
        data = make_config_dict(
            series=[
                series_dict(
                    "store",
                    generate="flat",
                    params={"level": 0.0, "noise": 0.0},
                    autoregressive={"phi": 0.8, "noise": 0.2},
                )
            ],
            n_total=n_total,
        )
        return np.std(run(parse_config(data))["store"].to_numpy())

    # A random walk's sd grows as sqrt(n), so 16x the periods would roughly
    # quadruple it. A stationary process settles instead.
    assert spread(4000) < 2 * spread(250)


# ---------------------------------------------------------------- determinism


def test_same_seed_gives_identical_output():
    config = parse_config(make_config_dict(seed=7))
    np.testing.assert_array_equal(
        run(config)["rainfall"].to_numpy(), run(config)["rainfall"].to_numpy()
    )


def test_different_seed_changes_output():
    first = run(parse_config(make_config_dict(seed=1)))
    second = run(parse_config(make_config_dict(seed=2)))
    assert not np.array_equal(
        first["disease_cases"].to_numpy(), second["disease_cases"].to_numpy()
    )


def test_locations_draw_independently():
    data = make_config_dict(locations=["north", "south"], n_total=52)
    df = run(parse_config(data))
    north = df[df["location"] == "north"]["rainfall"].to_numpy()
    south = df[df["location"] == "south"]["rainfall"].to_numpy()
    assert not np.array_equal(north, south)


def test_adding_a_decoy_series_does_not_shift_the_others():
    """Each series has its own RNG stream, keyed by name."""
    base = run(parse_config(make_config_dict()))
    data = make_config_dict()
    data["series"].insert(1, series_dict("decoy", generate="flat"))
    with_decoy = run(parse_config(data))
    np.testing.assert_array_equal(
        base["rainfall"].to_numpy(), with_decoy["rainfall"].to_numpy()
    )


# --------------------------------------------------------------- populations


def test_per_location_population_is_used():
    data = make_config_dict(
        locations={"north": {"population": 300_000}, "south": {"population": 180_000}}
    )
    df = run(parse_config(data))
    assert set(df[df["location"] == "north"]["population"]) == {300_000}
    assert set(df[df["location"] == "south"]["population"]) == {180_000}


def test_population_generator_varies_over_time():
    data = make_config_dict()
    data["series"][2]["counts"]["population"] = {
        "generate": "linear_trend",
        "params": {"start": 70_000, "slope": 90},
    }
    df = run(parse_config(data))
    assert df["population"].nunique() > 1


def test_unknown_generator_names_the_series():
    data = make_config_dict(series=[series_dict("rain", generate="nope")])
    with pytest.raises(KeyError, match="nope"):
        run(parse_config(data))
