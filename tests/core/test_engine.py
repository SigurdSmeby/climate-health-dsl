"""Integration tests for the engine: validated config → tidy DataFrame."""
import numpy as np
import pandas as pd
import pytest

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run


@pytest.fixture
def example_config():
    """The §7 example scenario (as a dict, like load_yaml would return it)."""
    return parse_config(
        {
            "period": "weekly",
            "n_total": 78,
            "seed": 42,
            "train_fraction": 0.8,
            "variables": [
                {"name": "rainfall", "generate": "seasonal_spike"},
                {"name": "mean_temperature", "generate": "seasonal_smooth"},
            ],
            "disease_cases": {
                "population": 100_000,
                "depends_on": [
                    {"variable": "rainfall", "lag": 3, "weight": 2.0},
                    {"variable": "mean_temperature", "lag": 3, "weight": 1.0},
                ],
                "autoregressive": False,
                "missing_rate": 0.05,
            },
        }
    )


def test_chap_columns_in_order(example_config):
    df = run(example_config)
    assert list(df.columns) == [
        "time_period",
        "location",
        "rainfall",
        "mean_temperature",
        "disease_cases",
        "population",
    ]


def test_row_count_is_n_total(example_config):
    df = run(example_config)
    assert len(df) == 78


def test_time_period_labels(example_config):
    df = run(example_config)
    assert df["time_period"].iloc[0] == "2000-W01"
    assert df["time_period"].iloc[52] == "2001-W01"  # year rollover


def test_population_is_constant(example_config):
    df = run(example_config)
    assert (df["population"] == 100_000).all()


def test_warmup_rows_blanked(example_config):
    df = run(example_config)
    # max lag in the example is 3, so the first 3 disease values are NaN.
    assert df["disease_cases"].iloc[:3].isna().all()


def test_rerun_is_bit_identical(example_config):
    a = run(example_config)
    b = run(example_config)
    # equals() treats NaN == NaN, which plain == does not.
    assert a.equals(b)


def test_different_seed_differs(example_config):
    a = run(example_config)
    changed = example_config.model_copy(update={"seed": 7})
    b = run(changed)
    assert not a.equals(b)


def test_column_names_come_from_yaml_not_hardcoded():
    # The engine must not hardcode CHAP names: a scenario with other variable
    # names gets columns named after them.
    config = parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "variables": [{"name": "wind", "generate": "seasonal_smooth"}],
            "disease_cases": {
                "population": 5_000,
                "depends_on": [{"variable": "wind", "lag": 1}],
            },
        }
    )
    df = run(config)
    assert list(df.columns) == [
        "time_period",
        "location",
        "wind",
        "disease_cases",
        "population",
    ]


def test_generator_params_are_passed_through():
    # params: from the YAML must reach the generator (noise: 0 makes the
    # series deterministic, so two different seeds give the same rainfall).
    def config_with_seed(seed):
        return parse_config(
            {
                "period": "weekly",
                "n_total": 52,
                "seed": seed,
                "variables": [
                    {
                        "name": "rainfall",
                        "generate": "seasonal_spike",
                        "params": {"noise": 0, "spike_center": 10},
                    }
                ],
                "disease_cases": {
                    "population": 1_000,
                    "depends_on": [{"variable": "rainfall"}],
                },
            }
        )

    a = run(config_with_seed(1))
    b = run(config_with_seed(2))
    assert np.array_equal(a["rainfall"], b["rainfall"])
    assert int(np.argmax(a["rainfall"])) == 10


def test_unknown_generator_name_raises_with_available():
    config = parse_config(
        {
            "period": "weekly",
            "n_total": 52,
            "variables": [{"name": "rainfall", "generate": "no_such_shape"}],
            "disease_cases": {
                "population": 1_000,
                "depends_on": [{"variable": "rainfall"}],
            },
        }
    )
    with pytest.raises(KeyError, match="seasonal_spike"):
        run(config)


def test_returns_dataframe(example_config):
    assert isinstance(run(example_config), pd.DataFrame)


def start_period_config(start_period, period="monthly"):
    return parse_config(
        {
            "period": period,
            "n_total": 24,
            "start_period": start_period,
            "variables": [{"name": "rainfall", "generate": "seasonal_spike"}],
            "disease_cases": {
                "population": 1_000,
                "depends_on": [{"variable": "rainfall"}],
            },
        }
    )


def test_start_period_sets_time_period_labels():
    df = run(start_period_config("2010-01"))
    assert df["time_period"].iloc[0] == "2010-01"
    assert df["time_period"].iloc[12] == "2011-01"


def test_start_period_can_begin_mid_year():
    df = run(start_period_config("2010-07"))
    assert df["time_period"].iloc[0] == "2010-07"
    assert df["time_period"].iloc[5] == "2010-12"
    assert df["time_period"].iloc[6] == "2011-01"  # rolls over correctly


def test_start_period_weekly():
    df = run(start_period_config("2015-W50", period="weekly"))
    assert df["time_period"].iloc[0] == "2015-W50"
    assert df["time_period"].iloc[2] == "2015-W52"
    assert df["time_period"].iloc[3] == "2016-W01"  # rolls over correctly


def test_start_period_daily_and_yearly():
    df = run(start_period_config("20100615", period="daily"))
    assert df["time_period"].iloc[0] == "20100615"
    assert df["time_period"].iloc[1] == "20100616"
    df = run(start_period_config("2003", period="yearly"))
    assert df["time_period"].iloc[0] == "2003"
    assert df["time_period"].iloc[23] == "2026"


def test_location_column_present_and_constant(example_config):
    # Default scenario: one location named "loc", right after time_period.
    df = run(example_config)
    assert list(df.columns)[:2] == ["time_period", "location"]
    assert (df["location"] == "loc").all()


def multi_location_config():
    return parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "seed": 1,
            "locations": ["oslo", "bergen"],
            "variables": [{"name": "rainfall", "generate": "seasonal_spike"}],
            "disease_cases": {
                "population": 10_000,
                "depends_on": [{"variable": "rainfall", "lag": 2}],
            },
        }
    )


def test_multi_location_row_count_and_stacking():
    df = run(multi_location_config())
    # Long format: n_total rows per location, location-major order.
    assert len(df) == 48
    assert list(df["location"].unique()) == ["oslo", "bergen"]
    oslo = df[df["location"] == "oslo"]
    assert len(oslo) == 24
    assert oslo["time_period"].iloc[0] == "2000-01"  # each location starts at t0


def test_locations_get_different_draws():
    # Same generative process per location, but independent randomness:
    # the locations must not be copies of each other.
    df = run(multi_location_config())
    oslo = df[df["location"] == "oslo"]["rainfall"].to_numpy()
    bergen = df[df["location"] == "bergen"]["rainfall"].to_numpy()
    assert not np.array_equal(oslo, bergen)


def test_per_location_population_in_output():
    config = parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "seed": 1,
            "locations": {
                "oslo": {"population": 700000},
                "bergen": {"population": 280000},
            },
            "variables": [{"name": "rainfall", "generate": "seasonal_spike"}],
            "disease_cases": {
                "population": 10000,  # the fallback; overridden per location
                "depends_on": [{"variable": "rainfall", "lag": 2}],
            },
        }
    )
    df = run(config)
    oslo_pop = df[df["location"] == "oslo"]["population"].unique()
    bergen_pop = df[df["location"] == "bergen"]["population"].unique()
    assert list(oslo_pop) == [700000]
    assert list(bergen_pop) == [280000]


def test_per_location_population_caps_disease_counts():
    # A tiny-population location must have its disease_cases capped lower
    # than a large-population one — proof the population actually drives the
    # incidence model per location, not just the output column.
    config = parse_config(
        {
            "period": "monthly",
            "n_total": 36,
            "seed": 3,
            "locations": {"big": {"population": 100000}, "small": {"population": 50}},
            "variables": [{"name": "rainfall", "generate": "seasonal_spike"}],
            "disease_cases": {
                "population": 100000,
                "depends_on": [{"variable": "rainfall", "lag": 1, "weight": 3.0}],
            },
        }
    )
    df = run(config)
    big_max = df[df["location"] == "big"]["disease_cases"].max()
    small_max = df[df["location"] == "small"]["disease_cases"].max()
    assert small_max <= 50
    assert big_max > 50


def population_config(population, locations=None, seed=1):
    data = {
        "period": "monthly",
        "n_total": 36,
        "seed": seed,
        "variables": [{"name": "rainfall", "generate": "seasonal_spike"}],
        "disease_cases": {
            "population": population,
            "depends_on": [{"variable": "rainfall", "lag": 1, "weight": 2.0}],
        },
    }
    if locations is not None:
        data["locations"] = locations
    return parse_config(data)


def test_constant_population_unchanged_by_generator_machinery():
    # A scalar population must stay byte-identical to before the feature: the
    # population path must consume no RNG, so disease draws are unshifted.
    a = run(population_config(10000))
    b = run(population_config(10000))
    assert a.equals(b)
    assert (a["population"] == 10000).all()


def test_linear_trend_population_rises_in_output():
    config = population_config(
        {"generate": "linear_trend", "params": {"start": 1000, "slope": 50}}
    )
    df = run(config)
    pop = df["population"].to_numpy()
    assert pop[0] == 1000
    assert pop[-1] == 1000 + 50 * 35  # start + slope * (n_total - 1)
    assert np.all(np.diff(pop) > 0)  # monotonically rising
    # Population is integer counts of people.
    assert pop.dtype.kind in "iu"


def test_growing_population_scales_disease_up():
    # With a fast-growing population the disease counts late in the series
    # should tend higher than early (more people → more cases at same rate).
    config = population_config(
        {"generate": "linear_trend", "params": {"start": 1000, "slope": 300}}
    )
    df = run(config)
    cases = df["disease_cases"].dropna().to_numpy()
    early = cases[: len(cases) // 3].mean()
    late = cases[-len(cases) // 3 :].mean()
    assert late > early


def test_per_location_population_generators():
    config = population_config(
        population=10000,  # fallback, unused here
        locations={
            "slow": {"population": {"generate": "linear_trend",
                                    "params": {"start": 1000, "slope": 10}}},
            "fast": {"population": {"generate": "linear_trend",
                                    "params": {"start": 1000, "slope": 500}}},
        },
    )
    df = run(config)
    slow_end = df[df["location"] == "slow"]["population"].iloc[-1]
    fast_end = df[df["location"] == "fast"]["population"].iloc[-1]
    assert fast_end > slow_end  # different trajectories per location


def test_generated_population_reproducible():
    config_spec = {"generate": "linear_trend", "params": {"start": 1000, "slope": 50}}
    a = run(population_config(config_spec, seed=4))
    b = run(population_config(config_spec, seed=4))
    assert a.equals(b)


def test_start_period_aligns_from_csv_data(tmp_path):
    # Bug #2: with a scenario start_period and a from_csv variable, the output
    # labels must match the real values read from the CSV — not label row 0
    # as a later date. CSV: 2010-01..2010-06 with rainfall 1..6.
    csv = tmp_path / "real.csv"
    pd.DataFrame(
        {
            "time_period": [f"2010-{m:02d}" for m in range(1, 7)],
            "rainfall": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(csv, index=False)
    config = parse_config(
        {
            "period": "monthly",
            "n_total": 3,
            "start_period": "2010-04",
            "variables": [
                {
                    "name": "rainfall",
                    "generate": "from_csv",
                    "params": {"file": str(csv), "column": "rainfall"},
                }
            ],
            "disease_cases": {
                "population": 1000,
                "depends_on": [{"variable": "rainfall", "lag": 1}],
            },
        }
    )
    df = run(config)
    # The row labelled 2010-04 must carry the CSV's April value (4.0), not 1.0.
    assert df["time_period"].tolist() == ["2010-04", "2010-05", "2010-06"]
    assert df["rainfall"].tolist() == [4.0, 5.0, 6.0]


def test_multi_location_is_reproducible():
    a = run(multi_location_config())
    b = run(multi_location_config())
    assert a.equals(b)
