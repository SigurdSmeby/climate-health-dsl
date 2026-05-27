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
