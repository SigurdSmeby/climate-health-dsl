"""Tests for the scenario schema: hard errors (parse_config) and warnings
(validate_scenario).

The schema is two-tier: impossible/certainly-wrong scenarios must RAISE with a
message naming the offending field; suspicious-but-legal scenarios must only
produce warning strings and still parse.
"""
import pytest
from pydantic import ValidationError

from dsl.core.config.schema import (
    ScenarioConfig,
    parse_config,
    validate_scenario,
)


def make_config_dict(**overrides) -> dict:
    """A minimal valid scenario dict; tests tweak it via keyword overrides."""
    data = {
        "period": "weekly",
        "n_total": 78,
        "seed": 42,
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
        },
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------- hard errors


def test_valid_config_parses():
    config = parse_config(make_config_dict())
    assert isinstance(config, ScenarioConfig)
    assert config.period == "weekly"
    assert config.n_total == 78
    assert config.variables[0].name == "rainfall"
    assert config.disease_cases.depends_on[0].lag == 3


def test_defaults_are_applied():
    config = parse_config(make_config_dict())
    assert config.train_fraction is None  # optional: omit → single CSV only
    assert config.disease_cases.autoregressive is False
    assert config.disease_cases.missing_rate == 0.0
    assert config.disease_cases.max_rate == 0.3
    assert config.disease_cases.median_rate == 0.1


def test_missing_required_field_names_it():
    data = make_config_dict()
    del data["n_total"]
    with pytest.raises(ValidationError, match="n_total"):
        parse_config(data)


def test_train_fraction_out_of_range_rejected():
    with pytest.raises(ValidationError, match="train_fraction"):
        parse_config(make_config_dict(train_fraction=1.5))


def test_missing_rate_out_of_range_rejected():
    data = make_config_dict()
    data["disease_cases"]["missing_rate"] = 2.0
    with pytest.raises(ValidationError, match="missing_rate"):
        parse_config(data)


def test_unsupported_period_rejected():
    with pytest.raises(ValidationError, match="period"):
        parse_config(make_config_dict(period="fortnightly"))


def test_typoed_field_rejected():
    # extra="forbid" must catch a misspelled key instead of silently ignoring it.
    data = make_config_dict()
    data["peroid"] = "weekly"
    with pytest.raises(ValidationError, match="peroid"):
        parse_config(data)


def test_dangling_depends_on_reference_raises_and_lists_valid_names():
    data = make_config_dict()
    data["disease_cases"]["depends_on"] = [{"variable": "rainfal", "lag": 3}]
    with pytest.raises(ValidationError) as excinfo:
        parse_config(data)
    message = str(excinfo.value)
    assert "rainfal" in message
    # The error must list the valid variable names to help fix the typo.
    assert "rainfall" in message
    assert "mean_temperature" in message


def test_count_distribution_defaults_to_poisson():
    config = parse_config(make_config_dict())
    assert config.disease_cases.count_distribution == "poisson"


def test_negative_binomial_accepted():
    data = make_config_dict()
    data["disease_cases"]["count_distribution"] = "negative_binomial"
    data["disease_cases"]["overdispersion"] = 5.0
    config = parse_config(data)
    assert config.disease_cases.count_distribution == "negative_binomial"
    assert config.disease_cases.overdispersion == 5.0


def test_unknown_count_distribution_rejected():
    data = make_config_dict()
    data["disease_cases"]["count_distribution"] = "gamma"
    with pytest.raises(ValidationError, match="count_distribution"):
        parse_config(data)


def test_overdispersion_must_be_positive():
    data = make_config_dict()
    data["disease_cases"]["overdispersion"] = 0
    with pytest.raises(ValidationError, match="overdispersion"):
        parse_config(data)


def test_median_rate_must_be_below_max_rate():
    # The sigmoid shift is logit(median_rate / max_rate), undefined at >= 1.
    data = make_config_dict()
    data["disease_cases"]["max_rate"] = 0.1
    data["disease_cases"]["median_rate"] = 0.1
    with pytest.raises(ValidationError, match="median_rate"):
        parse_config(data)


def test_lag_at_least_n_total_raises():
    data = make_config_dict(n_total=10)
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "lag": 10}]
    with pytest.raises(ValidationError, match="lag"):
        parse_config(data)


def test_start_period_default_is_none():
    assert parse_config(make_config_dict()).start_period is None


def test_start_period_configurable():
    config = parse_config(make_config_dict(period="monthly", start_period="2010-07"))
    assert config.start_period == "2010-07"


def test_start_period_must_match_resolution():
    # A monthly label on a weekly scenario is a hard error.
    with pytest.raises(ValidationError, match="start_period"):
        parse_config(make_config_dict(period="weekly", start_period="2010-07"))


def test_locations_mapping_form_sets_per_location_population():
    data = make_config_dict()
    data["locations"] = {
        "oslo": {"population": 700_000},
        "bergen": {"population": 280_000},
    }
    config = parse_config(data)
    # Names still come out as a plain list (engine/order unchanged).
    assert config.locations == ["oslo", "bergen"]
    assert config.population_for("oslo") == 700_000
    assert config.population_for("bergen") == 280_000


def test_list_form_falls_back_to_disease_population():
    data = make_config_dict()  # list form, disease_cases.population = 100_000
    config = parse_config(data)
    assert config.population_for("rainfall_location_does_not_matter") == 100_000


def test_mapping_form_without_population_uses_disease_default():
    # An empty override block means "use the top-level population".
    data = make_config_dict()
    data["locations"] = {"oslo": {}, "bergen": {"population": 5}}
    config = parse_config(data)
    assert config.population_for("oslo") == 100_000  # the disease default
    assert config.population_for("bergen") == 5


def test_disease_population_optional_when_all_locations_override():
    # No top-level population is fine as long as every location sets its own.
    data = make_config_dict()
    del data["disease_cases"]["population"]
    data["locations"] = {
        "oslo": {"population": 700_000},
        "bergen": {"population": 280_000},
    }
    config = parse_config(data)
    assert config.population_for("oslo") == 700_000
    assert config.population_for("bergen") == 280_000


def test_missing_population_with_fallback_location_raises():
    # If any location relies on the (now missing) fallback, it's an error.
    data = make_config_dict()
    del data["disease_cases"]["population"]
    data["locations"] = {"oslo": {"population": 700_000}, "bergen": {}}  # bergen falls back
    with pytest.raises(ValidationError, match="population"):
        parse_config(data)


def test_missing_population_list_form_raises():
    # The list form always falls back, so a missing population is an error.
    data = make_config_dict()
    del data["disease_cases"]["population"]  # list form locations default
    with pytest.raises(ValidationError, match="population"):
        parse_config(data)


def test_population_generator_form_parses():
    data = make_config_dict()
    data["disease_cases"]["population"] = {
        "generate": "linear_trend",
        "params": {"start": 1000, "slope": 10},
    }
    config = parse_config(data)
    source = config.population_for("anywhere")
    # The generator form resolves to a PopulationSpec, not a bare int.
    assert source.generate == "linear_trend"
    assert source.params == {"start": 1000, "slope": 10}


def test_population_generator_per_location():
    data = make_config_dict()
    data["locations"] = {
        "oslo": {"population": {"generate": "linear_trend", "params": {"start": 700}}},
        "bergen": {"population": 280},  # int form still allowed alongside
    }
    config = parse_config(data)
    assert config.population_for("oslo").generate == "linear_trend"
    assert config.population_for("bergen") == 280


def test_population_generator_requires_generate_key():
    data = make_config_dict()
    data["disease_cases"]["population"] = {"params": {"start": 1000}}  # no generate
    with pytest.raises(ValidationError, match="generate"):
        parse_config(data)


def test_population_int_form_still_works():
    # Backward compatibility: a plain int is unchanged.
    config = parse_config(make_config_dict())
    assert config.population_for("anywhere") == 100_000


def test_mapping_form_rejects_unknown_override_key():
    data = make_config_dict()
    data["locations"] = {"oslo": {"populaton": 5}}  # typo
    with pytest.raises(ValidationError, match="populaton"):
        parse_config(data)


def test_mapping_form_population_must_be_positive():
    data = make_config_dict()
    data["locations"] = {"oslo": {"population": 0}}
    with pytest.raises(ValidationError, match="population"):
        parse_config(data)


def test_empty_mapping_rejected():
    data = make_config_dict()
    data["locations"] = {}
    with pytest.raises(ValidationError, match="locations"):
        parse_config(data)


def test_locations_default_is_single_loc():
    config = parse_config(make_config_dict())
    assert config.locations == ["loc"]


def test_locations_accepts_multiple_names():
    config = parse_config(make_config_dict(locations=["oslo", "bergen"]))
    assert config.locations == ["oslo", "bergen"]


def test_empty_locations_rejected():
    with pytest.raises(ValidationError, match="locations"):
        parse_config(make_config_dict(locations=[]))


def test_duplicate_locations_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        parse_config(make_config_dict(locations=["oslo", "oslo"]))


# ------------------------------------------------------------------ warnings


def test_orphan_variable_warns_but_parses():
    data = make_config_dict()
    data["variables"].append({"name": "wind", "generate": "seasonal_smooth"})
    config = parse_config(data)  # must NOT raise — decoys are legal
    warnings = validate_scenario(config)
    assert any("wind" in w for w in warnings)


def test_clean_config_has_no_warnings():
    warnings = validate_scenario(parse_config(make_config_dict()))
    assert warnings == []


def test_high_missing_rate_warns():
    data = make_config_dict()
    data["disease_cases"]["missing_rate"] = 0.5
    warnings = validate_scenario(parse_config(data))
    assert any("missing_rate" in w for w in warnings)


def test_high_train_fraction_warns():
    warnings = validate_scenario(parse_config(make_config_dict(train_fraction=0.95)))
    assert any("train_fraction" in w for w in warnings)


def test_n_total_below_one_seasonal_cycle_warns():
    # 20 weekly periods < 52 (one year): legal, but seasonality won't show.
    data = make_config_dict(n_total=20)
    warnings = validate_scenario(parse_config(data))
    assert any("n_total" in w for w in warnings)
