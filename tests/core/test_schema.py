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

from tests.conftest import scenario_dict as make_config_dict


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


@pytest.mark.parametrize(
    "period,start,n",
    [
        ("daily", "99991231", 2),
        ("monthly", "9999-12", 2),
        ("weekly", "9999-W52", 2),
        ("yearly", "9999", 2),
    ],
)
def test_period_range_past_year_9999_rejected(period, start, n):
    # Bug #36: a range that crosses past year 9999 crashes or emits 5-digit
    # labels; reject it at the schema.
    data = make_config_dict(period=period, n_total=n, start_period=start)
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "lag": 0}]
    with pytest.raises(ValidationError, match="9999|range|year"):
        parse_config(data)


def test_seasonal_phase_warns_with_start_period():
    # Bug #16 (documented limitation): a mid-year start_period only relabels;
    # seasonal phase still begins at the cycle start. Warn so it's not a
    # silent surprise.
    data = make_config_dict(period="monthly", start_period="2010-07")
    warnings = validate_scenario(parse_config(data))
    assert any("seasonal" in w.lower() and "start_period" in w for w in warnings)


def test_negative_seed_rejected():
    # Bug #30: numpy needs a non-negative seed; reject it at the schema.
    with pytest.raises(ValidationError, match="seed"):
        parse_config(make_config_dict(seed=-1))


def test_zero_seed_ok():
    assert parse_config(make_config_dict(seed=0)).seed == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weight_rejected(bad):
    # Bug #17: NaN/Inf in a float config field must be rejected.
    data = make_config_dict()
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "weight": bad}]
    with pytest.raises(ValidationError):
        parse_config(data)


def test_empty_variable_name_rejected():
    # Bug #20: blank/whitespace variable names create unnamed columns.
    data = make_config_dict()
    data["variables"] = [{"name": "  ", "generate": "seasonal_spike"}]
    data["disease_cases"]["depends_on"] = [{"variable": "  ", "lag": 1}]
    with pytest.raises(ValidationError, match="name|empty|blank"):
        parse_config(data)


def test_empty_location_name_rejected():
    data = make_config_dict()
    data["locations"] = [""]
    with pytest.raises(ValidationError, match="location|empty|blank"):
        parse_config(data)


def test_train_fraction_yielding_empty_train_rejected():
    # Bug #22: floor(n_total * fraction) == 0 means an empty train split.
    with pytest.raises(ValidationError, match="train"):
        parse_config(make_config_dict(n_total=2, train_fraction=0.1))


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


@pytest.mark.parametrize(
    "reserved", ["time_period", "location", "disease_cases", "population"]
)
def test_variable_named_like_reserved_column_rejected(reserved):
    # Bug #1: a variable whose name collides with a built-in output column
    # would silently overwrite it. Must be a hard error.
    data = make_config_dict()
    data["variables"] = [{"name": reserved, "generate": "seasonal_spike"}]
    data["disease_cases"]["depends_on"] = [{"variable": reserved, "lag": 1}]
    with pytest.raises(ValidationError, match=reserved):
        parse_config(data)


def test_duplicate_variable_names_rejected():
    # Bug #4: two variables with the same name silently collide (one is lost).
    data = make_config_dict()
    data["variables"] = [
        {"name": "rainfall", "generate": "seasonal_spike"},
        {"name": "rainfall", "generate": "seasonal_smooth"},
    ]
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "lag": 1}]
    with pytest.raises(ValidationError, match="duplicate"):
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


def test_from_csv_fixed_source_with_multi_location_warns():
    # Bug #3: a from_csv variable pinned to one source_location, but the
    # scenario has several locations, means every location gets the SAME real
    # series. Warn (it's legal but usually a surprise).
    data = make_config_dict()
    data["locations"] = ["oslo", "bergen"]
    data["variables"] = [
        {
            "name": "rainfall",
            "generate": "from_csv",
            "params": {"file": "x.csv", "column": "rainfall",
                       "source_location": "A"},
        }
    ]
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "lag": 1}]
    warnings = validate_scenario(parse_config(data))
    assert any("from_csv" in w and "rainfall" in w for w in warnings)


def test_from_csv_multi_location_without_source_no_warning():
    # With per-location auto-match, a from_csv variable with NO source_location
    # and several locations is fine (each location reads its own rows), so
    # there is no duplication warning — the warning is only for a FIXED source.
    data = make_config_dict()
    data["locations"] = ["oslo", "bergen"]
    data["variables"] = [
        {"name": "rainfall", "generate": "from_csv",
         "params": {"file": "x.csv", "column": "rainfall"}}  # no source_location
    ]
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "lag": 1}]
    warnings = validate_scenario(parse_config(data))
    assert not any("from_csv" in w for w in warnings)


def test_lag_consuming_whole_train_split_warns():
    # Bug #34: if max lag >= the training periods, every train target is
    # warm-up NaN — warn that training has no observed cases.
    data = make_config_dict(n_total=10, train_fraction=0.8)
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "lag": 8}]
    warnings = validate_scenario(parse_config(data))
    assert any("train" in w and "lag" in w for w in warnings)


def test_from_csv_single_location_no_warning():
    # The same from_csv variable with a single location is fine — no warning.
    data = make_config_dict()
    data["locations"] = ["oslo"]
    data["variables"] = [
        {
            "name": "rainfall",
            "generate": "from_csv",
            "params": {"file": "x.csv", "column": "rainfall",
                       "source_location": "A"},
        }
    ]
    data["disease_cases"]["depends_on"] = [{"variable": "rainfall", "lag": 1}]
    warnings = validate_scenario(parse_config(data))
    assert not any("from_csv" in w for w in warnings)


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
