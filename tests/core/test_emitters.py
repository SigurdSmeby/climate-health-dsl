"""Tests for emitters: how a series' float signal becomes its output column.

Everything up to the final step is shared — generator, parents, events,
persistence, gaps — so an emitter is the small piece that differs. Adding an
output type is then one self-registering file, the same way a generator or a
transform is added, with nothing in the core to edit.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from dsl.core.config.schema import parse_config
from dsl.core.extension.emitter_base import (
    Emitter,
    emitter_registry,
    get_emitter,
    register_emitter,
)
from dsl.core.pipeline.engine import run
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict

# ------------------------------------------------------------- the registry


def test_counts_is_registered():
    assert "counts" in emitter_registry.names()


def test_unknown_emitter_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="counts"):
        get_emitter("no_such_emitter")


def test_registering_a_duplicate_name_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @register_emitter("counts")
        class _Duplicate(Emitter):
            def emit(self, signal, rng):
                return signal


def test_an_emitter_must_implement_emit():
    with pytest.raises(TypeError):
        Emitter()


# ------------------------------------------------------- the counts emitter


def test_counts_emitter_draws_whole_numbers():
    emitter = get_emitter("counts")(population=np.full(50, 100_000))
    drawn = emitter.emit(np.zeros(50), np.random.default_rng(0))
    assert np.all(drawn == np.floor(drawn))


def test_max_rate_caps_the_expected_rate_not_each_draw():
    """max_rate bounds the rate the counts are drawn AGAINST. Individual
    Poisson draws still scatter around it; only the population is a hard
    cap on a drawn value."""
    emitter = get_emitter("counts")(
        population=np.full(4000, 1000), max_rate=0.02, median_rate=0.001
    )
    # A huge signal saturates the sigmoid, so the rate sits at the ceiling.
    drawn = emitter.emit(np.full(4000, 50.0), np.random.default_rng(0))
    assert drawn.mean() == pytest.approx(1000 * 0.02, rel=0.05)


def test_population_is_a_hard_cap_on_every_draw():
    emitter = get_emitter("counts")(
        population=np.full(200, 5), max_rate=1.0, median_rate=0.9
    )
    drawn = emitter.emit(np.full(200, 50.0), np.random.default_rng(0))
    assert drawn.max() <= 5


def test_counts_emitter_sits_at_the_median_for_a_typical_period():
    """A signal of 0 is a typical period, so the rate lands on median_rate."""
    emitter = get_emitter("counts")(
        population=np.full(4000, 100_000), median_rate=0.1, max_rate=0.3
    )
    drawn = emitter.emit(np.zeros(4000), np.random.default_rng(0))
    assert drawn.mean() == pytest.approx(10_000, rel=0.02)


def test_counts_emitter_preserves_missing_input():
    """A period with no usable signal must not get a fabricated count."""
    signal = np.zeros(10)
    signal[:3] = np.nan
    emitter = get_emitter("counts")(population=np.full(10, 1000))
    drawn = emitter.emit(signal, np.random.default_rng(0))
    assert np.isnan(drawn[:3]).all()
    assert not np.isnan(drawn[3:]).any()


def test_negative_binomial_is_more_variable_than_poisson():
    def spread(**kw):
        emitter = get_emitter("counts")(population=np.full(3000, 100_000), **kw)
        return np.std(emitter.emit(np.zeros(3000), np.random.default_rng(0)))

    assert spread(distribution="negative_binomial", overdispersion=2.0) > spread(
        distribution="poisson"
    )


def test_counts_emitter_is_reproducible():
    def drawn():
        emitter = get_emitter("counts")(population=np.full(20, 1000))
        return emitter.emit(np.zeros(20), np.random.default_rng(3))

    np.testing.assert_array_equal(drawn(), drawn())


# ------------------------------------------------ reached through a scenario


def test_a_counts_block_selects_the_counts_emitter():
    df = run(parse_config(make_config_dict()))
    observed = df["disease_cases"].dropna().to_numpy()
    assert np.all(observed == np.floor(observed))


def test_a_series_without_a_block_is_written_as_floats():
    df = run(parse_config(make_config_dict()))
    rainfall = df["rainfall"].to_numpy()
    assert not np.all(rainfall == np.floor(rainfall))


def test_an_unknown_block_name_is_rejected_by_the_schema():
    """A misspelled block is a typo, not a new output type."""
    data = make_config_dict()
    data["series"][2]["cunts"] = data["series"][2].pop("counts")
    with pytest.raises(ValidationError, match="cunts"):
        parse_config(data)


# ------------------------------------------------- adding a new output type


def test_a_new_emitter_needs_no_core_change():
    """The point of the registry: one self-registering class is enough."""

    @register_emitter("_test_double")
    class _DoubleEmitter(Emitter):
        def __init__(self, factor: float = 2.0):
            self.factor = factor

        def emit(self, signal, rng):
            return signal * self.factor

    try:
        emitter = get_emitter("_test_double")(factor=3.0)
        np.testing.assert_array_equal(
            emitter.emit(np.ones(4), np.random.default_rng(0)),
            np.full(4, 3.0),
        )
    finally:
        emitter_registry._items.pop("_test_double", None)


def test_count_parameters_are_validated_by_the_block_not_the_engine():
    """Each emitter owns its own params, so the schema does not branch."""
    data = make_config_dict()
    data["series"][2]["counts"]["overdispersion"] = -1.0
    with pytest.raises(ValidationError, match="overdispersion"):
        parse_config(data)


def test_series_without_an_emitter_still_gets_its_features():
    """Events, persistence and gaps apply regardless of the output type."""
    data = make_config_dict(
        series=[
            series_dict(
                "rain",
                generate="flat",
                params={"level": 10.0, "noise": 0.0},
                missing_rate=0.5,
            )
        ]
    )
    assert run(parse_config(data))["rain"].isna().any()
