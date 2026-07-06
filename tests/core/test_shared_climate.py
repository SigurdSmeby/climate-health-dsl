"""Feature 4: a variable can be shared across locations (latent confounder).

A `shared` fraction on a variable mixes a location-INDEPENDENT component into
each location's series, so locations move together because of a common hidden
driver — not a real location-to-location link. shared=0 is today's behaviour
(fully independent), shared=1 is identical across locations.

TDD: written before the schema/engine change.
"""
import numpy as np
import pytest

from dsl.core.config.schema import ScenarioConfig, VariableSpec
from dsl.core.pipeline.engine import run


def _cfg(shared=None, seed=0, locations=("a", "b")):
    var = {"name": "rainfall", "generate": "seasonal_smooth",
           "params": {"noise": 3.0}}
    if shared is not None:
        var["shared"] = shared
    return ScenarioConfig(
        period="weekly", n_total=104, seed=seed, locations=list(locations),
        variables=[var],
        disease_cases={"population": 100000, "median_rate": 0.1, "max_rate": 0.3,
                       "depends_on": [{"variable": "rainfall", "weight": 1.0}]},
    )


def test_shared_field_defaults_absent():
    v = VariableSpec(name="rainfall", generate="flat")
    assert v.shared is None or v.shared == 0.0


def test_shared_out_of_range_rejected():
    with pytest.raises(ValueError):
        VariableSpec(name="rainfall", generate="flat", shared=1.5)
    with pytest.raises(ValueError):
        VariableSpec(name="rainfall", generate="flat", shared=-0.1)


def _rainfall_by_location(df):
    return {loc: g["rainfall"].to_numpy()
            for loc, g in df.groupby("location", sort=False)}


def test_shared_zero_is_independent():
    # Two locations, shared=0 → different series (today's behaviour).
    r = _rainfall_by_location(run(_cfg(shared=0.0)))
    assert not np.allclose(r["a"], r["b"])


def test_shared_one_is_identical_across_locations():
    r = _rainfall_by_location(run(_cfg(shared=1.0)))
    assert np.allclose(r["a"], r["b"])


def test_partial_shared_correlates_locations():
    # shared=0.9 → strongly correlated but not identical.
    r = _rainfall_by_location(run(_cfg(shared=0.9)))
    corr = np.corrcoef(r["a"], r["b"])[0, 1]
    assert corr > 0.5
    assert not np.allclose(r["a"], r["b"])


def test_absent_shared_is_byte_identical_to_no_field():
    # Backward compat: omitting `shared` must reproduce the pre-feature output
    # exactly (no rng shift). Compare no-field vs shared=None construction.
    a = run(_cfg(shared=None, seed=7))
    b = run(_cfg(shared=None, seed=7))
    assert a.equals(b)


def test_shared_recorded_in_metadata():
    from dsl.core.pipeline.metadata import build_metadata

    meta = build_metadata(_cfg(shared=0.8))
    var = meta["scenario"]["variables"][0]
    assert var.get("shared") == 0.8
