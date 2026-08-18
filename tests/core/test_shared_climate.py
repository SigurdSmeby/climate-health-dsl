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


def test_shared_location_name_rejected():
    # "shared" is the internal RNG key for the shared-driver stream; a
    # location literally named "shared" would collide with it.
    with pytest.raises(ValueError, match="shared"):
        _cfg(locations=("shared", "other"))


def _multi_location_csv(tmp_path):
    from tests.conftest import write_csv

    periods = [f"2010-{m + 1:02d}" for m in range(12)]
    return write_csv(
        tmp_path / "multi.csv", periods * 2,
        rainfall=list(range(12)) + list(range(100, 112)),
        location=["north"] * 12 + ["south"] * 12,
    )


def _from_csv_shared_config(csv_path, locations, source_location=None):
    params = {"file": str(csv_path), "column": "rainfall"}
    if source_location is not None:
        params["source_location"] = source_location
    var = {"name": "rainfall", "generate": "from_csv", "params": params, "shared": 1.0}
    return ScenarioConfig(
        period="monthly", n_total=12, seed=0, locations=list(locations),
        variables=[var],
        disease_cases={"population": 100000, "median_rate": 0.1, "max_rate": 0.3,
                       "depends_on": [{"variable": "rainfall", "weight": 1.0}]},
    )


def test_from_csv_shared_without_source_location_is_ambiguous(tmp_path):
    # Regression: no source_location + a multi-location CSV means there's no
    # well-defined location-independent series to share — must raise, not
    # silently reuse each location's own auto-matched rows as "shared".
    csv_path = _multi_location_csv(tmp_path)
    config = _from_csv_shared_config(csv_path, ["north", "south"])
    with pytest.raises(ValueError, match="source_location"):
        run(config)


def test_from_csv_shared_with_source_location_is_location_independent(tmp_path):
    csv_path = _multi_location_csv(tmp_path)
    config = _from_csv_shared_config(csv_path, ["north", "south"], source_location="north")
    r = _rainfall_by_location(run(config))
    # shared=1.0 → every location gets the SAME series (north's data), not
    # each location's own auto-matched rows.
    assert np.allclose(r["north"], r["south"])
    assert np.allclose(r["north"], np.arange(12, dtype=float))
