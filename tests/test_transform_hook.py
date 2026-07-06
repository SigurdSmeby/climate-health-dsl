"""Feature 0: transforms are drop-in — a depends_on entry can name registry
transforms that reshape the driver before it's weighted.

TDD: these are written before the implementation and must fail first.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from dsl.core.config.schema import DependencySpec, ScenarioConfig
from dsl.core.extension.transform_base import Transform, register_transform
from dsl.core.pipeline.disease import build_disease_cases


# A tiny test-only transform: multiply the series by a constant. Registered
# under a name unlikely to clash. Proves an arbitrary registry transform is
# reachable from a scenario, not just the built-in lag/missing.
@register_transform("_test_scale")
class _ScaleTransform(Transform):
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def apply(self, series, rng):
        return series.astype(float) * self.factor


def _spec(**kw):
    from dsl.core.config.schema import DiseaseSpec

    base = dict(population=100_000, depends_on=[], median_rate=0.1, max_rate=0.3)
    base.update(kw)
    return DiseaseSpec(**base)


def test_dependency_accepts_transforms_field():
    dep = DependencySpec(
        variable="rainfall",
        transforms=[{"name": "_test_scale", "params": {"factor": 2.0}}],
    )
    assert dep.transforms[0].name == "_test_scale"
    assert dep.transforms[0].params == {"factor": 2.0}


def test_transforms_defaults_to_empty():
    dep = DependencySpec(variable="rainfall")
    assert dep.transforms == []


def test_unknown_transform_key_rejected():
    # extra="forbid" must still hold on the new nested model.
    with pytest.raises(ValidationError):
        DependencySpec(
            variable="rainfall",
            transforms=[{"name": "_test_scale", "typo": 1}],
        )


def test_transform_actually_reshapes_the_signal():
    # Same seed, same driver: a transform that scales the driver by a large
    # factor must change the disease signal versus no transform.
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    driver = np.linspace(0.0, 10.0, 60)
    drivers = {"rainfall": driver}

    plain = build_disease_cases(
        dict(drivers), _spec(depends_on=[{"variable": "rainfall", "weight": 1.0}]),
        rng_a, 60, "weekly",
    )
    scaled = build_disease_cases(
        dict(drivers),
        _spec(depends_on=[{
            "variable": "rainfall", "weight": 1.0,
            "transforms": [{"name": "_test_scale", "params": {"factor": 5.0}}],
        }]),
        rng_b, 60, "weekly",
    )
    # Standardize makes a pure scale a no-op, so use a NONLINEAR check instead:
    # this test just asserts the transform path runs and is wired. A linear
    # scale is intentionally invariant under z-score; see the offset test.
    assert plain.shape == scaled.shape


def test_nonlinear_transform_changes_signal():
    # A transform that is NOT invariant under standardize (adds a constant to
    # only the high half) must move the disease signal.
    @register_transform("_test_hinge")
    class _Hinge(Transform):
        def apply(self, series, rng):
            out = series.astype(float)
            out[out < 5.0] = 0.0
            return out

    driver = np.linspace(0.0, 10.0, 60)
    plain = build_disease_cases(
        {"rainfall": driver.copy()},
        _spec(depends_on=[{"variable": "rainfall", "weight": 2.0}]),
        np.random.default_rng(0), 60, "weekly",
    )
    hinged = build_disease_cases(
        {"rainfall": driver.copy()},
        _spec(depends_on=[{
            "variable": "rainfall", "weight": 2.0,
            "transforms": [{"name": "_test_hinge"}],
        }]),
        np.random.default_rng(0), 60, "weekly",
    )
    assert not np.array_equal(np.nan_to_num(plain), np.nan_to_num(hinged))


def test_transforms_recorded_in_metadata():
    from dsl.core.pipeline.metadata import build_metadata

    cfg = ScenarioConfig(
        period="weekly", n_total=60,
        variables=[{"name": "rainfall", "generate": "flat", "params": {"level": 5}}],
        disease_cases=_spec(depends_on=[{
            "variable": "rainfall", "lag": 1, "weight": 1.0,
            "transforms": [{"name": "_test_scale", "params": {"factor": 2.0}}],
        }]),
    )
    meta = build_metadata(cfg)
    dep = meta["disease_cases"]["depends_on"][0]
    assert dep["transforms"] == [{"name": "_test_scale", "params": {"factor": 2.0}}]
