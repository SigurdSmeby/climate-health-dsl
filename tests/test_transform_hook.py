"""Transforms are drop-in: a depends_on entry can name any registered
transform to reshape a parent's values before they are weighted in.

The point is that the hook reaches the registry, not just the built-in
lag/missing — so a transform added as one self-registering file is usable
from a scenario with no change to the core.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from dsl.core.config.schema import DependencySpec, parse_config
from dsl.core.extension.transform_base import Transform, register_transform
from dsl.core.pipeline.engine import run
from tests.conftest import scenario_dict as make_config_dict
from tests.conftest import series_dict


# A tiny test-only transform: multiply the series by a constant, registered
# under a name unlikely to clash.
@register_transform("_test_scale")
class _ScaleTransform(Transform):
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def apply(self, series, rng):
        return series.astype(float) * self.factor


def _scenario(transforms):
    """A two-series scenario whose child applies `transforms` to its parent."""
    return make_config_dict(
        series=[
            series_dict("rain", generate="seasonal_spike"),
            series_dict(
                "dam",
                depends_on=[
                    {
                        "series": "rain",
                        "lag": 1,
                        "weight": 1.0,
                        "transforms": transforms,
                    }
                ],
            ),
        ]
    )


def test_transforms_default_to_empty():
    dep = DependencySpec(series="rain")
    assert dep.transforms == []


def test_registry_transform_is_reachable_from_a_scenario():
    config = parse_config(
        _scenario([{"name": "_test_scale", "params": {"factor": 3.0}}])
    )
    assert config.series[1].depends_on[0].transforms[0].name == "_test_scale"
    assert run(config)["dam"].notna().any()


def test_unknown_transform_name_is_rejected():
    config = parse_config(_scenario([{"name": "no_such_transform"}]))
    with pytest.raises(KeyError, match="no_such_transform"):
        run(config)


def test_transform_params_reach_the_transform():
    """An invalid param surfaces as the transform's own error."""
    config = parse_config(
        _scenario([{"name": "threshold", "params": {"mode": "not_a_mode"}}])
    )
    with pytest.raises(ValueError, match="mode"):
        run(config)


def test_transforms_apply_in_order():
    """Scaling then thresholding differs from thresholding then scaling."""
    scale = {"name": "_test_scale", "params": {"factor": 5.0}}
    hinge = {"name": "threshold", "params": {"mode": "hinge", "threshold": 10.0}}
    first = run(parse_config(_scenario([scale, hinge])))["dam"].to_numpy()
    second = run(parse_config(_scenario([hinge, scale])))["dam"].to_numpy()
    assert not np.allclose(first[1:], second[1:])


def test_transform_spec_rejects_a_typoed_key():
    with pytest.raises(ValidationError, match="parms"):
        parse_config(_scenario([{"name": "_test_scale", "parms": {}}]))
