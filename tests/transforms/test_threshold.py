"""Tests for the nonlinear threshold transform.

TDD: written before src/dsl/transforms/threshold.py. Threshold reshapes a
driver so its effect on disease is NONLINEAR — the whole point is to plant a
relationship a linear model can't fully recover.
"""
import numpy as np
import pytest

from dsl.transforms.threshold import ThresholdTransform


def test_hinge_zeroes_below_threshold(rng):
    # hinge: below the threshold, no effect (0); above, the excess passes through.
    series = np.array([0.0, 5.0, 10.0, 15.0])
    result = ThresholdTransform(mode="hinge", threshold=10.0).apply(series, rng)
    assert result[0] == 0.0 and result[1] == 0.0  # below → 0
    assert result[2] == 0.0                        # at threshold → 0 excess
    assert result[3] == 5.0                        # above → excess (15 - 10)


def test_step_is_binary(rng):
    series = np.array([0.0, 9.9, 10.0, 20.0])
    result = ThresholdTransform(mode="step", threshold=10.0).apply(series, rng)
    assert np.array_equal(result, np.array([0.0, 0.0, 1.0, 1.0]))


def test_quadratic_is_symmetric_around_center(rng):
    # U-shape: effect grows with distance from center, both sides.
    series = np.array([2.0, 5.0, 8.0])  # center 5 → distances 3, 0, 3
    result = ThresholdTransform(mode="quadratic", threshold=5.0).apply(series, rng)
    assert result[1] == 0.0
    assert result[0] == result[2] == 9.0  # 3**2

def test_nan_preserved(rng):
    # A missing input must stay missing (the disease model blanks that row).
    series = np.array([np.nan, 20.0])
    result = ThresholdTransform(mode="hinge", threshold=10.0).apply(series, rng)
    assert np.isnan(result[0])
    assert result[1] == 10.0


def test_nan_preserved_step(rng):
    # step needs its own hand-written NaN mask (np.where's >= comparison
    # resolves NaN to the else-branch, not NaN) — covered separately from
    # hinge/quadratic, which get NaN propagation for free from numpy math.
    series = np.array([np.nan, 20.0])
    result = ThresholdTransform(mode="step", threshold=10.0).apply(series, rng)
    assert np.isnan(result[0])
    assert result[1] == 1.0


def test_nan_preserved_quadratic(rng):
    series = np.array([np.nan, 20.0])
    result = ThresholdTransform(mode="quadratic", threshold=10.0).apply(series, rng)
    assert np.isnan(result[0])
    assert result[1] == 100.0


def test_does_not_modify_input(rng):
    series = np.array([0.0, 20.0])
    original = series.copy()
    ThresholdTransform(mode="hinge", threshold=10.0).apply(series, rng)
    assert np.array_equal(series, original)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        ThresholdTransform(mode="bogus", threshold=1.0)


def test_registered_and_reachable_from_scenario():
    # Proves the drop-in path: the name resolves in the registry (Feature 0
    # then makes it usable from depends_on transforms).
    from dsl.core.extension.transform_base import get_transform

    assert get_transform("threshold") is ThresholdTransform


def test_planted_threshold_is_recoverable():
    # Thesis check: run the disease model with a hinge relationship and confirm
    # disease responds ONLY where the driver exceeds the threshold. A driver
    # that never crosses the threshold produces no threshold-driven signal.
    from dsl.core.config.schema import DiseaseSpec
    from dsl.core.pipeline.disease import build_disease_cases

    n = 120
    # Driver rises linearly; only the second half exceeds threshold 5.
    driver = np.linspace(0.0, 10.0, n)
    spec = DiseaseSpec(
        population=500_000, median_rate=0.1, max_rate=0.4,
        depends_on=[{
            "variable": "rainfall", "weight": 4.0,
            "transforms": [{"name": "threshold",
                            "params": {"mode": "hinge", "threshold": 5.0}}],
        }],
    )
    counts = build_disease_cases(
        {"rainfall": driver}, spec, np.random.default_rng(0), n, "weekly",
    )
    below = np.nanmean(counts[: n // 2])
    above = np.nanmean(counts[n // 2:])
    # Above-threshold mean must be clearly higher — the planted nonlinearity.
    assert above > below * 1.5
