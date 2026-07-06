"""Tests for the distributed-lag transform.

TDD: written before src/dsl/transforms/distributed_lag.py. Real climate effects
smear across several lags (a kernel over weeks), not a single delay — this is the
DLNM setup forecasters are judged on.
"""
import numpy as np
import pytest

from dsl.transforms.distributed_lag import DistributedLagTransform


def test_single_weight_at_lag0_is_identity(rng):
    series = np.array([1.0, 2.0, 3.0, 4.0])
    result = DistributedLagTransform(weights=[1.0]).apply(series, rng)
    assert np.array_equal(result, series)


def test_impulse_spreads_over_kernel(rng):
    # An impulse convolved with [0.5, 0.3, 0.2] appears at t, t+1, t+2 scaled.
    # Place the impulse past the k-1=2 warm-up positions so all three land.
    series = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    result = DistributedLagTransform(weights=[0.5, 0.3, 0.2]).apply(series, rng)
    assert result[2] == pytest.approx(0.5)  # lag 0
    assert result[3] == pytest.approx(0.3)  # lag 1
    assert result[4] == pytest.approx(0.2)  # lag 2


def test_warmup_is_nan(rng):
    # Kernel of length 3 → the first 2 positions have no full past → NaN
    # (causal, matching the single-lag convention the disease model blanks).
    series = np.array([10.0, 20.0, 30.0, 40.0])
    result = DistributedLagTransform(weights=[0.5, 0.3, 0.2]).apply(series, rng)
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert not np.isnan(result[2])


def test_never_wraps(rng):
    # Must be causal, not circular: the end spike must not leak to the start.
    # Position 0 is warm-up (NaN) for a length-2 kernel; the point is the spike
    # only shows up at its own index and after, never before.
    series = np.array([0.0, 0.0, 0.0, 9.0])  # spike at the end
    result = DistributedLagTransform(weights=[1.0, 1.0]).apply(series, rng)
    assert result[1] == 0.0 and result[2] == 0.0  # nothing before the spike
    assert result[3] == 9.0  # spike appears at its own position


def test_does_not_modify_input(rng):
    series = np.array([1.0, 2.0, 3.0, 4.0])
    original = series.copy()
    DistributedLagTransform(weights=[0.5, 0.5]).apply(series, rng)
    assert np.array_equal(series, original)


def test_empty_weights_rejected():
    with pytest.raises(ValueError, match="weights"):
        DistributedLagTransform(weights=[])


def test_registered_and_reachable():
    from dsl.core.extension.transform_base import get_transform

    assert get_transform("distributed_lag") is DistributedLagTransform


def test_planted_kernel_is_recoverable():
    # Thesis check: a driver spike should produce elevated disease across the
    # kernel window, not just at one offset — the distributed-lag ground truth.
    from dsl.core.config.schema import DiseaseSpec
    from dsl.core.pipeline.disease import build_disease_cases

    n = 120
    driver = np.zeros(n)
    driver[40] = 20.0  # one sharp climate event
    spec = DiseaseSpec(
        population=500_000, median_rate=0.1, max_rate=0.4,
        depends_on=[{
            "variable": "rainfall", "weight": 4.0,
            "transforms": [{"name": "distributed_lag",
                            "params": {"weights": [0.4, 0.3, 0.2, 0.1]}}],
        }],
    )
    counts = build_disease_cases(
        {"rainfall": driver}, spec, np.random.default_rng(0), n, "weekly",
    )
    # Disease in the 4-period window after the event exceeds the quiet baseline.
    window = np.nanmean(counts[41:45])
    baseline = np.nanmean(counts[60:100])
    assert window > baseline
