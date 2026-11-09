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
    # Thesis check: run a scenario with a hinge relationship and confirm the
    # disease responds ONLY where the driver exceeds the threshold, so the
    # planted nonlinearity is visible in the generated data.
    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run
    from tests.conftest import scenario_dict as make_config_dict
    from tests.conftest import series_dict

    n = 120
    config = parse_config(
        make_config_dict(
            n_total=n,
            locations={"loc": {"population": 500_000}},
            series=[
                # Rises linearly: only the second half exceeds threshold 5.
                series_dict(
                    "rainfall",
                    generate="linear_trend",
                    params={"start": 0.0, "slope": 10.0 / n, "noise": 0.0},
                ),
                series_dict(
                    "disease_cases",
                    counts={"median_rate": 0.1, "max_rate": 0.4},
                    depends_on=[
                        {
                            "series": "rainfall",
                            "weight": 4.0,
                            "transforms": [
                                {
                                    "name": "threshold",
                                    "params": {"mode": "hinge", "threshold": 5.0},
                                }
                            ],
                        }
                    ],
                ),
            ],
        )
    )
    counts = run(config)["disease_cases"].to_numpy()
    below = np.nanmean(counts[: n // 2])
    above = np.nanmean(counts[n // 2 :])
    # Above-threshold mean must be clearly higher — the planted nonlinearity.
    assert above > below * 1.5


def test_a_non_numeric_threshold_is_rejected():
    """A string reached numpy as a raw UFuncTypeError, naming neither the
    transform nor the scenario."""
    with pytest.raises(ValueError, match="threshold must be a number"):
        ThresholdTransform(mode="hinge", threshold="abc")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_threshold_is_rejected(bad):
    """NaN compares false against every value, so `hinge` returned an
    all-NaN series — silently, looking like a driver that stopped
    mattering rather than a typo."""
    with pytest.raises(ValueError, match="finite"):
        ThresholdTransform(mode="hinge", threshold=bad)


def test_an_ordinary_threshold_still_works():
    """The guard must not reject the normal case, including an int."""
    import numpy as np

    out = ThresholdTransform(mode="hinge", threshold=2).apply(
        np.arange(5.0), np.random.default_rng(1)
    )
    assert list(out) == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_peak_is_an_inverted_u(rng):
    # The shape Halvard asked for: the threshold is an OPTIMUM, so it
    # contributes most and both sides contribute less. This is the opposite
    # of quadratic, which is why it is a separate mode.
    series = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
    result = ThresholdTransform(mode="peak", threshold=25.0).apply(series, rng)
    assert result[2] == 0.0          # at the optimum: the maximum
    assert result[2] > result[1] > result[0]  # falls off below
    assert result[2] > result[3] > result[4]  # and above


def test_peak_is_the_negation_of_quadratic(rng):
    series = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
    u = ThresholdTransform(mode="quadratic", threshold=25.0).apply(series, rng)
    inverted = ThresholdTransform(mode="peak", threshold=25.0).apply(series, rng)
    assert np.allclose(inverted, -u)


def test_peak_is_symmetric_around_the_optimum(rng):
    series = np.array([22.0, 28.0])  # equidistant from 25
    result = ThresholdTransform(mode="peak", threshold=25.0).apply(series, rng)
    assert result[0] == result[1]


def test_nan_preserved_peak(rng):
    result = ThresholdTransform(mode="peak", threshold=10.0).apply(
        np.array([10.0, np.nan]), rng
    )
    assert result[0] == 0.0
    assert np.isnan(result[1])


def test_quadratic_still_means_what_it_meant(rng):
    # Guards the compatibility promise: adding `peak` must not change the
    # numbers any existing scenario already generates.
    series = np.array([15.0, 25.0, 35.0])
    result = ThresholdTransform(mode="quadratic", threshold=25.0).apply(series, rng)
    assert np.allclose(result, np.array([100.0, 0.0, 100.0]))


def test_an_unhandled_mode_raises_rather_than_falling_through(rng):
    # __init__ makes this unreachable from YAML, so this guards maintenance:
    # a mode added to _MODES without a branch in apply() must fail loudly
    # instead of silently returning the negated quadratic.
    transform = ThresholdTransform(mode="hinge", threshold=1.0)
    transform.mode = "mode_added_without_a_branch"
    with pytest.raises(ValueError, match="unhandled mode"):
        transform.apply(np.array([5.0]), rng)
