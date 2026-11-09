"""Tests for _standardize, the per-parent z-score.

Standardizing puts every parent on one scale so YAML weights are
comparable. The interesting cases are the degenerate ones: a series that
does not really vary must contribute nothing, and a series that genuinely
varies must survive however small its values are.
"""
import numpy as np

from dsl.core.pipeline.signal import _standardize


def test_a_varying_series_is_z_scored():
    result = _standardize(np.array([10.0, 20.0, 30.0]))
    assert np.isclose(result.mean(), 0.0)
    assert np.isclose(result.std(), 1.0)


def test_an_exactly_constant_series_contributes_nothing():
    assert np.array_equal(_standardize(np.array([25.0] * 4)), np.zeros(4))


def test_a_near_constant_series_contributes_nothing():
    # The bug: 25.0 next to 25.000000000000004 — flat in intent, not in
    # bits, which a CSV or any arithmetic can produce. Its spread is a few
    # float64 eps, which missed an `== 0` guard, and dividing by it turned
    # rounding dust into a full-size signal. Measured end to end, a
    # visually flat rainfall column drove disease between 9 900 and
    # 26 900 — a 2.7x swing from a driver declared flat.
    near_constant = np.array([25.0, 25.000000000000004] * 2)
    assert np.array_equal(_standardize(near_constant), np.zeros(4))


def test_a_near_constant_negative_series_contributes_nothing():
    # Temperature in C is negative; the guard is on magnitude, not sign.
    series = np.array([-5.0, -5.0, -5.000000000000001])
    assert np.array_equal(_standardize(series), np.zeros(3))


def test_a_genuinely_varying_tiny_series_survives():
    # The other half of the fix, and the reason the tolerance is relative
    # with no absolute floor: these values are far smaller than the
    # rounding dust above, but they really do vary, so zeroing them would
    # be the same bug pointing the other way.
    result = _standardize(np.array([1e-18, 3e-18, 2e-18]))
    assert not np.allclose(result, 0.0)
    assert np.isclose(result.std(), 1.0)


def test_the_quietest_real_generator_output_survives():
    # `flat` with noise: 0.01 is the least-varying series a scenario can
    # ask for on purpose. Measured at 3.6e-4 of its own scale, roughly
    # twelve orders above the tolerance.
    result = _standardize(np.array([25.0, 25.009, 24.99, 25.004]))
    assert not np.allclose(result, 0.0)


def test_all_zeros_contributes_nothing():
    # scale is 0 here, so the tolerance is 0 — and std is 0 too, so the
    # <= comparison still catches it.
    assert np.array_equal(_standardize(np.array([0.0, 0.0, 0.0])), np.zeros(3))


def test_nan_positions_are_preserved_when_zeroed():
    # A blanked row must stay blank rather than be fabricated as 0.
    result = _standardize(np.array([25.0, np.nan, 25.000000000000004]))
    assert result[0] == 0.0
    assert np.isnan(result[1])
    assert result[2] == 0.0


def test_an_all_nan_series_stays_all_nan():
    assert np.all(np.isnan(_standardize(np.array([np.nan, np.nan]))))


def test_a_near_constant_csv_driver_does_not_move_the_disease(tmp_path, rng):
    """End to end, this is the symptom that mattered: a rainfall column
    that reads as a flat 25.0 drove disease between 9 900 and 26 900 — a
    2.7x swing from a driver the scenario declares flat, so the dataset
    contradicted its own ground truth."""
    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run
    from tests.conftest import scenario_dict as make_config_dict
    from tests.conftest import series_dict, write_csv

    periods = [f"2020-{m + 1:02d}" for m in range(12)]
    path = write_csv(
        tmp_path / "near.csv", periods,
        rainfall=[25.0 + (i % 2) * 4e-15 for i in range(12)],
    )
    config = parse_config(
        make_config_dict(
            period="monthly",
            n_total=12,
            series=[
                series_dict(
                    "rainfall",
                    generate="from_csv",
                    params={"file": str(path), "column": "rainfall"},
                ),
                series_dict(
                    "disease_cases",
                    counts={},
                    depends_on=[{"series": "rainfall", "weight": 2.0}],
                ),
            ],
        )
    )
    cases = run(config)["disease_cases"].to_numpy(dtype=float)
    # Poisson noise around a flat rate still varies, so this is a bound on
    # the driver's contribution rather than a demand for a constant column:
    # the 2.7x swing came from the standardized driver, not the emitter.
    assert cases.max() / cases.min() < 1.5
