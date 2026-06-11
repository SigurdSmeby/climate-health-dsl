"""Tests for the dependent disease signal — where the ground truth is created."""
import numpy as np
import pytest

from dsl.core.config.schema import DiseaseSpec
from dsl.core.pipeline.disease import build_disease_cases


def make_spec(**overrides) -> DiseaseSpec:
    """A minimal DiseaseSpec; tests tweak it via keyword overrides."""
    data = {
        "population": 100_000,
        "depends_on": [{"variable": "rainfall", "lag": 0, "weight": 1.0}],
    }
    data.update(overrides)
    return DiseaseSpec(**data)


def spiky_driver(n: int, peak: int, ppy: int = 52) -> np.ndarray:
    """A deterministic yearly-repeating driver with a clear peak."""
    pos = np.arange(n) % ppy
    dist = np.minimum(np.abs(pos - peak), ppy - np.abs(pos - peak))
    return 10.0 * np.exp(-0.5 * (dist / 3.0) ** 2)


def test_output_shape_and_integer_counts(rng):
    drivers = {"rainfall": spiky_driver(104, peak=20)}
    cases = build_disease_cases(drivers, make_spec(), rng, 104, "weekly")
    assert len(cases) == 104
    valid = cases[~np.isnan(cases)]
    # Counts: non-negative whole numbers, never above the population.
    assert (valid >= 0).all()
    assert (valid <= 100_000).all()
    assert np.array_equal(valid, np.round(valid))


def test_disease_tracks_single_driver(rng):
    # Weight 1, lag 0, no AR, no missing: cases must rise and fall with the
    # driver, so their correlation should be strongly positive.
    drivers = {"rainfall": spiky_driver(104, peak=20)}
    cases = build_disease_cases(drivers, make_spec(), rng, 104, "weekly")
    valid = ~np.isnan(cases)
    corr = np.corrcoef(drivers["rainfall"][valid], cases[valid])[0, 1]
    assert corr > 0.6


def test_lag_shifts_disease_signal(rng):
    # With a lag of k, the disease signal follows the driver k periods
    # later: of all candidate shifts, aligning the driver shifted by k must
    # correlate best with the cases.
    #
    # The driver here is deliberately APERIODIC (random, not seasonal).
    # Disease has its own seasonal baseline; with a seasonal driver, that
    # baseline correlates with the driver too and systematically biases the
    # recovered lag by a period. An aperiodic driver isolates the lag
    # mechanism itself.
    k = 5
    n = 208  # four years, so the correlation has plenty of signal
    driver = np.random.default_rng(99).normal(0.0, 1.0, n) * 5 + 10
    drivers = {"rainfall": driver}
    spec = make_spec(depends_on=[{"variable": "rainfall", "lag": k, "weight": 2.0}])
    cases = build_disease_cases(drivers, spec, rng, n, "weekly")

    def corr_at_shift(s: int) -> float:
        shifted_cases = cases[s:]
        driver = drivers["rainfall"][: n - s]
        valid = ~np.isnan(shifted_cases)
        return np.corrcoef(driver[valid], shifted_cases[valid])[0, 1]

    correlations = [corr_at_shift(s) for s in range(11)]
    assert int(np.argmax(correlations)) == k


def test_warmup_rows_are_nan(rng):
    drivers = {"rainfall": spiky_driver(52, peak=20)}
    spec = make_spec(depends_on=[{"variable": "rainfall", "lag": 3, "weight": 1.0}])
    cases = build_disease_cases(drivers, spec, rng, 52, "weekly")
    # The first max_lag rows have no valid lagged signal.
    assert np.isnan(cases[:3]).all()
    assert not np.isnan(cases[3:]).any()  # no missing_rate set


def test_missing_rate_injects_nan_beyond_warmup(rng):
    drivers = {"rainfall": spiky_driver(1000, peak=20)}
    spec = make_spec(missing_rate=0.3)
    cases = build_disease_cases(drivers, spec, rng, 1000, "weekly")
    nan_fraction = np.isnan(cases).mean()
    assert nan_fraction == pytest.approx(0.3, abs=0.05)


def test_reproducible_under_fixed_seed():
    drivers = {"rainfall": spiky_driver(104, peak=20)}
    a = build_disease_cases(
        drivers, make_spec(), np.random.default_rng(3), 104, "weekly"
    )
    b = build_disease_cases(
        drivers, make_spec(), np.random.default_rng(3), 104, "weekly"
    )
    c = build_disease_cases(
        drivers, make_spec(), np.random.default_rng(4), 104, "weekly"
    )
    # NaN != NaN, so compare with equal_nan.
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, c, equal_nan=True)


def test_autoregressive_changes_signal(rng):
    drivers = {"rainfall": spiky_driver(104, peak=20)}
    plain = build_disease_cases(
        drivers, make_spec(), np.random.default_rng(3), 104, "weekly"
    )
    ar = build_disease_cases(
        drivers,
        make_spec(autoregressive=True),
        np.random.default_rng(3),
        104,
        "weekly",
    )
    assert not np.array_equal(plain, ar, equal_nan=True)


def test_weights_change_driver_influence(rng):
    # A heavily weighted driver should dominate the signal more than a
    # zero-ish weighted one: correlation with the driver must be higher.
    driver = spiky_driver(208, peak=20)
    drivers = {"rainfall": driver}
    heavy = make_spec(depends_on=[{"variable": "rainfall", "weight": 5.0}])
    light = make_spec(depends_on=[{"variable": "rainfall", "weight": 0.0}])
    cases_heavy = build_disease_cases(
        drivers, heavy, np.random.default_rng(3), 208, "weekly"
    )
    cases_light = build_disease_cases(
        drivers, light, np.random.default_rng(3), 208, "weekly"
    )
    corr_heavy = np.corrcoef(driver, cases_heavy)[0, 1]
    corr_light = np.corrcoef(driver, cases_light)[0, 1]
    assert corr_heavy > corr_light


def test_max_rate_caps_incidence(rng):
    # Even with an extreme driver pushing the sigmoid to 1, the rate is at
    # most max_rate * population, so counts stay in that ballpark.
    drivers = {"rainfall": 1000.0 * spiky_driver(104, peak=20)}
    spec = make_spec(max_rate=0.1, median_rate=0.05)
    cases = build_disease_cases(drivers, spec, rng, 104, "weekly")
    # Poisson noise can exceed the mean a little; 1.2x is generous slack.
    assert np.nanmax(cases) <= 0.1 * 100_000 * 1.2


def test_negative_binomial_is_more_variable_than_poisson():
    # Same scenario, same seed: NB counts must scatter more widely than
    # Poisson around a similar mean (that's the whole point of overdispersion).
    drivers = {"rainfall": spiky_driver(600, peak=20)}
    poisson = build_disease_cases(
        drivers, make_spec(), np.random.default_rng(0), 600, "weekly"
    )
    nb = build_disease_cases(
        drivers,
        make_spec(count_distribution="negative_binomial", overdispersion=2.0),
        np.random.default_rng(0),
        600,
        "weekly",
    )
    p_valid = poisson[~np.isnan(poisson)]
    nb_valid = nb[~np.isnan(nb)]
    # Means stay in the same ballpark; NB variance is clearly larger.
    assert np.mean(nb_valid) == pytest.approx(np.mean(p_valid), rel=0.3)
    assert np.var(nb_valid) > np.var(p_valid)


def test_negative_binomial_still_non_negative_integers_capped(rng):
    drivers = {"rainfall": spiky_driver(200, peak=20)}
    spec = make_spec(count_distribution="negative_binomial", overdispersion=1.0)
    cases = build_disease_cases(drivers, spec, rng, 200, "weekly")
    valid = cases[~np.isnan(cases)]
    assert (valid >= 0).all()
    assert (valid <= 100_000).all()
    assert np.array_equal(valid, np.round(valid))


def test_negative_binomial_reproducible():
    drivers = {"rainfall": spiky_driver(104, peak=20)}
    spec = make_spec(count_distribution="negative_binomial", overdispersion=3.0)
    a = build_disease_cases(drivers, spec, np.random.default_rng(5), 104, "weekly")
    b = build_disease_cases(drivers, spec, np.random.default_rng(5), 104, "weekly")
    assert np.array_equal(a, b, equal_nan=True)


def test_poisson_is_the_default(rng):
    # Omitting count_distribution must give byte-identical output to the
    # explicit poisson value (proves the default is non-breaking).
    drivers = {"rainfall": spiky_driver(104, peak=20)}
    default = build_disease_cases(
        drivers, make_spec(), np.random.default_rng(1), 104, "weekly"
    )
    explicit = build_disease_cases(
        drivers,
        make_spec(count_distribution="poisson"),
        np.random.default_rng(1),
        104,
        "weekly",
    )
    assert np.array_equal(default, explicit, equal_nan=True)


def test_constant_driver_with_nan_still_blanks(rng):
    # Bug #14: a constant driver hits _standardize's zero-variance branch,
    # which must STILL preserve the NaN mask so a missing input blanks the
    # disease row (otherwise #7's fix is bypassed for constant drivers).
    driver = np.full(10, 5.0)
    driver[3] = np.nan
    drivers = {"rainfall": driver}
    spec = make_spec(depends_on=[{"variable": "rainfall", "lag": 0, "weight": 2.0}])
    cases = build_disease_cases(drivers, spec, rng, 10, "weekly")
    assert np.isnan(cases[3]), "missing input must blank disease even for a constant driver"


def test_zero_weight_dependency_does_not_blank(rng):
    # Bug #23: a weight-0 dependency contributes nothing, so its lag warm-up
    # must NOT blank disease rows. Compare to the same scenario at lag 0.
    driver = np.arange(12, dtype=float) + 1.0
    drivers = {"rainfall": driver}
    spec = make_spec(
        depends_on=[{"variable": "rainfall", "lag": 5, "weight": 0.0}]
    )
    cases = build_disease_cases(drivers, spec, rng, 12, "weekly")
    # No dependency actually affects the signal, so nothing should be blanked.
    assert not np.isnan(cases).any()


def test_zero_weight_missing_driver_does_not_blank(rng):
    # A NaN in a weight-0 driver must not blank disease either (it has no effect).
    driver = np.arange(10, dtype=float) + 1.0
    driver[4] = np.nan
    drivers = {"rainfall": driver}
    spec = make_spec(depends_on=[{"variable": "rainfall", "lag": 0, "weight": 0.0}])
    cases = build_disease_cases(drivers, spec, rng, 10, "weekly")
    assert not np.isnan(cases[4])


def test_nan_covariate_blanks_disease_cases(rng):
    # Bug #7: a missing (NaN) driver value at period t must blank disease_cases
    # at t — you can't compute a known signal from a missing input. Previously
    # the NaN was nan_to_num'd to the mean, fabricating a confident count.
    driver = np.arange(10, dtype=float) + 1.0
    driver[5] = np.nan  # one missing real value
    drivers = {"rainfall": driver}
    spec = make_spec(depends_on=[{"variable": "rainfall", "lag": 0, "weight": 2.0}])
    cases = build_disease_cases(drivers, spec, rng, 10, "weekly")
    assert np.isnan(cases[5]), "disease_cases must be NaN where the driver is NaN"
    # Other rows are unaffected (still valid counts).
    assert not np.isnan(cases[0])
    assert not np.isnan(cases[9])


def test_nan_covariate_blanks_after_lag(rng):
    # The blanking must follow the lag: a NaN at driver index k surfaces in
    # disease_cases at index k + lag.
    driver = np.arange(12, dtype=float) + 1.0
    driver[3] = np.nan
    drivers = {"rainfall": driver}
    spec = make_spec(depends_on=[{"variable": "rainfall", "lag": 2, "weight": 1.0}])
    cases = build_disease_cases(drivers, spec, rng, 12, "weekly")
    assert np.isnan(cases[5])  # 3 + lag 2


def test_extreme_weight_no_overflow_warning(rng):
    # Bug #6: a huge weight made np.exp overflow and warn, even though the
    # output is fine (the sigmoid saturates). No warning should leak.
    import warnings

    drivers = {"rainfall": spiky_driver(104, peak=20)}
    spec = make_spec(depends_on=[{"variable": "rainfall", "lag": 1, "weight": 5000.0}])
    with warnings.catch_warnings():
        # Escalate only the numpy overflow warning to an error.
        warnings.filterwarnings("error", message="overflow encountered")
        cases = build_disease_cases(drivers, spec, rng, 104, "weekly")
    valid = cases[~np.isnan(cases)]
    assert (valid >= 0).all() and (valid <= 100_000).all()


def test_constant_driver_does_not_crash(rng):
    # A constant series has zero variance; standardization must not divide
    # by zero and produce NaN/inf.
    drivers = {"rainfall": np.full(52, 7.0)}
    cases = build_disease_cases(drivers, make_spec(), rng, 52, "weekly")
    assert not np.isnan(cases).any()
