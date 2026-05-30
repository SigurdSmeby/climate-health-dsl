"""Ground-truth recovery: prove the simulator embeds the lag it claims.

This is the validity check for the whole project. A scenario declares
"disease follows this driver by k periods"; if a simple statistical check
on the OUTPUT recovers that same k, the relationship is genuinely in the
data — known ground truth, not an assumption.

Method: for each candidate shift s, correlate the driver with the disease
series moved s periods earlier; the best-correlating shift is the
recovered lag.

Note on drivers: the recovery scenario uses an APERIODIC driver
(seasonal_smooth with amplitude 0, i.e. pure noise around a mean). With a
seasonal driver, the disease's own seasonal baseline correlates with the
driver too and can bias the recovered lag by a period — a real
epidemiological confound (seasonality masks lag structure), not a bug.
The seasonal case is covered separately with a ±1 tolerance.
"""
import numpy as np

from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run

TRUE_LAG = 5
MAX_SHIFT = 12


def recovered_lag(driver: np.ndarray, cases: np.ndarray) -> int:
    """Return the shift whose driver↔cases correlation is strongest."""
    n = len(driver)
    correlations = []
    for shift in range(MAX_SHIFT + 1):
        shifted_cases = cases[shift:]
        aligned_driver = driver[: n - shift]
        valid = ~np.isnan(shifted_cases)
        correlations.append(
            np.corrcoef(aligned_driver[valid], shifted_cases[valid])[0, 1]
        )
    return int(np.argmax(correlations))


def test_recovers_known_lag_from_aperiodic_driver():
    # One noisy driver causes disease with lag 5. Four years of weekly data
    # gives the correlation plenty of signal.
    config = parse_config(
        {
            "period": "weekly",
            "n_total": 208,
            "seed": 42,
            "variables": [
                {
                    "name": "rainfall",
                    "generate": "seasonal_smooth",
                    # amplitude 0 + noise: an aperiodic (pure-noise) driver,
                    # so the seasonal baseline cannot confound the lag.
                    "params": {"mean": 10.0, "amplitude": 0.0, "noise": 2.0},
                }
            ],
            "disease_cases": {
                "population": 100_000,
                "depends_on": [
                    {"variable": "rainfall", "lag": TRUE_LAG, "weight": 2.0}
                ],
            },
        }
    )
    df = run(config)
    lag = recovered_lag(df["rainfall"].to_numpy(), df["disease_cases"].to_numpy())
    assert lag == TRUE_LAG


def test_recovery_is_stable_across_seeds():
    # The embedded relationship must not depend on a lucky seed.
    for seed in (0, 1, 2):
        config = parse_config(
            {
                "period": "weekly",
                "n_total": 208,
                "seed": seed,
                "variables": [
                    {
                        "name": "rainfall",
                        "generate": "seasonal_smooth",
                        "params": {"mean": 10.0, "amplitude": 0.0, "noise": 2.0},
                    }
                ],
                "disease_cases": {
                    "population": 100_000,
                    "depends_on": [
                        {"variable": "rainfall", "lag": TRUE_LAG, "weight": 2.0}
                    ],
                },
            }
        )
        df = run(config)
        lag = recovered_lag(
            df["rainfall"].to_numpy(), df["disease_cases"].to_numpy()
        )
        assert lag == TRUE_LAG, f"seed {seed} recovered {lag}, expected {TRUE_LAG}"


def test_seasonal_driver_recovers_lag_within_one_period():
    # The realistic case: a seasonal rainfall driver. The disease's own
    # seasonal baseline correlates with it and can pull the estimate off by
    # one period — so the assertion allows ±1. The exact-recovery guarantee
    # is the aperiodic test above.
    config = parse_config(
        {
            "period": "weekly",
            "n_total": 208,
            "seed": 42,
            "variables": [
                {
                    "name": "rainfall",
                    "generate": "seasonal_spike",
                    "params": {"spike_center": 20, "spike_width": 4.0},
                }
            ],
            "disease_cases": {
                "population": 100_000,
                "depends_on": [
                    {"variable": "rainfall", "lag": TRUE_LAG, "weight": 2.0}
                ],
            },
        }
    )
    df = run(config)
    lag = recovered_lag(df["rainfall"].to_numpy(), df["disease_cases"].to_numpy())
    assert abs(lag - TRUE_LAG) <= 1
