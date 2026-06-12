"""Shared pytest fixtures and helpers.

A fixture is a function pytest runs before each test that asks for it (by
naming it as an argument). The seeded ``rng`` makes random-drawing tests
deterministic; ``scenario_dict`` and ``write_csv`` are shared builders used
across the suite.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """A fresh, seeded random generator — same numbers in every test run."""
    return np.random.default_rng(0)


def scenario_dict(**overrides) -> dict:
    """A minimal valid scenario dict (weekly, two climate drivers).

    Tests tweak it via keyword overrides, e.g. ``scenario_dict(n_total=10)``.
    Returns a plain dict; call ``parse_config`` on it for a typed config.
    """
    data = {
        "period": "weekly",
        "n_total": 78,
        "seed": 42,
        "variables": [
            {"name": "rainfall", "generate": "seasonal_spike"},
            {"name": "mean_temperature", "generate": "seasonal_smooth"},
        ],
        "disease_cases": {
            "population": 100_000,
            "depends_on": [
                {"variable": "rainfall", "lag": 3, "weight": 2.0},
                {"variable": "mean_temperature", "lag": 3, "weight": 1.0},
            ],
        },
    }
    data.update(overrides)
    return data


def write_csv(path, periods, **columns) -> str:
    """Write a CHAP-format CSV with a ``time_period`` column plus the given
    data columns, and return its path as a string.

    Scalars are broadcast to every period; lists must match ``periods``'s
    length. Example: ``write_csv(tmp/"x.csv", periods, rainfall=[1, 2, 3])``.
    """
    data = {"time_period": list(periods)}
    data.update(columns)
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)
