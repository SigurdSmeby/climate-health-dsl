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

    One ``series`` list: two plain climate series plus a count series (the
    disease), which is marked by its ``counts`` block. Population lives on
    the location, since a count series draws against where it happens.

    Tests tweak it via keyword overrides, e.g. ``scenario_dict(n_total=10)``.
    Returns a plain dict; call ``parse_config`` on it for a typed config.
    """
    data = {
        "period": "weekly",
        "n_total": 78,
        "seed": 42,
        "locations": {"loc": {"population": 100_000}},
        "series": [
            {"name": "rainfall", "generate": "seasonal_spike"},
            {"name": "mean_temperature", "generate": "seasonal_smooth"},
            {
                "name": "disease_cases",
                "counts": {},
                "depends_on": [
                    {"series": "rainfall", "lag": 3, "weight": 2.0},
                    {"series": "mean_temperature", "lag": 3, "weight": 1.0},
                ],
            },
        ],
    }
    data.update(overrides)
    return data


def series_dict(name: str, **fields) -> dict:
    """One entry for a scenario's ``series`` list.

    Defaults to a plain ``flat`` climate series; pass ``counts={...}`` to make
    it a count series, or ``depends_on=[...]`` to give it parents.
    """
    entry = {"name": name}
    entry.update(fields)
    if "generate" not in entry and "depends_on" not in entry:
        entry["generate"] = "flat"
    return entry


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
