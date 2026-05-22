"""Shared pytest fixtures.

A fixture is a function pytest runs before each test that asks for it (by
naming it as an argument). The seeded ``rng`` here makes every test that
draws random numbers deterministic.
"""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """A fresh, seeded random generator — same numbers in every test run."""
    return np.random.default_rng(0)
