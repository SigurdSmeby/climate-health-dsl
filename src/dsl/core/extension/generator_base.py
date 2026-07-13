"""Base class and registry for variable generators.

A *generator* CREATES a variable's time series from nothing — parameters,
the time axis, and randomness. (A *transform* modifies an existing series;
see ``transform_base.py``.)
"""
from abc import ABC, abstractmethod

import numpy as np

from .registry import Registry

generator_registry = Registry("generator")
register_generator = generator_registry.register
get_generator = generator_registry.get


class VariableGenerator(ABC):
    """Base class for anything that creates a variable's time series."""

    @abstractmethod
    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Return an array of length ``n_periods``.

        ``period`` is the resolution ("daily"/"weekly"/...), so seasonality
        can scale via ``periods_per_year``. All randomness must come from
        ``rng`` (the seeded generator threaded through the run) so output
        is reproducible.
        """
