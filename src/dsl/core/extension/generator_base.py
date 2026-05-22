"""Defines what every variable generator must look like, plus their registry.

A *generator* CREATES a variable's time series from nothing — parameters,
the time axis, and randomness. (A *transform*, by contrast, modifies a
series that already exists; see ``transform_base.py``.)
"""
from abc import ABC, abstractmethod

import numpy as np

from .registry import Registry  # same folder (core/extension/), so relative

# The single registry instance every generator file registers itself into.
generator_registry = Registry("generator")
register_generator = generator_registry.register  # convenience aliases
get_generator = generator_registry.get


class VariableGenerator(ABC):
    """Base class for anything that CREATES a variable's time series.

    ABC stands for "abstract base class": a class that cannot be
    instantiated itself and exists only to declare methods subclasses MUST
    implement. That guarantee is what lets the engine call ``.generate()``
    on any registered generator without knowing which one it is.
    """

    @abstractmethod
    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Return an array of length ``n_periods`` (the variable's values).

        Parameters
        ----------
        n_periods:
            How many time steps to produce.
        period:
            The resolution ("daily"/"weekly"/"monthly"/"yearly"), so the
            generator can scale seasonality via ``periods_per_year``.
        rng:
            The single seeded random generator threaded through the whole
            run — all randomness must come from it, for reproducibility.
        """
