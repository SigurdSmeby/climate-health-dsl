"""Defines what every transform must look like, plus their registry.

A *transform* MODIFIES a series that already exists — array in, changed
array out. (A *generator*, by contrast, creates a series from nothing; see
``generator_base.py``.) Lag, missing-value injection, and noise are
transforms: they never invent a variable, only alter one.
"""
from abc import ABC, abstractmethod

import numpy as np

from .registry import Registry  # same folder (core/extension/), so relative

# The single registry instance every transform file registers itself into.
transform_registry = Registry("transform")
register_transform = transform_registry.register  # convenience aliases
get_transform = transform_registry.get


class Transform(ABC):
    """Base class for anything that MODIFIES an existing time series.

    Subclasses must implement ``apply`` — the abstract method guarantees
    every registered transform can be called the same way.
    """

    @abstractmethod
    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a modified copy of ``series`` (never mutate the input).

        Parameters
        ----------
        series:
            The values to modify.
        rng:
            The single seeded random generator threaded through the run —
            any randomness (e.g. which entries go missing) must come from
            it, so the output is reproducible.
        """
