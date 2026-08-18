"""Base class and registry for transforms.

A *transform* MODIFIES a series that already exists — array in, changed
array out. (A *generator* creates a series from nothing; see
``generator_base.py``.)
"""
from abc import ABC, abstractmethod

import numpy as np

from .registry import Registry

transform_registry = Registry("transform")
register_transform = transform_registry.register
get_transform = transform_registry.get


class Transform(ABC):
    """Base class for anything that modifies an existing time series."""

    @abstractmethod
    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a modified COPY of ``series`` (never mutate the input).

        Any randomness must come from ``rng`` so output is reproducible.
        """
