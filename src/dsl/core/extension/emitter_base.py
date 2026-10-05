"""Base class and registry for emitters.

An *emitter* turns a series' finished float signal into the column that
reaches the CSV. Everything before it is shared by every series — generator,
parents, events, persistence, gaps — so this is the one step that differs,
and adding an output type means adding one self-registering class here rather
than a branch in the engine.

(A *generator* creates a series from nothing, a *transform* modifies one; see
``generator_base.py`` and ``transform_base.py``.)
"""
from abc import ABC, abstractmethod

import numpy as np

from .registry import Registry

emitter_registry = Registry("emitter")
register_emitter = emitter_registry.register
get_emitter = emitter_registry.get


class Emitter(ABC):
    """Base class for anything that turns a signal into an output column."""

    @abstractmethod
    def emit(self, signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the output column for ``signal``.

        The signal is NaN wherever a lag warm-up or a missing parent leaves
        it undefined; those periods must stay NaN rather than receive a
        fabricated value. Any randomness must come from ``rng`` so output is
        reproducible.
        """
