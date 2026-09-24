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

    def events(self) -> list[dict]:
        """Return the features this generator deliberately placed.

        Override this when a generator puts something at an identifiable
        period — a seasonal peak, an outbreak shock. Reporting them makes
        the periods exact ground truth, so anything counting features (which
        fold holds how many spikes, say) reads them instead of guessing from
        the data.

        Called after ``generate``, on the same instance, so an implementation
        may report what it actually drew.

        Returns:
            One dict per feature, each with at least ``kind`` (a short label)
            and ``index`` (the period it sits at); ``magnitude`` and
            ``duration`` where they are meaningful. Empty by default, which
            is right for any generator that plants nothing identifiable.
            Example: [{"kind": "seasonal_spike", "index": 26,
            "magnitude": 20.0}]
        """
        return []
