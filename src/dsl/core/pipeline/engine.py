"""The orchestrator: translate a validated ScenarioConfig into a DataFrame.

This is the "translate config into executable code" step of the DSL. The
engine never hard-codes which generators exist: it reads each variable's
``generate:`` string and looks it up in the registry. Column names come
from the variable names in the YAML — a CHAP-ready scenario simply names
its variables ``rainfall`` and ``mean_temperature``.
"""
import numpy as np
import pandas as pd

# Importing the extension packages is what triggers auto-discovery: every
# generator/transform module runs its @register decorator on import. Without
# these two lines the registries would be empty.
import dsl.generators  # noqa: F401
import dsl.transforms  # noqa: F401
from dsl.core.config.schema import ScenarioConfig
from dsl.core.extension.generator_base import get_generator
from dsl.core.pipeline.disease import build_disease_cases
from dsl.core.pipeline.periods import format_period


def run(config: ScenarioConfig) -> pd.DataFrame:
    """Run the whole simulation described by ``config``.

    Returns a tidy DataFrame with one row per period and columns:
    ``time_period``, one column per YAML variable (in declaration order),
    ``disease_cases``, and a constant ``population``.
    """
    # The single seeded random generator. Every random draw in the run flows
    # from this one object, which is what makes output bit-for-bit
    # reproducible for a given seed.
    rng = np.random.default_rng(config.seed)

    # Generate each declared variable through the registry. get_generator
    # returns the CLASS registered under that name; calling it with the
    # YAML params builds an instance, whose .generate() makes the series.
    drivers: dict[str, np.ndarray] = {}
    for spec in config.variables:
        generator = get_generator(spec.generate)(**spec.params)
        drivers[spec.name] = generator.generate(config.n_total, config.period, rng)

    # Build the dependent signal from the drivers — the ground truth.
    disease = build_disease_cases(
        drivers, config.disease_cases, rng, config.n_total, config.period
    )

    # Assemble the tidy frame: CHAP-style label column first, then the
    # variables in YAML declaration order, then disease and population.
    columns: dict[str, object] = {
        "time_period": [
            format_period(i, config.period) for i in range(config.n_total)
        ]
    }
    columns.update(drivers)
    columns["disease_cases"] = disease
    # Population is a scalar in the config; pandas broadcasts it to a
    # constant column of length n_total.
    columns["population"] = config.disease_cases.population
    return pd.DataFrame(columns)
