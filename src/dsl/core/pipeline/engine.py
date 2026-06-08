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
from dsl.core.config.schema import PopulationSpec, ScenarioConfig
from dsl.core.extension.generator_base import get_generator
from dsl.core.pipeline.disease import build_disease_cases
from dsl.core.pipeline.periods import format_period, parse_period


def _resolve_population(
    source: "int | PopulationSpec",
    n_periods: int,
    period: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Turn a population source into a length-``n_periods`` integer array.

    A plain int becomes a constant array and draws NO randomness (so scalar
    scenarios stay byte-identical). A PopulationSpec is run through the
    generator registry like a covariate, then rounded to non-negative whole
    people (a population is integer and can't be negative).
    """
    if isinstance(source, int):
        # np.full broadcasts the constant; no rng touched.
        return np.full(n_periods, source, dtype=int)
    series = get_generator(source.generate)(**source.params).generate(
        n_periods, period, rng
    )
    return np.maximum(np.round(series), 0).astype(int)


def run(config: ScenarioConfig) -> pd.DataFrame:
    """Run the whole simulation described by ``config``.

    Returns a tidy, long-format DataFrame with ``n_total`` rows per location
    and columns: ``time_period``, ``location``, one column per YAML variable
    (in declaration order), ``disease_cases``, and a constant ``population``.
    """
    # The single seeded random generator. Every random draw in the run flows
    # from this one object, which is what makes output bit-for-bit
    # reproducible for a given seed. Locations draw from it in turn, so each
    # location gets its own independent (but reproducible) series.
    rng = np.random.default_rng(config.seed)

    location_frames = [
        _run_one_location(config, location, rng) for location in config.locations
    ]
    # Stack location blocks on top of each other (long format, the CHAP
    # convention); ignore_index renumbers the rows 0..N-1.
    return pd.concat(location_frames, ignore_index=True)


def _run_one_location(
    config: ScenarioConfig, location: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate the full series for a single named location."""
    # Generate each declared variable through the registry. get_generator
    # returns the CLASS registered under that name; calling it with the
    # YAML params builds an instance, whose .generate() makes the series.
    drivers: dict[str, np.ndarray] = {}
    for spec in config.variables:
        params = dict(spec.params)
        # When the scenario starts at a real-world period, a from_csv variable
        # must read its real data from that same period — otherwise the output
        # would label row 0 as start_period but fill it with the CSV's first
        # row (mismatched dates). Inject it unless the variable set its own.
        if (
            spec.generate == "from_csv"
            and config.start_period is not None
            and "start_period" not in params
        ):
            params["start_period"] = config.start_period
        generator = get_generator(spec.generate)(**params)
        drivers[spec.name] = generator.generate(config.n_total, config.period, rng)

    # Resolve this location's population to a per-period array (its own
    # override or the default; a constant, or a generated growth trajectory)
    # and build a disease spec carrying it, so the incidence model and the
    # population cap both use the right per-period number for this location.
    population = _resolve_population(
        config.population_for(location), config.n_total, config.period, rng
    )
    disease_spec = config.disease_cases.model_copy(update={"population": population})

    # Build the dependent signal from the drivers — the ground truth.
    disease = build_disease_cases(
        drivers, disease_spec, rng, config.n_total, config.period
    )

    # Where on the real calendar the series starts: row 0 is start_period
    # if set (e.g. "2010-07"), else the first period of the year 2000.
    if config.start_period is not None:
        start_year, offset = parse_period(config.start_period, config.period)
    else:
        start_year, offset = 2000, 0

    # Assemble the tidy frame: CHAP-style label column first, the location,
    # then the variables in YAML declaration order, disease, and population.
    columns: dict[str, object] = {
        "time_period": [
            format_period(i + offset, config.period, start_year)
            for i in range(config.n_total)
        ],
        # A scalar here is broadcast by pandas to a constant column.
        "location": location,
    }
    columns.update(drivers)
    columns["disease_cases"] = disease
    columns["population"] = population
    return pd.DataFrame(columns)
