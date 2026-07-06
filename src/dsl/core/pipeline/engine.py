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
import dsl.generators
import dsl.transforms  # noqa: F401
from dsl.core.config.schema import PopulationSpec, ScenarioConfig
from dsl.core.extension.generator_base import get_generator
from dsl.core.pipeline.disease import build_disease_cases
from dsl.core.pipeline.periods import format_period, parse_period


def _child_rng(seed: int, *keys: str) -> np.random.Generator:
    """A reproducible Generator derived from ``seed`` plus a component key.

    Each component (a named variable, a location's population, a location's
    disease draw) gets its OWN independent stream keyed by stable strings, so
    reordering variables/locations or adding an unused decoy cannot shift
    another component's draws. The key is hashed to ints fed into a
    SeedSequence alongside the scenario seed, so the same key always yields
    the same stream for a given seed.
    """
    # Turn each key into a stable non-negative int (Python's hash is salted
    # per-process, so use a fixed hashlib digest instead).
    import hashlib

    entropy = [int(seed) & 0xFFFFFFFF]
    for key in keys:
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:4]
        entropy.append(int.from_bytes(digest, "big"))
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _build_generator(name: str, params: dict, variable: str | None = None):
    """Instantiate a generator, turning an unexpected-param TypeError into a
    clear message naming the variable, generator, and bad param."""
    try:
        return get_generator(name)(**params)
    except TypeError as exc:
        where = f"variable '{variable}'" if variable else "population"
        raise ValueError(
            f"{where}: generator '{name}' got an invalid param ({exc})."
        ) from exc


def _resolve_population(
    source: "int | PopulationSpec",
    n_periods: int,
    period: str,
    rng: np.random.Generator,
    start_period: "str | None" = None,
) -> np.ndarray:
    """Turn a population source into a length-``n_periods`` integer array.

    A plain int becomes a constant array and draws NO randomness (so scalar
    scenarios stay byte-identical). A PopulationSpec is run through the
    generator registry like a covariate (with the scenario ``start_period``
    injected for a from_csv source, so its real values align to the output
    labels), then validated and rounded to non-negative whole people.
    """
    if isinstance(source, int):
        # np.full broadcasts the constant; no rng touched.
        return np.full(n_periods, source, dtype=int)
    params = dict(source.params)
    if (
        source.generate == "from_csv"
        and start_period is not None
        and "start_period" not in params
    ):
        params["start_period"] = start_period
    series = _build_generator(source.generate, params).generate(
        n_periods, period, rng
    )
    # A population must be a known, finite, non-negative headcount at every
    # period — a missing or non-finite value can't drive the incidence model.
    if not np.all(np.isfinite(series)):
        raise ValueError(
            f"population generator '{source.generate}' produced a missing or "
            f"non-finite value; population must be finite at every period."
        )
    return np.maximum(np.round(series), 0).astype(int)


def run(config: ScenarioConfig) -> pd.DataFrame:
    """Run the whole simulation described by ``config``.

    Returns a tidy, long-format DataFrame with ``n_total`` rows per location
    and columns: ``time_period``, ``location``, one column per YAML variable
    (in declaration order), ``disease_cases``, and a constant ``population``.
    """
    # Output is reproducible from config.seed, but each component draws from
    # its OWN stream (derived from the seed plus a stable key), so reordering
    # variables/locations or adding a decoy can't shift unrelated draws.
    location_frames = [
        _run_one_location(config, location) for location in config.locations
    ]
    # Stack location blocks on top of each other (long format, the CHAP
    # convention); ignore_index renumbers the rows 0..N-1.
    return pd.concat(location_frames, ignore_index=True)


def _run_one_location(config: ScenarioConfig, location: str) -> pd.DataFrame:
    """Generate the full series for a single named location."""
    seed = config.seed
    # Each variable gets its own rng keyed by location + name, so its values
    # depend on its name, not its position or the presence of other variables.
    drivers: dict[str, np.ndarray] = {}
    for spec in config.variables:
        params = dict(spec.params)
        if spec.generate == "from_csv":
            # Align a from_csv variable to the scenario's calendar: read its
            # real data starting at start_period (else row 0 would be mislabeled
            # as start_period), unless the variable set its own.
            if config.start_period is not None and "start_period" not in params:
                params["start_period"] = config.start_period
            # With no source_location and a multi-location CSV, give THIS output
            # location its own matching rows.
            if "source_location" not in params:
                csv_locations = get_generator("from_csv").locations_in(
                    params.get("file", "")
                )
                if len(csv_locations) > 1:
                    if location not in csv_locations:
                        raise ValueError(
                            f"from_csv: variable '{spec.name}' has no "
                            f"source_location and output location '{location}' "
                            f"is not in {params.get('file')} (available: "
                            f"{csv_locations}); set source_location or rename "
                            f"the location to match."
                        )
                    params["source_location"] = location
        generator = _build_generator(spec.generate, params, spec.name)
        var_rng = _child_rng(seed, location, "variable", spec.name)
        own = generator.generate(config.n_total, config.period, var_rng)
        if spec.shared:
            # Latent regional driver: a second component generated from a
            # location-INDEPENDENT stream (same for every location), mixed in by
            # `shared`. The sqrt weights keep the total variance ~constant, so
            # turning sharing up doesn't just scale the series. shared=1 → the
            # own component drops out and every location gets the same signal;
            # shared=0 never reaches here (kept byte-identical to the old path).
            shared_rng = _child_rng(seed, "shared", "variable", spec.name)
            shared_series = generator.generate(
                config.n_total, config.period, shared_rng
            )
            s = spec.shared
            own = np.sqrt(1.0 - s) * own + np.sqrt(s) * shared_series
        drivers[spec.name] = own

    # This location's population as a per-period array (its override or the
    # default; constant or a generated trajectory), threaded into the disease
    # spec so the incidence model and cap use it.
    population = _resolve_population(
        config.population_for(location),
        config.n_total,
        config.period,
        _child_rng(seed, location, "population"),
        config.start_period,
    )
    disease_spec = config.disease_cases.model_copy(update={"population": population})

    # Build the dependent signal from the drivers — the ground truth. Its own
    # stream means the disease draw is unaffected by how many covariates exist.
    disease = build_disease_cases(
        drivers,
        disease_spec,
        _child_rng(seed, location, "disease"),
        config.n_total,
        config.period,
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
