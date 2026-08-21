"""The orchestrator: translate a validated ScenarioConfig into a DataFrame.

The engine never hard-codes which generators exist: it looks each variable's
``generate:`` string up in the registry. Column names come from the variable
names in the YAML.
"""
import hashlib

import numpy as np
import pandas as pd

# Importing the extension packages triggers auto-discovery: every generator/
# transform module registers itself on import.
import dsl.generators
import dsl.transforms  # noqa: F401
from dsl.core.config.schema import PopulationSpec, ScenarioConfig
from dsl.core.extension.generator_base import get_generator
from dsl.core.pipeline.disease import build_disease_cases
from dsl.core.pipeline.periods import format_period, parse_period


def run(config: ScenarioConfig) -> pd.DataFrame:
    """Run the whole simulation described by ``config``.

    Execute the full pipeline: for each location, generate all variables,
    resolve the population, build the disease signal, and assemble into a
    tidy, long-format DataFrame.

    Args:
        config: A validated ScenarioConfig object.

    Returns:
        A DataFrame with n_total rows per location and columns:
        [time_period, location, <variable columns>, disease_cases, population].
        Example shape: (108 rows, 5 cols) for 3 locations x 36 months.

    Errors Caught (raised to caller):
        ValueError: If generator instantiation fails or population is invalid.
        KeyError: If a generator name is not registered.
    """
    # Shared (regional) variables must be identical across locations; cache
    # each one the first time it's computed so later locations reuse it.
    shared_cache: dict[str, np.ndarray] = {}
    location_frames = [
        _run_one_location(config, location, shared_cache)
        for location in config.locations
    ]
    return pd.concat(location_frames, ignore_index=True)


def _run_one_location(
    config: ScenarioConfig,
    location: str,
    shared_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Generate the full time series for a single location.

    Generate all variables, resolve the population, build the disease
    incidence signal from the drivers, and assemble everything into a
    DataFrame.

    Args:
        config: The validated scenario configuration.
        location: The location identifier (e.g., "north", "south").
        shared_cache: Dict caching shared (regional) variables across
            locations (mutated in place by _generate_variable).

    Returns:
        A DataFrame with columns: time_period, location, one column per
        variable, disease_cases, population. One row per time period.
        Example: 36 rows for n_total=36.

    Errors Caught (raised to caller):
        ValueError: If a generator param is invalid or population is
            non-finite.
    """
    # Step 1: Generate all variables for this location.
    drivers = {}
    for spec in config.variables:
        drivers[spec.name] = _generate_variable(config, spec, location, shared_cache)
    # Now drivers = {"rainfall": [50.5, 59.3, ...], "humidity": [65.2, ...]}

    # Step 2: Resolve this location's population (its override or the
    # default), threaded into the disease spec so the incidence model and
    # cap use it.
    population = _resolve_population(
        config.population_for(location),
        config.n_total,
        config.period,
        _child_rng(config.seed, location, "population"),
        config.start_period,
    )
    disease_spec = config.disease_cases.model_copy(update={"population": population})
    # Now population = [100000, 100000, 100000, ...] (one value per period)

    # Step 3: Build the disease signal using the drivers.
    disease = build_disease_cases(
        drivers,
        disease_spec,
        _child_rng(config.seed, location, "disease"),
        config.n_total,
        config.period,
    )
    # Now disease = [nan, nan, 5.0, 8.0, 12.0, ...] (Poisson draws, warm-up NaN)

    # Step 4: Assemble into a DataFrame. Row 0 is start_period if set, else
    # the first period of the year 2000.
    if config.start_period is not None:
        start_year, offset = parse_period(config.start_period, config.period)
    else:
        start_year, offset = 2000, 0

    # Tidy frame: CHAP label column first, then location, the variables in
    # YAML declaration order, disease, and population.
    columns: dict[str, object] = {
        "time_period": [
            format_period(i + offset, config.period, start_year)
            for i in range(config.n_total)
        ],
        "location": location,  # scalar -> broadcast to a constant column
    }
    columns.update(drivers)
    columns["disease_cases"] = disease
    columns["population"] = population
    return pd.DataFrame(columns)
    # Returns columns: [time_period, location, <variables>, disease_cases,
    # population] with config.n_total rows.


def _generate_variable(
    config: ScenarioConfig,
    spec,
    location: str,
    shared_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Generate one variable's time series for one location.

    Instantiate the generator named in spec.generate and run it. If the
    variable has ``shared`` set, mix in a location-independent regional
    component so nearby locations correlate.

    Args:
        config: The validated scenario configuration.
        spec: The VariableSpec for this variable (name, generate, params,
            shared).
        location: The location identifier (e.g., "north", "south").
        shared_cache: Dict caching shared (regional) variables; mutated in
            place so later locations reuse the same shared draw.

    Returns:
        A numpy array of config.n_total float values (one per time period).
        Example: array([45.2, 54.1, 63.8, ...]).

    Errors Caught (raised to caller):
        ValueError: If the generator rejects a param, or a from_csv variable
            can't be matched to this output location.
    """
    params = dict(spec.params)
    if spec.generate == "from_csv":
        _inject_start_period(params, config.start_period)
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
    var_rng = _child_rng(config.seed, location, "variable", spec.name)
    own = generator.generate(config.n_total, config.period, var_rng)
    # Now own = this location's own draw, ignoring any shared component.

    if spec.shared:
        own = _mix_shared_component(config, spec, params, generator, own, shared_cache)
    return own


def _mix_shared_component(
    config: ScenarioConfig,
    spec,
    params: dict,
    generator,
    own: np.ndarray,
    shared_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Blend in a latent regional driver shared across locations.

    A second component drawn from a location-INDEPENDENT stream, mixed in by
    spec.shared. The sqrt weights keep total variance ~constant: shared=1
    means every location gets an identical series; shared=0 never reaches
    here (the plain own-only path stays byte-identical).

    Args:
        config: The validated scenario configuration.
        spec: The VariableSpec for this variable.
        params: This location's resolved generator params (may include a
            per-location source_location for from_csv).
        generator: The generator instance already built for this location.
        own: This location's own (unmixed) series.
        shared_cache: Dict caching the shared series; mutated in place.

    Returns:
        own and the shared series mixed as sqrt(1-s)*own + sqrt(s)*shared.

    Errors Caught (raised to caller):
        ValueError: If a from_csv variable has shared set but no
            source_location on a multi-location file (the shared series has
            no unambiguous source to draw from).
    """
    shared_series = shared_cache.get(spec.name)
    if shared_series is None:
        shared_rng = _child_rng(config.seed, "shared", "variable", spec.name)
        if spec.generate == "from_csv":
            # `generator` above is bound to THIS location's own
            # source_location (auto-matched per-location when the variable
            # set none) — reusing it would make the "shared" component just
            # this location's own data again. A shared series needs its OWN
            # location-independent source: use the variable's explicit
            # source_location if it set one, else this is ambiguous on a
            # multi-location file.
            if "source_location" not in dict(spec.params):
                csv_locations = get_generator("from_csv").locations_in(
                    params.get("file", "")
                )
                if len(csv_locations) > 1:
                    raise ValueError(
                        f"from_csv: variable '{spec.name}' has shared set "
                        f"but no source_location, and {params.get('file')} "
                        f"has several locations ({csv_locations}); the "
                        f"shared series needs one explicit source_location "
                        f"to draw from."
                    )
            shared_generator = _build_generator(
                spec.generate, dict(spec.params), spec.name
            )
        else:
            shared_generator = generator
        shared_series = shared_generator.generate(
            config.n_total, config.period, shared_rng
        )
        # Identical for every location, so compute once — except from_csv,
        # whose params (source_location) vary per location.
        if spec.generate != "from_csv":
            shared_cache[spec.name] = shared_series

    s = spec.shared
    return np.sqrt(1.0 - s) * own + np.sqrt(s) * shared_series


def _resolve_population(
    source: "int | PopulationSpec",
    n_periods: int,
    period: str,
    rng: np.random.Generator,
    start_period: "str | None" = None,
) -> np.ndarray:
    """Turn a population source into a length-n_periods integer array.

    If source is a fixed int, return a constant array (draws NO randomness,
    so scalar-population scenarios stay byte-identical). If source is a
    PopulationSpec, run it through the generator registry like a covariate,
    then round to non-negative whole people.

    Args:
        source: Either a fixed population (int) or a PopulationSpec.
        n_periods: Number of time periods.
        period: Period type (e.g., "monthly", "daily").
        rng: Seeded random generator for reproducibility (unused for a
            fixed int).
        start_period: Scenario start period, to align a from_csv source.

    Returns:
        An integer array of length n_periods, one population value per
        period. Example: array([100000, 100000, ...]) for fixed
        population=100000, or array([98500, 99200, 100100, ...]) for a
        generated population series.

    Errors Caught (raised to caller):
        ValueError: If the generated population contains NaN or Inf values.
    """
    # Early return if source is a fixed population (not a generator spec).
    if isinstance(source, int):
        return np.full(n_periods, source, dtype=int)

    params = dict(source.params)
    if source.generate == "from_csv":
        _inject_start_period(params, start_period)
    series = _build_generator(source.generate, params).generate(
        n_periods, period, rng
    )
    if not np.all(np.isfinite(series)):
        raise ValueError(
            f"population generator '{source.generate}' produced a missing or "
            f"non-finite value; population must be finite at every period."
        )
    return np.maximum(np.round(series), 0).astype(int)


def _child_rng(seed: int, *keys: str) -> np.random.Generator:
    """Create a reproducible Generator derived from seed plus component keys.

    Each component (a variable, a location's population, a disease draw)
    gets its OWN stream keyed by stable strings, so reordering variables/
    locations or adding a decoy cannot shift another component's draws.
    Keys are hashed with sha256 (Python's hash() is salted per-process, so
    it can't be used directly for a reproducible seed).

    Args:
        seed: The scenario's base seed.
        *keys: Component keys identifying this stream (e.g. location,
            "variable", variable name).

    Returns:
        A seeded np.random.Generator, deterministic for the same
        (seed, keys).
    """
    entropy = [int(seed) & 0xFFFFFFFF]
    for key in keys:
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:4]
        entropy.append(int.from_bytes(digest, "big"))
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _build_generator(name: str, params: dict, variable: str | None = None):
    """Instantiate a generator by its registry name.

    Args:
        name: The generator's registered name (e.g. "seasonal_smooth").
        params: Keyword params to pass to the generator's constructor.
        variable: The variable name this generator is for, if any (used
            only to make the error message specific); None means it's for
            the population.

    Returns:
        The instantiated generator, ready for .generate(...).

    Errors Caught (raised to caller):
        ValueError: If a param is unexpected (turns the raw TypeError into a
            message naming the variable, generator, and bad param).
    """
    try:
        return get_generator(name)(**params)
    except TypeError as exc:
        where = f"variable '{variable}'" if variable else "population"
        raise ValueError(
            f"{where}: generator '{name}' got an invalid param ({exc})."
        ) from exc


def _inject_start_period(params: dict, start_period: "str | None") -> None:
    """Align a from_csv source to the scenario calendar, unless it set its own.

    Args:
        params: The from_csv generator's params dict (mutated in place).
        start_period: The scenario's start_period, or None.
    """
    if start_period is not None and "start_period" not in params:
        params["start_period"] = start_period
