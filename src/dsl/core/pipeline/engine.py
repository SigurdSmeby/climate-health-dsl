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


def _child_rng(seed: int, *keys: str) -> np.random.Generator:
    """A reproducible Generator derived from ``seed`` plus a component key.

    Each component (a variable, a location's population, a disease draw)
    gets its OWN stream keyed by stable strings, so reordering variables/
    locations or adding a decoy cannot shift another component's draws.
    Keys are hashed with sha256 (Python's hash() is salted per-process).
    """
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


def _inject_start_period(params: dict, start_period: "str | None") -> None:
    """Align a from_csv source to the scenario calendar, unless it set its own."""
    if start_period is not None and "start_period" not in params:
        params["start_period"] = start_period


def _resolve_population(
    source: "int | PopulationSpec",
    n_periods: int,
    period: str,
    rng: np.random.Generator,
    start_period: "str | None" = None,
) -> np.ndarray:
    """Turn a population source into a length-``n_periods`` integer array.

    A plain int becomes a constant array and draws NO randomness (scalar
    scenarios stay byte-identical). A PopulationSpec runs through the
    generator registry like a covariate, then is rounded to non-negative
    whole people.
    """
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


def run(config: ScenarioConfig) -> pd.DataFrame:
    """Run the whole simulation described by ``config``.

    Returns a tidy, long-format DataFrame with ``n_total`` rows per location
    and columns: ``time_period``, ``location``, one column per YAML variable
    (in declaration order), ``disease_cases``, and ``population``.
    """
    shared_cache: dict[str, np.ndarray] = {}
    location_frames = [
        _run_one_location(config, location, shared_cache)
        for location in config.locations
    ]
    return pd.concat(location_frames, ignore_index=True)


def _generate_variable(
    config: ScenarioConfig,
    spec,
    location: str,
    shared_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Generate one variable's series for one location."""
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

    if spec.shared:
        # Latent regional driver: a second component from a location-
        # INDEPENDENT stream, mixed in by `shared`. The sqrt weights keep
        # total variance ~constant. shared=1 → every location identical;
        # shared=0 never reaches here (byte-identical to the plain path).
        shared_series = shared_cache.get(spec.name)
        if shared_series is None:
            shared_rng = _child_rng(config.seed, "shared", "variable", spec.name)
            if spec.generate == "from_csv":
                # `generator` above is bound to THIS location's own
                # source_location (auto-matched per-location when the
                # variable set none) — reusing it would make the "shared"
                # component just this location's own data again. A shared
                # series needs its OWN location-independent source: use the
                # variable's explicit source_location if it set one, else
                # this is ambiguous on a multi-location file.
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
        own = np.sqrt(1.0 - s) * own + np.sqrt(s) * shared_series
    return own


def _run_one_location(
    config: ScenarioConfig,
    location: str,
    shared_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Generate the full series for a single named location."""
    drivers = {
        spec.name: _generate_variable(config, spec, location, shared_cache)
        for spec in config.variables
    }

    # This location's population (its override or the default), threaded into
    # the disease spec so the incidence model and cap use it.
    population = _resolve_population(
        config.population_for(location),
        config.n_total,
        config.period,
        _child_rng(config.seed, location, "population"),
        config.start_period,
    )
    disease_spec = config.disease_cases.model_copy(update={"population": population})

    disease = build_disease_cases(
        drivers,
        disease_spec,
        _child_rng(config.seed, location, "disease"),
        config.n_total,
        config.period,
    )

    # Row 0 is start_period if set, else the first period of the year 2000.
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
        "location": location,  # scalar → broadcast to a constant column
    }
    columns.update(drivers)
    columns["disease_cases"] = disease
    columns["population"] = population
    return pd.DataFrame(columns)
