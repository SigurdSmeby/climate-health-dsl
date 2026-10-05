"""The orchestrator: translate a validated ScenarioConfig into a DataFrame.

Series are generated parents-first, in the dependency order the schema's
topological sort provides, so a series can be built from others as well as
from a generator. The engine never hard-codes which generators exist: it
looks each series' ``generate:`` string up in the registry. Column names come
from the series names in the YAML.
"""
import hashlib

import numpy as np
import pandas as pd

# Importing the extension packages triggers auto-discovery: every generator,
# transform and emitter module registers itself on import.
import dsl.emitters
import dsl.generators
import dsl.transforms  # noqa: F401
from dsl.core.config.schema import PopulationSpec, ScenarioConfig, SeriesSpec
from dsl.core.extension.emitter_base import get_emitter
from dsl.core.extension.generator_base import get_generator
from dsl.core.pipeline.periods import format_period, parse_period
from dsl.core.pipeline.signal import apply_missing, build_signal


def run(config: ScenarioConfig) -> pd.DataFrame:
    """Run the whole simulation described by ``config``.

    For each location: build every series in dependency order and assemble
    the results into a tidy, long-format DataFrame.

    Args:
        config: A validated ScenarioConfig object.

    Returns:
        A DataFrame with n_total rows per location and columns:
        [time_period, location, <series columns>, population]. Series appear
        in declaration order, whatever order they were generated in.
        Example shape: (108 rows, 5 cols) for 3 locations x 36 months.

    Errors Caught (raised to caller):
        ValueError: If generator instantiation fails or population is invalid.
        KeyError: If a generator name is not registered.
    """
    # Regional: drawn once, so every location is struck in the same periods.
    event_masks = {
        name: event.periods(
            config.n_total, _child_rng(config.seed, "event", name)
        )
        for name, event in config.events.items()
    }
    frames = [
        _run_one_location(config, location, event_masks)
        for location in config.locations
    ]
    return pd.concat(frames, ignore_index=True)


def series_events(config: ScenarioConfig) -> list[dict]:
    """Collect every feature the scenario deliberately planted.

    Two sources, both exact rather than detected: each generator reports the
    peaks or shocks it placed, and each declared event reports the periods it
    fires in. Anything that needs to count features — how many spikes fall in
    a given fold, say — reads this instead of guessing from the output.

    Runs the generators again on their own seeded streams, so the numbers
    match a run of the same config without consuming or disturbing it.

    Args:
        config: A validated ScenarioConfig.

    Returns:
        One dict per feature: the series and location it belongs to, plus the
        reporter's own fields (kind, index, and magnitude/duration where
        meaningful). Empty if nothing identifiable was planted.
        Example: [{"series": "rainfall", "location": "north",
        "kind": "seasonal_spike", "index": 26, "magnitude": 20.0}]

    Errors Caught (raised to caller):
        ValueError: If a generator param is invalid.
        KeyError: If a generator name is not registered.
    """
    found: list[dict] = []

    for location in config.locations:
        for spec in config.series:
            if spec.generate is None:
                continue
            generator = _series_generator(config, spec, location)
            generator.generate(
                config.n_total,
                config.period,
                _child_rng(config.seed, location, "series", spec.name),
            )
            for event in generator.events():
                found.append(
                    {"series": spec.name, "location": location, **event}
                )

    # Scenario events are regional: one draw, so they are reported once per
    # series that actually reacts to them.
    for name, event in config.events.items():
        mask = event.periods(
            config.n_total, _child_rng(config.seed, "event", name)
        )
        for spec in config.series:
            if name not in spec.events:
                continue
            for index in np.flatnonzero(mask):
                found.append(
                    {
                        "series": spec.name,
                        "location": None,  # regional: every location at once
                        "kind": name,
                        "index": int(index),
                    }
                )
    return found


def _run_one_location(
    config: ScenarioConfig,
    location: str,
    event_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build every series for a single location and assemble its frame.

    Args:
        config: The validated scenario configuration.
        location: The location identifier (e.g., "north", "south").
        event_masks: Which periods each declared event fires in, shared by
            every location.

    Returns:
        A DataFrame with columns [time_period, location, <series>,
        population], one row per time period.
        Example: 36 rows for n_total=36.

    Errors Caught (raised to caller):
        ValueError: If a generator param is invalid or population is
            non-finite.
        KeyError: If a generator or transform name is not registered.
    """
    by_name = {spec.name: spec for spec in config.series}
    values: dict[str, np.ndarray] = {}
    population: np.ndarray | None = None

    # Parents before children, so every dependency is ready when needed.
    for name in config.series_order():
        spec = by_name[name]
        series, series_population = _build_series(
            config, spec, location, values, event_masks
        )
        values[name] = series
        # The population column shows the first count series' headcount;
        # with several diseases they share the location's population.
        if series_population is not None and population is None:
            population = series_population

    if config.start_period is not None:
        start_year, offset = parse_period(config.start_period, config.period)
    else:
        start_year, offset = 2000, 0

    # Tidy frame: the period label first, then location, the series in
    # DECLARATION order (not generation order), and population.
    columns: dict[str, object] = {
        "time_period": [
            format_period(i + offset, config.period, start_year)
            for i in range(config.n_total)
        ],
        "location": location,  # scalar -> broadcast to a constant column
    }
    for spec in config.series:
        columns[spec.name] = values[spec.name]
    if population is not None:
        columns["population"] = population
    return pd.DataFrame(columns)


def _build_series(
    config: ScenarioConfig,
    spec: SeriesSpec,
    location: str,
    built: dict[str, np.ndarray],
    event_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, "np.ndarray | None"]:
    """Build one series: base, parents, events, persistence, counts, gaps.

    Args:
        config: The validated scenario configuration.
        spec: The series to build.
        location: The location identifier.
        built: Already-built series for this location, keyed by name.
        event_masks: Which periods each declared event fires in.

    Returns:
        A (values, population) pair. population is the resolved headcount
        array for a count series, else None.

    Errors Caught (raised to caller):
        ValueError: If a generator param is invalid.
        KeyError: If a generator or transform name is not registered.
    """
    if spec.generate is None:
        # A dependency-only series starts from nothing; its parents are
        # what give it shape.
        base = np.zeros(config.n_total)
    else:
        base = _generate_base(config, spec, location)

    signal = build_signal(
        base, spec, built, _child_rng(config.seed, location, "signal", spec.name)
    )

    # Before anything downstream reads this series, so a child sees the
    # post-event values and a count draw responds to the shock.
    for name, effect in spec.events.items():
        # Its own stream, keyed by series and event, so a range-valued
        # strength is reproducible and adding an event elsewhere does not
        # shift this one's draws.
        signal = effect.apply(
            signal,
            event_masks[name],
            _child_rng(config.seed, location, "event-strength", spec.name, name),
        )

    population = None
    if spec.counts is not None:
        population = _resolve_population(
            config.population_for(location),
            config.n_total,
            config.period,
            _child_rng(config.seed, location, "population", spec.name),
            config.start_period,
        )
        # The block name selects the emitter; its params come from the block.
        emitter = get_emitter("counts")(
            population=population,
            **spec.counts.model_dump(exclude={"population"}),
        )
        signal = emitter.emit(
            signal, _child_rng(config.seed, location, "counts", spec.name)
        )

    signal = apply_missing(
        signal,
        spec.missing_rate,
        _child_rng(config.seed, location, "missing", spec.name),
    )
    return signal, population


def _generate_base(
    config: ScenarioConfig, spec: SeriesSpec, location: str
) -> np.ndarray:
    """Run a series' own generator for one location.

    Args:
        config: The validated scenario configuration.
        spec: The series being built (name, generate, params).
        location: The location identifier (e.g., "north", "south").

    Returns:
        A numpy array of config.n_total float values, before any parent
        contributions are added.
        Example: array([45.2, 54.1, 63.8, ...]).

    Errors Caught (raised to caller):
        ValueError: If the generator rejects a param, or a from_csv series
            can't be matched to this output location.
    """
    generator = _series_generator(config, spec, location)
    return generator.generate(
        config.n_total,
        config.period,
        _child_rng(config.seed, location, "series", spec.name),
    )


def _series_generator(config: ScenarioConfig, spec: SeriesSpec, location: str):
    """Build a series' generator, resolved for one location.

    Shared by generation and by event collection, so the two cannot drift:
    a from_csv series needs its per-location source resolved either way.

    Args:
        config: The validated scenario configuration.
        spec: The series whose generator to build.
        location: The location identifier.

    Returns:
        The instantiated generator, ready for .generate(...).

    Errors Caught (raised to caller):
        ValueError: If the generator rejects a param, or a from_csv series
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
                        f"from_csv: series '{spec.name}' has no "
                        f"source_location and output location '{location}' "
                        f"is not in {params.get('file')} (available: "
                        f"{csv_locations}); set source_location or rename "
                        f"the location to match."
                    )
                params["source_location"] = location
    return _build_generator(spec.generate, params, spec.name)


def _resolve_population(
    source: "int | PopulationSpec",
    n_periods: int,
    period: str,
    rng: np.random.Generator,
    start_period: "str | None" = None,
) -> np.ndarray:
    """Turn a population source into a length-n_periods integer array.

    If source is a fixed int, return a constant array (draws NO randomness).
    If source is a PopulationSpec, run it through the generator registry
    like any other series, then round to non-negative whole people.

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
    headcount = _build_generator(source.generate, params).generate(
        n_periods, period, rng
    )
    if not np.all(np.isfinite(headcount)):
        raise ValueError(
            f"population generator '{source.generate}' produced a missing or "
            f"non-finite value; population must be finite at every period."
        )
    return np.maximum(np.round(headcount), 0).astype(int)


def _child_rng(seed: int, *keys: str) -> np.random.Generator:
    """Create a reproducible Generator derived from seed plus component keys.

    Each component (a series, a location's population, a count draw) gets
    its OWN stream keyed by stable strings, so reordering series/locations
    or adding a decoy cannot shift another component's draws.
    Keys are hashed with sha256 (Python's hash() is salted per-process, so
    it can't be used directly for a reproducible seed).

    Args:
        seed: The scenario's base seed.
        *keys: Component keys identifying this stream (e.g. location,
            "series", series name).

    Returns:
        A seeded np.random.Generator, deterministic for the same
        (seed, keys).
    """
    entropy = [int(seed) & 0xFFFFFFFF]
    for key in keys:
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:4]
        entropy.append(int.from_bytes(digest, "big"))
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _build_generator(name: str, params: dict, series: str | None = None):
    """Instantiate a generator by its registry name.

    Args:
        name: The generator's registered name (e.g. "seasonal_smooth").
        params: Keyword params to pass to the generator's constructor.
        series: The series name this generator is for, if any (used only to
            make the error message specific); None means it's for the
            population.

    Returns:
        The instantiated generator, ready for .generate(...).

    Errors Caught (raised to caller):
        ValueError: If a param is unexpected (turns the raw TypeError into a
            message naming the series, generator, and bad param).
    """
    try:
        return get_generator(name)(**params)
    except TypeError as exc:
        where = f"series '{series}'" if series else "population"
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
