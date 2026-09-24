"""Step 2 of the pipeline: validate the raw dict into a typed ScenarioConfig.

Validation is two-tier:

- **Hard errors** (impossible scenarios) raise during ``parse_config`` and
  stop the run before anything is generated. Most checks come free from
  Pydantic; the cross-field checks live in validators below.
- **Warnings** (suspicious but legal scenarios) come from the separate
  ``validate_scenario`` function, which never raises. The CLI prints them
  and proceeds.

A scenario is ONE list of series. A series is a count series iff it carries a
``counts:`` block, and any series may ``depends_on`` any other, so the list is
a DAG that ``series_order`` topologically sorts. Cycles are rejected.

Generator-specific parameters are deliberately NOT modelled here: the schema
validates each series' envelope (name / generate / params) and each
generator validates its own params, so this file does not grow when
generators are added. Two pragmatic name-based exceptions look inside params:
the lag-adding transforms (``_transform_lag``) and the from_csv
multi-location warning in ``validate_scenario``.
"""
import math
from collections import Counter
from graphlib import CycleError, TopologicalSorter
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsl.core.pipeline.periods import format_period, parse_period, periods_per_year

# Every model: extra="forbid" rejects typo'd keys; allow_inf_nan=False rejects
# NaN/Inf in float fields (a non-finite weight would corrupt the output).
_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)


# Classes first (data-driven module exception): they are the main point of
# this file, so they come before the helper/public functions that use them.
class PopulationSpec(BaseModel):
    """A generator that produces the ``population`` series over time.

    Same envelope as ``VariableSpec`` minus ``name``. Lets population grow
    instead of being a fixed scalar — e.g.
    ``{generate: linear_trend, params: {start: 70000, slope: 90}}``.
    """

    model_config = _STRICT

    generate: str
    params: dict = Field(default_factory=dict)


class LocationSpec(BaseModel):
    """Per-location overrides under the mapping form of ``locations:``.

    ``None`` means "use the count series' own ``counts.population``".
    A model (not a plain dict) so a typo'd override key is rejected.
    """

    model_config = _STRICT

    population: int | PopulationSpec | None = None


class AutoregressiveSpec(BaseModel):
    """AR(1) persistence on a series: ``x[t] = phi * x[t-1] + noise``.

    ``phi`` is the fraction of a period's value carried into the next, and
    is a declared ground truth a model can be scored on recovering.
    """

    model_config = _STRICT

    # lt=1.0: phi == 1 is a random walk (non-stationary) — it wanders without
    # bound and never returns, which is not what "persistence" should mean.
    phi: float = Field(ge=0.0, lt=1.0)
    noise: float = Field(default=0.2, ge=0.0)


class CountsSpec(BaseModel):
    """The ``counts:`` block — turns a series' float signal into integer counts.

    Its presence is what makes a series a disease signal; these are the
    incidence model's own parameters, so they live here rather than on the
    series, and cannot be written anywhere else.
    """

    model_config = _STRICT

    # A fixed headcount or a generator (growth). May be omitted only when
    # EVERY location sets its own (checked on ScenarioConfig).
    population: int | PopulationSpec | None = None
    max_rate: float = Field(default=0.3, gt=0.0, le=1.0)
    median_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    # "poisson" has variance == mean; "negative_binomial" adds overdispersion
    # (real surveillance counts are usually more variable than Poisson).
    distribution: Literal["poisson", "negative_binomial"] = "poisson"
    # variance = mean + mean²/overdispersion: SMALLER means MORE variable.
    overdispersion: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def _check_rates(self) -> "CountsSpec":
        """Cross-field checks a single Field() range can't express.

        Returns:
            self, unchanged, if both checks pass.

        Errors Caught (raised to caller):
            ValueError: If median_rate >= max_rate (the sigmoid shift is
                undefined outside (0, 1)), or population is a fixed int < 1.
        """
        # The model shifts its sigmoid by logit(median_rate / max_rate),
        # only defined for a ratio strictly between 0 and 1.
        if self.median_rate >= self.max_rate:
            raise ValueError(
                f"median_rate ({self.median_rate}) must be smaller than "
                f"max_rate ({self.max_rate})."
            )
        # A Field range can't sit on a union arm, so enforce it here.
        if isinstance(self.population, int) and self.population < 1:
            raise ValueError(f"population must be >= 1, got {self.population}.")
        return self


class EventSpec(BaseModel):
    """A named shock under ``events:`` — when it fires.

    Either a ``rate`` (each period independently, seeded) or explicit ``at``
    periods, never both: two sources of truth for the timing would be
    ambiguous. Events are regional — drawn once and shared by every location,
    so a storm hits the whole region in the same periods.
    """

    model_config = _STRICT

    rate: float | None = Field(default=None, ge=0.0, le=1.0)
    at: list[int] | None = None

    @model_validator(mode="after")
    def _check_timing(self) -> "EventSpec":
        """Check exactly one timing source is given, and that it is usable.

        Returns:
            self, unchanged, if the timing is well formed.

        Errors Caught (raised to caller):
            ValueError: If neither or both of rate/at are set, or any listed
                period is negative.
        """
        if (self.rate is None) == (self.at is None):
            raise ValueError(
                "an event needs exactly one of 'rate' (fires randomly) or "
                "'at' (fires at these periods)."
            )
        if self.at is not None:
            if not self.at:
                raise ValueError("'at' must list at least one period.")
            if any(period < 0 for period in self.at):
                raise ValueError(f"'at' periods must be >= 0, got {self.at}.")
        return self

    def periods(self, n_total: int, rng: np.random.Generator) -> np.ndarray:
        """Return the periods this event fires in.

        Args:
            n_total: Length of the series, so a rate draw covers every period.
            rng: Seeded generator, used only for a rate-driven event.

        Returns:
            A boolean mask of length n_total, True where the event fires.
            Example: array([False, False, True, False, ...]) for at=[2].
        """
        if self.at is not None:
            mask = np.zeros(n_total, dtype=bool)
            mask[self.at] = True
            return mask
        return rng.random(n_total) < self.rate


class EventEffectSpec(BaseModel):
    """What an event does to one series when it fires.

    ``multiplier`` scales the series (proportional: it grows with the series'
    own level and can never move a series sitting at zero); ``add`` shifts it
    by a fixed amount regardless of level. Both are signed, so one event can
    raise rainfall and lower temperature in the same period.
    """

    model_config = _STRICT

    multiplier: float | None = None
    add: float | None = None

    @model_validator(mode="after")
    def _check_effect(self) -> "EventEffectSpec":
        """Check exactly one of multiplier/add is given.

        Returns:
            self, unchanged, if exactly one effect is set.

        Errors Caught (raised to caller):
            ValueError: If neither or both are set.
        """
        if (self.multiplier is None) == (self.add is None):
            raise ValueError(
                "an event effect needs exactly one of 'multiplier' (scales "
                "the series) or 'add' (shifts it by a fixed amount)."
            )
        return self

    def apply(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply this effect wherever the event fires.

        Args:
            values: The series so far.
            mask: Boolean mask of the periods the event fires in.

        Returns:
            A copy of values with the effect applied at the masked periods.
        """
        out = values.copy()
        if self.multiplier is not None:
            out[mask] = out[mask] * self.multiplier
        else:
            out[mask] = out[mask] + self.add
        return out


class TransformSpec(BaseModel):
    """A registry transform applied to a driver — the generator envelope's
    twin: ``name`` is looked up in the transform registry, ``params`` is
    validated by the transform itself."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    params: dict = Field(default_factory=dict)


class DependencySpec(BaseModel):
    """One entry under ``depends_on:`` — a parent of this series.

    Any series may depend on any other, so this is how both a climate chain
    (rain -> dam) and a disease's drivers are expressed.
    """

    model_config = _STRICT

    series: str
    # ge=0: a negative lag would mean disease precedes its cause.
    lag: int = Field(default=0, ge=0)
    weight: float = 1.0
    # Applied after the causal lag, before standardize.
    transforms: list[TransformSpec] = Field(default_factory=list)


class SeriesSpec(BaseModel):
    """One entry under ``series:`` — a single time series the scenario builds.

    ``name`` becomes the output column; ``generate`` is looked up in the
    generator registry and its ``params`` validated by the generator itself.
    A series with ``depends_on`` may omit ``generate`` (its base is flat 0).
    A ``counts`` block makes it a disease signal rather than a float column.
    """

    model_config = _STRICT

    name: str = Field(min_length=1)
    # Optional: a series with parents defaults to a flat 0 base.
    generate: str | None = None
    params: dict = Field(default_factory=dict)
    depends_on: list[DependencySpec] = Field(default_factory=list)
    # Applies to any series: a broken gauge as readily as a reporting gap.
    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Persistence. Natural on a physical store (soil moisture, a dam), not
    # only on counts.
    autoregressive: AutoregressiveSpec | None = None
    # Named events this series reacts to, and how: {"storm": {"multiplier": 2}}.
    events: dict[str, EventEffectSpec] = Field(default_factory=dict)
    # Presence of this block is what makes the series a count series.
    counts: CountsSpec | None = None

    @model_validator(mode="after")
    def _check_series(self) -> "SeriesSpec":
        """Check the name is usable and the series has some source at all.

        Returns:
            self, unchanged, if both checks pass.

        Errors Caught (raised to caller):
            ValueError: If the name is blank, or the series has neither a
                generator nor any dependency to build it from.
        """
        if not self.name.strip():
            raise ValueError("series name must not be blank.")
        if self.generate is None and not self.depends_on:
            raise ValueError(
                f"series '{self.name}' has neither 'generate' nor "
                f"'depends_on'; it needs at least one source."
            )
        return self


class ScenarioConfig(BaseModel):
    """The whole scenario file, validated and typed."""

    model_config = _STRICT

    period: Literal["daily", "weekly", "monthly", "yearly"]
    n_total: int = Field(ge=1)
    # ge=0: numpy's default_rng requires a non-negative seed.
    seed: int = Field(default=0, ge=0)
    # None → only the single full CSV; a value in (0, 1) also writes
    # train.csv/test.csv as a row split.
    train_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    # Real-world period the series starts at (e.g. "2010-07"). None means
    # the first period of the year 2000.
    start_period: str | None = None
    # Accepts a plain list (names only) or a mapping (names with overrides);
    # _normalize_locations turns the mapping into this list + the overrides.
    locations: list[str] = Field(default=["loc"], min_length=1)
    # Filled from the mapping form; excluded from dumps so metadata
    # round-trips (metadata.py rebuilds the mapping form).
    location_overrides: dict[str, LocationSpec] = Field(
        default_factory=dict, exclude=True
    )
    # Named shocks a series can opt into; regional, so every location is
    # struck in the same periods.
    events: dict[str, EventSpec] = Field(default_factory=dict)
    # One list for everything: climate drivers and disease signals alike.
    # min_length=1: a scenario with no series has nothing to generate.
    series: list[SeriesSpec] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_locations(cls, data: object) -> object:
        """Accept either a name list or a name -> overrides mapping.

        Args:
            data: The raw dict from YAML (before Pydantic parses fields),
                or a non-dict which is passed through unchanged.

        Returns:
            data, with a mapping-form "locations" rewritten to a plain name
            list plus a new "location_overrides" key holding the mapping.

        Errors Caught (raised to caller):
            ValueError: If "locations" is an empty mapping.
        """
        if not isinstance(data, dict):
            return data
        locations = data.get("locations")
        if isinstance(locations, dict):
            if not locations:
                raise ValueError("locations mapping must not be empty.")
            # dict preserves insertion order → location order is YAML order.
            data["locations"] = list(locations.keys())
            data["location_overrides"] = locations
        return data

    @model_validator(mode="after")
    def _check_cross_section(self) -> "ScenarioConfig":
        """Hard errors that span multiple fields, one helper per concern.

        Returns:
            self, unchanged, if every check passes.

        Errors Caught (raised to caller):
            ValueError: From any of the _check_* helpers below.
        """
        self._check_start_period()
        self._check_locations()
        self._check_train_fraction()
        defined = [spec.name for spec in self.series]
        self._check_series_names(defined)
        self._check_dependencies(defined)
        self._check_acyclic()
        self._check_events()
        return self

    def _check_start_period(self) -> None:
        """Check start_period is valid for the resolution and fits the calendar.

        The whole n_total-period range must land within the 4-digit
        calendar (year <= 9999).

        Errors Caught (raised to caller):
            ValueError: If start_period doesn't parse for this period type,
                or the period range runs past year 9999.
        """
        if self.start_period is None:
            return
        try:
            start_year, offset = parse_period(self.start_period, self.period)
        except ValueError as exc:
            raise ValueError(f"start_period: {exc}") from exc
        try:
            last = format_period(offset + self.n_total - 1, self.period, start_year)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"the period range starting {self.start_period} for "
                f"{self.n_total} periods runs past the supported calendar "
                f"(year 9999): {exc}"
            ) from exc
        # A label longer than its normal width means a 5-digit year slipped in.
        normal_widths = {"daily": 8, "weekly": 8, "monthly": 7, "yearly": 4}
        if len(str(last)) > normal_widths[self.period]:
            raise ValueError(
                f"the period range starting {self.start_period} for "
                f"{self.n_total} periods ends at '{last}', past year 9999."
            )

    def _check_locations(self) -> None:
        """Check location names are usable and every location has a population.

        Errors Caught (raised to caller):
            ValueError: If locations has duplicates, a blank name, the
                an override population < 1, or a location with no
                population source at all (neither its own override nor its
                counts block's population fallback).
        """
        if len(set(self.locations)) != len(self.locations):
            raise ValueError(
                f"locations contains duplicate names: {self.locations}."
            )
        if any(not loc.strip() for loc in self.locations):
            raise ValueError("location names must not be blank.")
        # A Field range can't sit on a union arm, so enforce it here.
        for name, override in self.location_overrides.items():
            if isinstance(override.population, int) and override.population < 1:
                raise ValueError(
                    f"location '{name}' population must be >= 1, "
                    f"got {override.population}."
                )
        # A count series' counts.population is the fallback; it may be
        # omitted ONLY when every location sets its own.
        uncovered = [
            loc
            for loc in self.locations
            if self.location_overrides.get(loc) is None
            or self.location_overrides[loc].population is None
        ]
        if not uncovered:
            return
        for spec in self.series:
            if spec.counts is not None and spec.counts.population is None:
                raise ValueError(
                    f"series '{spec.name}' needs counts.population because "
                    f"these locations do not set their own: {uncovered}."
                )

    def _check_train_fraction(self) -> None:
        """Check both the train and test partitions end up non-empty.

        Errors Caught (raised to caller):
            ValueError: If train_fraction with n_total gives an empty train
                or test split.
        """
        if self.train_fraction is None:
            return
        n_train = math.floor(self.n_total * self.train_fraction)
        if n_train < 1 or self.n_total - n_train < 1:
            raise ValueError(
                f"train_fraction {self.train_fraction} with n_total "
                f"{self.n_total} gives an empty train or test split "
                f"(train={n_train}, test={self.n_total - n_train})."
            )

    def _check_series_names(self, defined: list[str]) -> None:
        """Check series names won't clash with each other or built-in columns.

        Series names become output columns: a clash with a built-in column
        would silently overwrite it, and a duplicate name would silently
        drop one.

        Args:
            defined: The declared series names, in YAML order.

        Errors Caught (raised to caller):
            ValueError: If a name is reserved (time_period, location,
                population) or duplicated.
        """
        # "disease_cases" is deliberately absent: it is an ordinary series
        # name, and the conventional one for a count series.
        reserved = {"time_period", "location", "population"}
        clashes = sorted(reserved.intersection(defined))
        if clashes:
            raise ValueError(
                f"series names may not be reserved column names: {clashes}. "
                f"Reserved: {sorted(reserved)}."
            )
        duplicates = sorted(n for n, c in Counter(defined).items() if c > 1)
        if duplicates:
            raise ValueError(
                f"series contains duplicate names: {duplicates}."
            )

    def _check_dependencies(self, defined: list[str]) -> None:
        """Check every dependency is resolvable and its lag fits the series.

        Every dependency must name a declared series, with a lag (including
        lag added by its transforms) small enough that some non-warm-up data
        can actually appear.

        Args:
            defined: The declared series names, in YAML order.

        Errors Caught (raised to caller):
            ValueError: If a dependency names an undeclared series, or its
                lag (with transform warm-up) reaches or exceeds n_total.
        """
        for spec in self.series:
            for dep in spec.depends_on:
                if dep.series not in defined:
                    raise ValueError(
                        f"series '{spec.name}' depends on '{dep.series}', "
                        f"which is not a defined series. Defined series: "
                        f"{defined}."
                    )
                if dep.lag >= self.n_total:
                    raise ValueError(
                        f"series '{spec.name}' depends on '{dep.series}' "
                        f"with lag {dep.lag}, but n_total is {self.n_total}; "
                        f"the lag must be smaller than the series length for "
                        f"the relationship to appear."
                    )
                warmup = dep.lag + _transform_lag(dep.transforms)
                if warmup >= self.n_total:
                    raise ValueError(
                        f"series '{spec.name}' depends on '{dep.series}', "
                        f"which blanks {warmup} warm-up periods (lag "
                        f"{dep.lag} plus lag added by its transforms), but "
                        f"n_total is {self.n_total}; every value would be NaN."
                    )

    def _check_events(self) -> None:
        """Check every referenced event exists and fits inside the series.

        Errors Caught (raised to caller):
            ValueError: If a series reacts to an undeclared event, or an
                event's explicit periods run past n_total.
        """
        for name, event in self.events.items():
            if event.at is None:
                continue
            beyond = [period for period in event.at if period >= self.n_total]
            if beyond:
                raise ValueError(
                    f"event '{name}' fires at periods {beyond}, but n_total "
                    f"is {self.n_total}; every period must be < n_total."
                )
        for spec in self.series:
            unknown = sorted(set(spec.events) - set(self.events))
            if unknown:
                raise ValueError(
                    f"series '{spec.name}' reacts to undeclared events: "
                    f"{unknown}. Declared events: {sorted(self.events)}."
                )

    def _check_acyclic(self) -> None:
        """Reject a dependency loop, naming the series involved.

        The check is the topological sort itself, so a cycle of ANY length
        is caught — a direct A<->B pair, a three-series loop, or a long
        chain that bites its own tail. A lag >= 1 loop is still a cycle:
        resolving one would need a per-timestep engine, so v1 rejects it and
        points at shared events, which cover same-period co-movement.

        Errors Caught (raised to caller):
            ValueError: If the dependency graph contains a cycle.
        """
        try:
            self.series_order()
        except CycleError as exc:
            # CycleError.args[1] is the cycle path, first node repeated last.
            cycle = " -> ".join(exc.args[1])
            raise ValueError(
                f"series dependencies must not form a cycle: {cycle}. "
                f"Give the relationship one direction, or use a shared event "
                f"if these series should move together in the same period."
            ) from exc

    def series_order(self) -> list[str]:
        """Return series names in dependency order: parents before children.

        The order the engine must generate in, so every parent's values
        exist by the time a child needs them. Declaration order is the
        author's; this one is the graph's.

        Returns:
            Every series name, each appearing after all of its parents.
            Example: ["wind", "temp", "rain"] for rain <- temp <- wind.

        Errors Caught (raised to caller):
            CycleError: If the graph has a cycle. Callers outside this class
                should rely on _check_acyclic having run at parse time.
        """
        graph = {
            spec.name: {dep.series for dep in spec.depends_on}
            for spec in self.series
        }
        return list(TopologicalSorter(graph).static_order())

    def population_for(
        self, location: str, spec: "SeriesSpec"
    ) -> "int | PopulationSpec":
        """Resolve the population source for a count series at a location.

        The single place the engine asks "what is the population here?".

        Args:
            location: The location name.
            spec: The count series asking (its counts block holds the
                scenario-wide fallback).

        Returns:
            The location's own override if the mapping form set one, else
            the series' counts.population.
        """
        override = self.location_overrides.get(location)
        if override is not None and override.population is not None:
            return override.population
        return spec.counts.population


# Helper functions after the classes.
def _transform_lag(transforms: "list[TransformSpec]") -> int:
    """Sum the extra warm-up periods a dependency's transforms add.

    ponytail: knows 'lag' and 'distributed_lag' by name; grow a
    Transform.warmup API if a third lag-adding transform appears.

    Args:
        transforms: The dependency's transforms list, in application order.

    Returns:
        Total extra warm-up periods, on top of the dependency's own lag.
        Example: 2 for [{name: distributed_lag, params: {weights: [.5,.3,.2]}}].
    """
    extra = 0
    for tf in transforms:
        if tf.name == "lag":
            extra += int(tf.params.get("n", 0) or 0)
        elif tf.name == "distributed_lag":
            weights = tf.params.get("weights") or []
            extra += max(len(weights) - 1, 0)
    return extra


# Public functions last.
def parse_config(data: dict) -> ScenarioConfig:
    """Validate a raw scenario dict into a ScenarioConfig.

    Args:
        data: The raw dict from YAML.

    Returns:
        A validated ScenarioConfig object.

    Errors Caught (raised to caller):
        ValidationError: If any field fails validation, with field-specific
            messages.
    """
    return ScenarioConfig(**data)


def validate_scenario(config: ScenarioConfig) -> list[str]:
    """Soft validation: return warnings for suspicious-but-legal scenarios.

    These warnings do not stop execution — the CLI prints them and proceeds.

    Args:
        config: A validated ScenarioConfig.

    Returns:
        A list of warning messages (empty if no issues found).
        Example: ["series 'rainfall' is declared but nothing depends on it "
        "(decoy/confounder, or a mistake?)"]
    """
    warnings: list[str] = []

    # Orphan series may be an intentional decoy/confounder → warning only.
    used = {dep.series for spec in config.series for dep in spec.depends_on}
    for spec in config.series:
        # A count series is an output in its own right, so it is never orphaned.
        if spec.name not in used and spec.counts is None:
            warnings.append(
                f"series '{spec.name}' is declared but nothing depends on it "
                f"(decoy/confounder, or a mistake?)"
            )

    reacted_to = {name for spec in config.series for name in spec.events}
    for name in config.events:
        if name not in reacted_to:
            warnings.append(
                f"event '{name}' is declared but no series reacts to it; it "
                f"will have no effect on the output."
            )

    if not any(spec.counts is not None for spec in config.series):
        warnings.append(
            "no series has a 'counts' block, so the dataset has no disease "
            "signal — only climate columns."
        )

    for spec in config.series:
        if spec.missing_rate >= 0.5:
            warnings.append(
                f"series '{spec.name}' has missing_rate {spec.missing_rate}; "
                f"half or more of its values will be NaN."
            )

    if config.train_fraction is not None and config.train_fraction >= 0.95:
        warnings.append(
            f"train_fraction is {config.train_fraction}; the test split will "
            f"contain very few rows."
        )

    cycle = periods_per_year(config.period)
    if config.n_total < cycle:
        warnings.append(
            f"n_total ({config.n_total}) is shorter than one seasonal cycle "
            f"({cycle} {config.period} periods); seasonality will not be visible."
        )

    # Known limitation: start_period only relabels the output; seasonal
    # generators still begin their cycle at index 0, so a mid-year start has
    # the wrong seasonal phase.
    if config.start_period is not None:
        _, offset = parse_period(config.start_period, config.period)
        if offset != 0:
            warnings.append(
                f"start_period '{config.start_period}' begins mid-cycle; "
                f"seasonal generators and the disease baseline still start "
                f"their seasonal phase at the cycle start, so seasonality is "
                f"not aligned to the calendar."
            )

    # If the largest lag covers the whole training split, every training
    # target is warm-up NaN — nothing to learn from.
    deps = [dep for spec in config.series for dep in spec.depends_on]
    if config.train_fraction is not None and deps:
        n_train = math.floor(config.n_total * config.train_fraction)
        max_lag = max(d.lag + _transform_lag(d.transforms) for d in deps)
        if max_lag >= n_train:
            warnings.append(
                f"max dependency lag ({max_lag}) covers the whole training "
                f"split ({n_train} periods); train.csv will have no observed "
                f"values (all warm-up NaN)."
            )

    # A fixed source_location feeds ONE real series to every output location
    # — a likely surprise with several locations.
    if len(config.locations) > 1:
        for var in config.series:
            if var.generate == "from_csv" and var.params.get("source_location"):
                warnings.append(
                    f"series '{var.name}' uses from_csv with a fixed "
                    f"source_location '{var.params['source_location']}', but the "
                    f"scenario has {len(config.locations)} locations; every "
                    f"location will get the same real series."
                )

    return warnings
