"""Step 2 of the pipeline: validate the raw dict into a typed ScenarioConfig.

Validation is two-tier:

- **Hard errors** (impossible scenarios) raise during ``parse_config`` and
  stop the run before anything is generated. Most checks come free from
  Pydantic; the cross-field checks live in validators below.
- **Warnings** (suspicious but legal scenarios) come from the separate
  ``validate_scenario`` function, which never raises. The CLI prints them
  and proceeds.

Generator-specific parameters are deliberately NOT modelled here: the schema
validates each variable's envelope (name / generate / params) and each
generator validates its own params, so this file does not grow when
generators are added. Two pragmatic name-based exceptions look inside params:
the lag-adding transforms (``_transform_lag``) and the from_csv
multi-location warning in ``validate_scenario``.
"""
import math
from collections import Counter
from typing import Literal

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

    ``None`` means "use the scenario's top-level ``disease_cases.population``".
    A model (not a plain dict) so a typo'd override key is rejected.
    """

    model_config = _STRICT

    population: int | PopulationSpec | None = None


class VariableSpec(BaseModel):
    """One entry under ``variables:`` — a variable the scenario generates.

    ``name`` becomes the output column; ``generate`` is looked up in the
    generator registry; ``params`` is passed straight to that generator,
    which validates it itself.
    """

    model_config = _STRICT

    name: str = Field(min_length=1)
    generate: str
    params: dict = Field(default_factory=dict)
    # Fraction of this variable's signal shared across all locations (a
    # latent regional driver): 0/None independent, 1 identical.
    shared: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _name_not_blank(self) -> "VariableSpec":
        """Reject a name that is empty or only whitespace.

        Returns:
            self, unchanged, if the name is non-blank.

        Errors Caught (raised to caller):
            ValueError: If name.strip() is empty.
        """
        if not self.name.strip():
            raise ValueError("variable name must not be blank.")
        return self


class TransformSpec(BaseModel):
    """A registry transform applied to a driver — the generator envelope's
    twin: ``name`` is looked up in the transform registry, ``params`` is
    validated by the transform itself."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    params: dict = Field(default_factory=dict)


class DependencySpec(BaseModel):
    """One entry under ``depends_on:`` — a driver of the disease signal."""

    model_config = _STRICT

    variable: str
    # ge=0: a negative lag would mean disease precedes its cause.
    lag: int = Field(default=0, ge=0)
    weight: float = 1.0
    # Applied after the causal lag, before standardize.
    transforms: list[TransformSpec] = Field(default_factory=list)


class DiseaseSpec(BaseModel):
    """The ``disease_cases:`` section — how the dependent signal is built."""

    model_config = _STRICT

    depends_on: list[DependencySpec]
    # A fixed headcount or a generator (growth). May be omitted only when
    # EVERY location sets its own (checked on ScenarioConfig).
    population: int | PopulationSpec | None = None
    autoregressive: bool = False
    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Incidence-model knobs.
    max_rate: float = Field(default=0.3, gt=0.0, le=1.0)
    median_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    # "poisson" has variance == mean; "negative_binomial" adds overdispersion
    # (real surveillance counts are usually more variable than Poisson).
    count_distribution: Literal["poisson", "negative_binomial"] = "poisson"
    # variance = mean + mean²/overdispersion: SMALLER means MORE variable.
    overdispersion: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def _check_rates(self) -> "DiseaseSpec":
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
    variables: list[VariableSpec]
    disease_cases: DiseaseSpec

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
        defined = [v.name for v in self.variables]
        self._check_variables(defined)
        self._check_dependencies(defined)
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
                reserved name "shared", an override population < 1, or a
                location with no population source at all (neither its own
                override nor the disease_cases.population fallback).
        """
        if len(set(self.locations)) != len(self.locations):
            raise ValueError(
                f"locations contains duplicate names: {self.locations}."
            )
        if any(not loc.strip() for loc in self.locations):
            raise ValueError("location names must not be blank.")
        # "shared" is the internal RNG key for a variable's latent regional
        # driver (see engine._generate_variable); a location with this exact
        # name would draw the same stream as that driver, silently breaking
        # `shared:` for it.
        if "shared" in self.locations:
            raise ValueError(
                "location name 'shared' is reserved (used internally for the "
                "shared-variable latent driver); choose a different name."
            )
        # A Field range can't sit on a union arm, so enforce it here.
        for name, override in self.location_overrides.items():
            if isinstance(override.population, int) and override.population < 1:
                raise ValueError(
                    f"location '{name}' population must be >= 1, "
                    f"got {override.population}."
                )
        # disease_cases.population is the fallback; it may be omitted ONLY
        # when every location sets its own.
        if self.disease_cases.population is None:
            uncovered = [
                loc
                for loc in self.locations
                if self.location_overrides.get(loc) is None
                or self.location_overrides[loc].population is None
            ]
            if uncovered:
                raise ValueError(
                    "disease_cases.population is required because these "
                    f"locations do not set their own: {uncovered}."
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

    def _check_variables(self, defined: list[str]) -> None:
        """Check variable names won't clash with each other or built-in columns.

        Variable names become output columns: a clash with a built-in
        column would silently overwrite it, and a duplicate variable name
        would silently drop one.

        Args:
            defined: The declared variable names, in YAML order.

        Errors Caught (raised to caller):
            ValueError: If a name is reserved (time_period, location,
                disease_cases, population) or duplicated.
        """
        reserved = {"time_period", "location", "disease_cases", "population"}
        clashes = sorted(reserved.intersection(defined))
        if clashes:
            raise ValueError(
                f"variable names may not be reserved column names: {clashes}. "
                f"Reserved: {sorted(reserved)}."
            )
        duplicates = sorted(n for n, c in Counter(defined).items() if c > 1)
        if duplicates:
            raise ValueError(
                f"variables contains duplicate names: {duplicates}."
            )

    def _check_dependencies(self, defined: list[str]) -> None:
        """Check every dependency is resolvable and its lag fits the series.

        Every dependency must name a declared variable, with a lag
        (including lag added by its transforms) small enough that some
        non-warm-up data can actually appear.

        Args:
            defined: The declared variable names, in YAML order.

        Errors Caught (raised to caller):
            ValueError: If a dependency names an undeclared variable, or its
                lag (with transform warm-up) reaches or exceeds n_total.
        """
        for dep in self.disease_cases.depends_on:
            if dep.variable not in defined:
                raise ValueError(
                    f"disease_cases depends on '{dep.variable}', which is not "
                    f"a defined variable. Defined variables: {defined}."
                )
            if dep.lag >= self.n_total:
                raise ValueError(
                    f"depends_on '{dep.variable}' has lag {dep.lag}, but "
                    f"n_total is {self.n_total}; the lag must be smaller than "
                    f"the series length for the relationship to appear."
                )
            warmup = dep.lag + _transform_lag(dep.transforms)
            if warmup >= self.n_total:
                raise ValueError(
                    f"depends_on '{dep.variable}' blanks {warmup} warm-up "
                    f"periods (lag {dep.lag} plus lag added by its "
                    f"transforms), but n_total is {self.n_total}; every "
                    f"disease_cases value would be NaN."
                )

    def population_for(self, location: str) -> "int | PopulationSpec":
        """Resolve the population source for a location.

        The single place the engine asks "what is the population here?".

        Args:
            location: The location name.

        Returns:
            The location's own override if the mapping form set one, else
            disease_cases.population (the scenario-wide fallback).
        """
        override = self.location_overrides.get(location)
        if override is not None and override.population is not None:
            return override.population
        return self.disease_cases.population


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
        Example: ["variable 'rainfall' is declared but no disease_cases "
        "dependency uses it (decoy/confounder, or a mistake?)"]
    """
    warnings: list[str] = []

    # Orphan variables may be an intentional decoy/confounder → warning only.
    used = {dep.variable for dep in config.disease_cases.depends_on}
    for var in config.variables:
        if var.name not in used:
            warnings.append(
                f"variable '{var.name}' is declared but no disease_cases "
                f"dependency uses it (decoy/confounder, or a mistake?)"
            )

    if config.disease_cases.missing_rate >= 0.5:
        warnings.append(
            f"missing_rate is {config.disease_cases.missing_rate}; half or "
            f"more of disease_cases will be NaN."
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
    if config.train_fraction is not None and config.disease_cases.depends_on:
        n_train = math.floor(config.n_total * config.train_fraction)
        max_lag = max(
            d.lag + _transform_lag(d.transforms)
            for d in config.disease_cases.depends_on
        )
        if max_lag >= n_train:
            warnings.append(
                f"max dependency lag ({max_lag}) covers the whole training "
                f"split ({n_train} periods); train.csv will have no observed "
                f"disease_cases (all warm-up NaN)."
            )

    # A fixed source_location feeds ONE real series to every output location
    # — a likely surprise with several locations.
    if len(config.locations) > 1:
        for var in config.variables:
            if var.generate == "from_csv" and var.params.get("source_location"):
                warnings.append(
                    f"variable '{var.name}' uses from_csv with a fixed "
                    f"source_location '{var.params['source_location']}', but the "
                    f"scenario has {len(config.locations)} locations; every "
                    f"location will get the same real series."
                )

    return warnings
