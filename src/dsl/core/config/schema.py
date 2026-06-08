"""Step 2 of the pipeline: validate the raw dict into a typed ScenarioConfig.

Validation is two-tier:

- **Hard errors** (impossible or certainly-wrong scenarios) raise during
  ``parse_config`` with a message naming the offending field, and the run
  stops before anything is generated. Most checks come free from Pydantic
  (types, ranges, unknown fields); two cross-section checks (referential
  integrity, lag sanity) live in a model validator below.
- **Warnings** (suspicious but legal scenarios) come from the separate
  ``validate_scenario`` function, which returns human-readable strings and
  never raises. The CLI prints them and proceeds.

This is the ONLY core file that may be edited after the initial build, and
only to add a genuinely new top-level concept. Generator-specific parameters
are deliberately NOT modelled here: the schema validates each variable's
envelope (name / generate / params) and each generator validates its own
params, so this file does not grow when generators are added.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsl.core.pipeline.periods import parse_period, periods_per_year


class PopulationSpec(BaseModel):
    """A generator that produces the ``population`` series over time.

    The same envelope as ``VariableSpec`` minus ``name`` (population isn't a
    named covariate). Lets population grow/change instead of being a fixed
    scalar — e.g. ``{generate: linear_trend, params: {start: 70000, slope: 90}}``.
    The engine runs it through the generator registry like any covariate and
    rounds the result to non-negative integer people.
    """

    model_config = ConfigDict(extra="forbid")

    generate: str
    params: dict = Field(default_factory=dict)


class LocationSpec(BaseModel):
    """Per-location overrides under the mapping form of ``locations:``.

    Currently only ``population`` can be overridden; ``None`` means "use the
    scenario's top-level ``disease_cases.population``". population may itself
    be a generator (a growth trajectory) just like the top-level one. The
    model exists so a typo'd override key is rejected, and so future
    per-location settings extend this one block rather than reshaping the YAML.
    """

    model_config = ConfigDict(extra="forbid")

    population: int | PopulationSpec | None = None


class VariableSpec(BaseModel):
    """One entry under ``variables:`` — a variable the scenario generates.

    ``name`` becomes the output column name, so CHAP-compatible scenarios
    name their variables ``rainfall`` and ``mean_temperature``. ``generate``
    is looked up in the generator registry; ``params`` is passed straight to
    that generator, which validates it itself.
    """

    # extra="forbid" makes Pydantic reject unknown keys (e.g. a typo like
    # "generete:") instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")

    name: str
    generate: str
    params: dict = Field(default_factory=dict)


class DependencySpec(BaseModel):
    """One entry under ``depends_on:`` — a driver of the disease signal."""

    model_config = ConfigDict(extra="forbid")

    variable: str
    # ge=0: a negative lag would mean disease precedes its cause.
    lag: int = Field(default=0, ge=0)
    weight: float = 1.0


class DiseaseSpec(BaseModel):
    """The ``disease_cases:`` section — how the dependent signal is built."""

    model_config = ConfigDict(extra="forbid")

    depends_on: list[DependencySpec]
    # A fixed headcount, or a generator that produces a population series over
    # time (growth). int is tried first, so a plain number stays an int.
    # Optional: may be omitted only when EVERY location sets its own
    # population (checked on ScenarioConfig, which can see the locations).
    population: int | PopulationSpec | None = None
    autoregressive: bool = False
    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Incidence-model knobs; defaults match the reference implementation.
    max_rate: float = Field(default=0.3, gt=0.0, le=1.0)
    median_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    # How case counts are drawn from the incidence rate. "poisson" (default)
    # has variance == mean; "negative_binomial" adds overdispersion (real
    # surveillance counts are usually more variable than Poisson allows).
    count_distribution: Literal["poisson", "negative_binomial"] = "poisson"
    # The negative-binomial dispersion parameter (only used when
    # count_distribution is "negative_binomial"): variance = mean + mean^2 /
    # overdispersion, so SMALLER means MORE variable; large → close to Poisson.
    overdispersion: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def _check_rates(self) -> "DiseaseSpec":
        """median_rate must stay below max_rate.

        The model shifts its sigmoid by logit(median_rate / max_rate), which
        is only defined for a ratio strictly between 0 and 1.
        """
        if self.median_rate >= self.max_rate:
            raise ValueError(
                f"median_rate ({self.median_rate}) must be smaller than "
                f"max_rate ({self.max_rate})."
            )
        # The int form must be a real headcount (the Field range can't sit on
        # a union arm, so it's enforced here).
        if isinstance(self.population, int) and self.population < 1:
            raise ValueError(f"population must be >= 1, got {self.population}.")
        return self


class ScenarioConfig(BaseModel):
    """The whole scenario file, validated and typed."""

    model_config = ConfigDict(extra="forbid")

    # Literal[...] restricts the value to exactly these strings; anything
    # else ("fortnightly") is a field-level validation error.
    period: Literal["daily", "weekly", "monthly", "yearly"]
    n_total: int = Field(ge=1)
    seed: int = 0
    # None means "write only the single full CSV"; a value in (0, 1) also
    # writes train.csv/test.csv as a row split.
    train_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    # The real-world period the series starts at (e.g. "2010-07" for a
    # monthly scenario). None means the first period of the year 2000.
    start_period: str | None = None
    # CHAP datasets carry a location column; each named location gets its own
    # independently drawn series. Two YAML forms are accepted:
    #   locations: [oslo, bergen]                      # names only
    #   locations: {oslo: {population: 700000}, ...}   # with per-location overrides
    # Both normalize to the same internal pair below (a name list + an
    # overrides dict), so the engine always sees an ordered list of names.
    locations: list[str] = Field(default=["loc"], min_length=1)
    # Per-location overrides, keyed by name. Empty for the list form. Not a
    # YAML field itself — it is populated from the mapping form by the
    # validator below, and excluded from dumps so metadata round-trips.
    location_overrides: dict[str, LocationSpec] = Field(
        default_factory=dict, exclude=True
    )
    variables: list[VariableSpec]
    disease_cases: DiseaseSpec

    @model_validator(mode="before")
    @classmethod
    def _normalize_locations(cls, data: object) -> object:
        """Accept either a name list or a name→overrides mapping.

        Runs before field validation (mode="before"), turning the mapping
        form into the internal ``locations`` list + ``location_overrides``
        dict. The list form is left untouched.
        """
        if not isinstance(data, dict):
            return data
        locations = data.get("locations")
        if isinstance(locations, dict):
            if not locations:
                raise ValueError("locations mapping must not be empty.")
            # dict preserves insertion order, so location order is the YAML order.
            data["locations"] = list(locations.keys())
            data["location_overrides"] = locations
        return data

    # mode="after" runs once the individual fields are already validated, so
    # cross-field relationships can be checked safely here.
    @model_validator(mode="after")
    def _check_cross_section(self) -> "ScenarioConfig":
        """Hard errors that span multiple fields.

        1. Referential integrity: every ``depends_on.variable`` must name a
           declared variable.
        2. Lag sanity: a lag >= n_total can never appear in the data.
        3. Location names must be unique (duplicates would silently double
           rows in the output).
        4. ``start_period`` must be a valid label for the chosen resolution.
        """
        if self.start_period is not None:
            try:
                parse_period(self.start_period, self.period)
            except ValueError as exc:
                raise ValueError(f"start_period: {exc}") from exc
        if len(set(self.locations)) != len(self.locations):
            raise ValueError(
                f"locations contains duplicate names: {self.locations}."
            )
        # A per-location int population must be a real headcount (the Field
        # range can't sit on the union arm, so it's enforced here).
        for name, override in self.location_overrides.items():
            if isinstance(override.population, int) and override.population < 1:
                raise ValueError(
                    f"location '{name}' population must be >= 1, "
                    f"got {override.population}."
                )
        # disease_cases.population is the fallback for any location that does
        # not set its own. It may be omitted ONLY when every location does —
        # otherwise a location would have no population.
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
        defined = [v.name for v in self.variables]
        # Variable names become output columns, so they must not collide with
        # the built-in columns (which would silently overwrite them) or with
        # each other (a duplicate silently drops one series).
        reserved = {"time_period", "location", "disease_cases", "population"}
        clashes = sorted(reserved.intersection(defined))
        if clashes:
            raise ValueError(
                f"variable names may not be reserved column names: {clashes}. "
                f"Reserved: {sorted(reserved)}."
            )
        duplicates = sorted({n for n in defined if defined.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"variables contains duplicate names: {duplicates}."
            )
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
        return self

    def population_for(self, location: str) -> "int | PopulationSpec":
        """The population SOURCE for ``location``.

        Returns the location's own population if the mapping form set one,
        otherwise the scenario's top-level ``disease_cases.population``. The
        source is either a fixed ``int`` headcount or a ``PopulationSpec``
        (a generator); the engine turns it into a per-period array. This is
        the single place the engine asks "what is the population here?", so
        list-form and mapping-form scenarios go through the same path.
        """
        override = self.location_overrides.get(location)
        if override is not None and override.population is not None:
            return override.population
        return self.disease_cases.population


def parse_config(data: dict) -> ScenarioConfig:
    """Validate a raw scenario dict (from ``load_yaml``) into a ScenarioConfig.

    Raises ``pydantic.ValidationError`` (with field-specific messages) on any
    hard error; see the module docstring for what counts as one.
    """
    # ScenarioConfig(**data) unpacks the dict's keys as keyword arguments;
    # Pydantic does all validation in the constructor.
    return ScenarioConfig(**data)


def validate_scenario(config: ScenarioConfig) -> list[str]:
    """Return warnings for suspicious-but-legal scenarios. Never raises.

    The CLI prints these to stderr and proceeds — they flag likely mistakes
    (or unusual-but-intentional choices like decoy variables) without
    blocking the run.
    """
    warnings: list[str] = []

    # Orphan variables: declared but not used by any dependency. May be an
    # intentional decoy/confounder, so this is a warning, never an error.
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

    # A from_csv variable feeds the SAME real series to every output location
    # (there is no per-location source mapping), which with several locations
    # is a likely surprise — warn regardless of whether source_location is set.
    if len(config.locations) > 1:
        for var in config.variables:
            if var.generate == "from_csv":
                warnings.append(
                    f"variable '{var.name}' uses from_csv, but the scenario has "
                    f"{len(config.locations)} locations; every location will get "
                    f"the same real series."
                )

    return warnings
