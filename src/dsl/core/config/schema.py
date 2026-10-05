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

    A model (not a plain dict) so a typo'd override key is rejected.
    ``population`` is required whenever any series in the scenario counts.
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

    Population is deliberately NOT here: it belongs to a location, not to a
    disease, so two count series cannot disagree about the same place.
    """

    model_config = _STRICT

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
            ValueError: If median_rate >= max_rate — the sigmoid shift is
                undefined outside (0, 1).
        """
        # The model shifts its sigmoid by logit(median_rate / max_rate),
        # only defined for a ratio strictly between 0 and 1.
        if self.median_rate >= self.max_rate:
            raise ValueError(
                f"median_rate ({self.median_rate}) must be smaller than "
                f"max_rate ({self.max_rate})."
            )
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
    # Resolved into `at` once the split is known, so every fold's TEST half
    # holds this many — otherwise the author must work out the fold
    # boundaries by hand and redo it whenever k or n_total changes. A list
    # gives a count per fold, in order, for deliberately uneven coverage.
    per_fold: int | list[int] | None = None

    @model_validator(mode="after")
    def _check_timing(self) -> "EventSpec":
        """Check exactly one timing source is given, and that it is usable.

        Returns:
            self, unchanged, if the timing is well formed.

        Errors Caught (raised to caller):
            ValueError: If not exactly one of rate/at/per_fold is set, or any
                listed period is negative.
        """
        given = sum(
            x is not None for x in (self.rate, self.at, self.per_fold)
        )
        if given != 1:
            raise ValueError(
                "an event needs exactly one of 'rate' (fires randomly), 'at' "
                "(fires at these periods) or 'per_fold' (one per fold's test "
                f"half); {given} were given."
            )
        if self.per_fold is not None:
            counts = (
                self.per_fold
                if isinstance(self.per_fold, list)
                else [self.per_fold]
            )
            if not counts:
                raise ValueError("'per_fold' must not be an empty list.")
            if any(n < 0 for n in counts):
                raise ValueError(
                    f"'per_fold' counts must be >= 0, got {self.per_fold}."
                )
            if not isinstance(self.per_fold, list) and self.per_fold < 1:
                raise ValueError(
                    "'per_fold' must be >= 1; use a list like [1, 0, 1] to "
                    "leave a specific fold empty on purpose."
                )
            return self
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


class RangeSpec(BaseModel):
    """A uniform range, drawn per event rather than fixed.

    A single number means every event has exactly that strength, so a model
    can match it by memorising the constant. A range makes each one differ,
    which asks the harder question: did the model recover how *hard* the
    event was, not just when it happened.
    """

    model_config = _STRICT

    min: float
    max: float

    @model_validator(mode="after")
    def _check_order(self) -> "RangeSpec":
        """Check the range is non-empty.

        Returns:
            self, unchanged, if min < max.

        Errors Caught (raised to caller):
            ValueError: If min is not below max.
        """
        if self.min >= self.max:
            raise ValueError(
                f"min ({self.min}) must be below max ({self.max})."
            )
        return self

    def draw(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` values uniformly from the range.

        Args:
            size: How many values to draw.
            rng: Seeded generator, so the draws are reproducible.

        Returns:
            An array of ``size`` floats in [min, max).
        """
        return rng.uniform(self.min, self.max, size=size)


class EventEffectSpec(BaseModel):
    """What an event does to one series when it fires.

    ``multiplier`` scales the series (proportional: it grows with the series'
    own level and can never move a series sitting at zero); ``add`` shifts it
    by a fixed amount regardless of level. Both are signed, so one event can
    raise rainfall and lower temperature in the same period.
    """

    model_config = _STRICT

    # A number is exact; a {min, max} range is drawn per event, so each one
    # differs.
    multiplier: float | RangeSpec | None = None
    add: float | RangeSpec | None = None

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

    def apply(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        rng: "np.random.Generator | None" = None,
    ) -> np.ndarray:
        """Apply this effect wherever the event fires.

        Args:
            values: The series so far.
            mask: Boolean mask of the periods the event fires in.
            rng: Seeded generator, needed only when the effect is a range —
                each firing then gets its own strength.

        Returns:
            A copy of values with the effect applied at the masked periods.

        Errors Caught (raised to caller):
            ValueError: If the effect is a range but no rng was given.
        """
        out = values.copy()
        effect = self.multiplier if self.multiplier is not None else self.add
        if isinstance(effect, RangeSpec):
            if rng is None:
                raise ValueError(
                    "a {min, max} event effect needs a random generator; "
                    "this is an internal error, not a scenario problem."
                )
            strength = effect.draw(int(mask.sum()), rng)
        else:
            strength = effect
        if self.multiplier is not None:
            out[mask] = out[mask] * strength
        else:
            out[mask] = out[mask] + strength
        return out


class Fold(BaseModel):
    """One train/test division of the dataset.

    A time split names periods and keeps every location on both sides; a
    location split names locations and keeps every period. Both are expressed
    here so the writer needs no case per kind.
    """

    model_config = _STRICT

    index: int
    train_periods: list[int]
    test_periods: list[int]
    train_locations: list[str]
    test_locations: list[str]


class SplitSpec(BaseModel):
    """The ``split:`` block — how the dataset is divided for evaluation.

    ``time`` cuts the periods, asking a model to forecast forward;
    ``location`` holds whole places out, asking it to generalise sideways.
    """

    model_config = _STRICT

    kind: Literal["time", "location"]
    # None for a location split: it defaults to one fold per location.
    k: int | None = Field(default=None, ge=1)
    # Only meaningful for a time split; holding a location out has no ordering.
    scheme: Literal["expanding", "blocked"] | None = None
    # A floor on the first fold's training size, so early folds are not too
    # short to learn anything from.
    min_train: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_shape(self) -> "SplitSpec":
        """Check the options given suit the kind of split chosen.

        Returns:
            self, with scheme defaulted for a time split.

        Errors Caught (raised to caller):
            ValueError: If a location split is given time-only options.
        """
        if self.kind == "location":
            if self.scheme is not None:
                raise ValueError(
                    "'scheme' applies only to a time split; holding a "
                    "location out has no period ordering to scheme."
                )
            if self.min_train is not None:
                raise ValueError(
                    "'min_train' applies only to a time split."
                )
        elif self.scheme is None:
            # Expanding is the forecasting default: never train on the future.
            object.__setattr__(self, "scheme", "expanding")
        return self


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
    # How the dataset is divided for evaluation; None writes no folds.
    split: SplitSpec | None = None
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
        defined = [spec.name for spec in self.series]
        self._check_series_names(defined)
        self._check_dependencies(defined)
        self._check_acyclic()
        self._check_events()
        self._check_split()
        self._resolve_per_fold_events()
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
        # Population belongs to the place, so every location must state it
        # as soon as anything in the scenario counts against it.
        if not any(spec.counts is not None for spec in self.series):
            return
        missing = [
            name
            for name in self.locations
            if self.location_overrides.get(name) is None
            or self.location_overrides[name].population is None
        ]
        if missing:
            raise ValueError(
                f"these locations need a population: {missing}. A count "
                f"series draws against its location's population, so write "
                f"it as 'locations: {{{missing[0]}: {{population: 100000}}}}'."
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

    def _resolve_per_fold_events(self) -> None:
        """Turn each ``per_fold`` event into the periods it fires at.

        Spreads the requested count evenly inside every fold's TEST half, so
        the periods follow from the split instead of being written down — and
        move on their own when k or n_total changes.

        Errors Caught (raised to caller):
            ValueError: If a per_fold event has no time split to place it in.
        """
        wanted = [name for name, e in self.events.items() if e.per_fold]
        if not wanted:
            return
        if self.split is None:
            raise ValueError(
                f"events {wanted} use 'per_fold', which needs a 'split:' "
                f"block to know where the folds are; add one, or use 'at'."
            )
        if self.split.kind != "time":
            raise ValueError(
                f"events {wanted} use 'per_fold', but the split is by "
                f"{self.split.kind}, which has no period boundaries to place "
                f"an event between; use 'at' or 'rate' instead."
            )

        folds = self.folds()
        for name in wanted:
            event = self.events[name]
            if isinstance(event.per_fold, list):
                if len(event.per_fold) != len(folds):
                    raise ValueError(
                        f"event '{name}' gives 'per_fold' "
                        f"{event.per_fold} ({len(event.per_fold)} values), "
                        f"but the split has {len(folds)} folds; give one "
                        f"count per fold, or a single number for all."
                    )
                counts = event.per_fold
            else:
                counts = [event.per_fold] * len(folds)

            periods: list[int] = []
            for fold, count in zip(folds, counts):
                test = fold.test_periods
                # Evenly inside the block, away from both edges: with n=1
                # that is the middle, with n=2 the thirds, and so on.
                for i in range(count):
                    periods.append(test[(len(test) * (2 * i + 1)) // (2 * count)])
            # Pydantic models are not frozen here, but assignment revalidates;
            # set both fields at once so the one-of check still holds.
            object.__setattr__(event, "at", sorted(set(periods)))
            object.__setattr__(event, "per_fold", None)

    def _check_split(self) -> None:
        """Check the split divides this scenario into usable folds.

        The size checks ask ``folds()`` for the real boundaries rather than
        re-deriving them, so validation cannot drift from what is actually
        written: every fold must have something to train on AND something to
        test on, on both the period and the location axis.

        Errors Caught (raised to caller):
            ValueError: If a location split has too few locations, if
                min_train leaves no room to test, or if the requested k
                produces a fold with an empty train or test side.
        """
        if self.split is None:
            return

        if self.split.kind == "location":
            if len(self.locations) < 2:
                raise ValueError(
                    f"a location split needs at least 2 locations to hold "
                    f"one out, but the scenario has {len(self.locations)}."
                )
            if self.split.k is not None and self.split.k > len(self.locations):
                raise ValueError(
                    f"split k is {self.split.k}, but the scenario has only "
                    f"{len(self.locations)} locations to divide."
                )
        else:
            min_train = self.split.min_train or 0
            if min_train >= self.n_total:
                raise ValueError(
                    f"split min_train is {min_train}, but n_total is "
                    f"{self.n_total}; there would be no periods left to "
                    f"test on."
                )

        self._check_folds_are_usable()

    def _check_folds_are_usable(self) -> None:
        """Reject a k that yields a fold with an empty side.

        Both axes matter. A time split with too many folds leaves fold 0 no
        periods before its test block; a location split holding out every
        location leaves nothing to train on.

        Errors Caught (raised to caller):
            ValueError: If any fold has an empty train or test side, naming
                the axis that collapsed.
        """
        k = len(self.folds())
        for fold in self.folds():
            empty = (
                not fold.train_periods
                or not fold.test_periods
                or not fold.train_locations
                or not fold.test_locations
            )
            if not empty:
                continue
            if self.split.kind == "location":
                side = "train on" if not fold.train_locations else "test on"
                raise ValueError(
                    f"split k is {k} over {len(self.locations)} locations, "
                    f"which leaves fold {fold.index} with no location to "
                    f"{side}; every fold needs at least one location on each "
                    f"side, so k must be between 2 and "
                    f"{len(self.locations)}."
                )
            side = "train on" if not fold.train_periods else "test on"
            raise ValueError(
                f"split k is {k} with n_total {self.n_total}, which leaves "
                f"fold {fold.index} with no periods to {side}. An expanding "
                f"split keeps its first n_total // (k + 1) periods for "
                f"training only, which is 0 here; use a smaller k or a "
                f"longer n_total."
            )

    def folds(self) -> list[Fold]:
        """Divide the scenario into train/test folds.

        The single place fold boundaries are decided, so the writer and the
        report agree on what a fold is.

        Returns:
            One Fold per division, in order. Empty when no split is declared.
            Example: 5 folds, each naming the periods and locations on both
            sides.
        """
        if self.split is None:
            return []
        if self.split.kind == "location":
            return self._location_folds()
        return self._time_folds()

    def _time_folds(self) -> list[Fold]:
        """Divide the periods, keeping every location on both sides.

        Returns:
            One Fold per test block. "expanding" trains on everything before
            the block (so a model never sees the future); "blocked" trains on
            everything outside it.
        """
        k = self.split.k or 1
        everyone = list(self.locations)
        if self.split.scheme == "blocked":
            # Every period is tested on exactly once, and training is
            # whatever lies outside the block — including later periods.
            reserved = 0
        else:
            # Fold 0 must have something to train on, so the opening periods
            # are never tested. min_train raises that floor.
            reserved = max(self.split.min_train or 0, self.n_total // (k + 1))
        span = self.n_total - reserved
        base, extra = divmod(span, k)

        folds, start = [], reserved
        for index in range(k):
            size = base + (1 if index < extra else 0)
            test = list(range(start, start + size))
            if self.split.scheme == "blocked":
                # Built once per fold, not once per period: this runs on
                # every folds() call, and the report calls it repeatedly.
                held_out = set(test)
                train = [p for p in range(self.n_total) if p not in held_out]
            else:
                train = list(range(start))
            start += size
            folds.append(
                Fold(
                    index=index,
                    train_periods=train,
                    test_periods=test,
                    train_locations=everyone,
                    test_locations=everyone,
                )
            )
        return folds

    def _location_folds(self) -> list[Fold]:
        """Hold locations out, keeping every period on both sides.

        Returns:
            One Fold per held-out group. With k below the location count the
            locations are divided into k groups rather than dropped.
        """
        names = list(self.locations)
        k = self.split.k or len(names)
        periods = list(range(self.n_total))
        base, extra = divmod(len(names), k)

        folds, start = [], 0
        for index in range(k):
            size = base + (1 if index < extra else 0)
            held = names[start : start + size]
            start += size
            folds.append(
                Fold(
                    index=index,
                    train_periods=periods,
                    test_periods=periods,
                    train_locations=[n for n in names if n not in held],
                    test_locations=held,
                )
            )
        return folds

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

    def population_for(self, location: str) -> "int | PopulationSpec":
        """Resolve a location's population.

        The single place the engine asks "how many people live here?". Only
        meaningful when the scenario has a count series, which is when the
        schema requires every location to declare one.

        Args:
            location: The location name.

        Returns:
            The location's declared population — a fixed headcount or a
            generator producing one value per period.
        """
        return self.location_overrides[location].population


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
    max_lag = max(
        (d.lag + _transform_lag(d.transforms) for d in deps), default=0
    )
    if config.split is not None:
        if config.split.scheme == "blocked":
            warnings.append(
                "split scheme 'blocked' trains on periods that come AFTER "
                "the test block, so a fold can leak the future into training; "
                "use 'expanding' to evaluate forecasting."
            )
        folds = config.folds()
        if deps and folds:
            shortest = min(len(f.train_periods) for f in folds)
            if max_lag >= shortest:
                warnings.append(
                    f"max dependency lag ({max_lag}) covers the whole "
                    f"training split of the shortest fold ({shortest} "
                    f"periods); that fold has no observed values to learn "
                    f"from (all warm-up NaN)."
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
