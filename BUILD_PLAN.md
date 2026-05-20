# DSL — Build Plan & Specification

A YAML-based **external domain-specific language (DSL)** for generating synthetic
climate-health datasets with known ground truth, used to evaluate predictive
models for climate-sensitive diseases. Output is formatted to match
[CHAP](https://chap.dhis2.org/) conventions.

> **Audience of this document:** an AI coding agent (Claude Code) building this
> project from an empty directory on Arch Linux. Follow the build order in
> §5 exactly. Use **test-driven development**: for every component, write the
> tests first, watch them fail, then implement until green.

---

## 0. Before you start (read this first)

**Read the existing reference code before writing Phase 5.** An older, working but
messy version of this tool exists at:

```
/home/sigurd/Documents/Master/climate_health_simulations
```

Read it to understand and **port the exact domain math** — specifically the disease
signal model (the sigmoid → rate × population × max_rate → Poisson logic, the
seasonal baseline, and the standardization of drivers) and the precise CHAP CSV
column behaviour. Phase 5 of this plan describes that model; the old code is the
authoritative source for the exact formulas, so reproduce them rather than inventing
new ones.

**Critical:** read the old code for the **maths and output behaviour only — NOT the
architecture.** The old code is factory-based with a fixed variable-type enum; that
rigidity is exactly what this project replaces. The architecture, file structure,
registry/plugin pattern, schema, and validation all follow *this* plan, not the old
code's design.

**How to build:**
- Work through the phases in §5 **in order**. After each phase: run `pytest`, get it
  green, then commit using that phase's provided Conventional Commit message.
- It's TDD: write the listed tests first, watch them fail, then implement.
- If a CHAP detail or a modelling specific is uncertain, implement a sensible default
  and leave a clear `# TODO:` comment rather than blocking — do not guess silently.
- Reproducibility is mandatory: every random draw flows from one seeded
  `np.random.default_rng(config.seed)`. The old code's global `np.random` (no seed)
  was a bug; do not copy it.
- Lag is **causal** (see Phase 4) — a forward shift with a blanked warm-up, not the
  old code's circular `np.roll`. This is intentional.

---

## 1. Purpose & design goals

The tool reads a scenario described in a YAML file, validates it, generates
synthetic climate variables (e.g. rainfall, temperature) and a dependent disease
signal with explicitly controlled relationships (lags, weights, missing data),
and writes the result in **CHAP's verified format** — a single CSV with columns
`time_period, rainfall, mean_temperature, disease_cases, population` (confirmed
against `chap_core`'s data model). The disease signal is a population-relative
Poisson incidence model, not a plain linear sum.

Hard requirements, in priority order:

1. **Modular & extensible.** Adding a new variable generator must require
   creating *one new file* and nothing else — no edits to the parser, schema,
   or engine. This is achieved with a **registry (plugin) pattern**, not by
   editing a central dispatcher.
2. **Reproducible.** Given a seed, output is bit-for-bit identical. Every source
   of randomness flows from a single seeded `numpy.random.Generator`. This is a
   correctness requirement — "known ground truth" depends on it.
3. **Explicit & readable.** All simulation assumptions live in the YAML file,
   not hidden in code. Validation is two-tier: impossible or certainly-wrong
   scenarios raise a clear, field-specific **error** and stop the run; suspicious
   but legal scenarios emit a **warning** and proceed. Errors point at the
   offending field.
4. **Thoroughly tested.** Layered tests (unit → validation → integration →
   ground-truth recovery) so a mistake is immediately visible.

---

## 2. Tech stack & conventions

- **Language:** Python 3.11+
- **Dependencies:**
  - `pydantic>=2` — schema definition and validation (readable errors)
  - `pyyaml` — YAML parsing
  - `numpy` — numeric generation
  - `pandas` — tabular assembly and output
- **Dev dependencies:** `pytest`, `pytest-cov`
- **Layout:** `src/` layout (package not importable unless installed → forces
  tests to run against the installed package).
- **Style:** type hints everywhere; small, single-responsibility modules.
- **Comments — important:** the maintainer is new to Python. Write generously for
  a learner. Every module begins with a docstring explaining its role in the
  pipeline. Every class and public function has a docstring saying what it does,
  its parameters, and what it returns. Add inline comments on any non-obvious
  line that explain the **Python idiom and the *why*** (e.g. what a decorator
  does, why `np.random.default_rng` is used, what an abstract method enforces) —
  not comments that merely restate the code. Favour clear, explicit code over
  clever one-liners.
- **Tooling:** use `uv` — `uv venv` to create the environment, `uv pip install -e
  ".[dev]"` to install, and `uv run pytest` / `uv run dsl ...` to execute without
  activating. uv also manages the Python version. (Install it first if missing: on
  Arch, `sudo pacman -S uv`.)

### Core vs extensions — the most important structural rule

The package is split into two zones, and Claude Code must respect the boundary:

- **`src/dsl/core/` — the locked machinery.** The registry, the abstract base
  classes, the engine, the schema, output, and period helpers. These are written
  once in the build and then left alone. The *only* core file ever edited
  afterwards is `schema.py`, and only when adding a genuinely new top-level
  concept (a new global field, a new disease-model option).
- **`src/dsl/generators/` and `src/dsl/transforms/` — the extension zones.**
  Each file here is one self-contained feature. Adding a feature means adding
  *one file in one of these folders* and nothing else.

This is what "modular" means for this project: features are added by dropping a
file into an extension zone, never by editing the core.

---

## 3. Architecture overview

The pipeline is a straight line through five decoupled layers. This mirrors the
three DSL responsibilities (parse → validate → translate to executable code):

```
YAML file
   │  loader.py        (parse:    yaml.safe_load → dict)
   ▼
raw dict
   │  schema.py        (validate: dict → ScenarioConfig via Pydantic)
   ▼
ScenarioConfig
   │  engine.py        (translate: look up generators in the REGISTRY,
   │                     run them, build dependent disease signal,
   │                     apply transforms)
   ▼
pandas.DataFrame (tidy, with known ground truth)
   │  output.py        (format to CHAP conventions, train/test split)
   ▼
CSV files on disk
```

**The central design move:** the engine never hard-codes which generators
exist. It reads `generate: seasonal_spike` from the config, looks that string up
in a registry, and instantiates whatever is registered under it. Generators
register *themselves* via a decorator and are auto-discovered. The schema stays
generic about generator parameters (it validates the *envelope* — `name`,
`generate`, `params` — and each generator validates *its own* params). As a
result, the schema and engine do not grow when generators are added.

Use a **shallow inheritance hierarchy**: one abstract base class per extension
point (`VariableGenerator`, `Transform`), with concrete subclasses extending the
*base directly*. Do **not** create deep chains where subclasses extend other
subclasses — share code via helper functions or composition instead.

### Two kinds of extension: generators vs transforms

There are exactly two extension points, and the difference is simple:

- A **generator** *creates* a variable's values from nothing — from parameters,
  the time axis, and randomness. Input: number of periods, the resolution, a
  random source. Output: a full time series. Rainfall and temperature are
  generators. Wind, if added, is also a generator. They *invent* a variable.
- A **transform** *modifies* a series that already exists. Input: an array.
  Output: a changed array. `lag` (shift in time), `missing` (insert NaN gaps),
  and noise are transforms. They never invent a variable; they alter one.

Mental model: **generators produce a series; transforms modify a series.**

Note that *adding a variable* and *adding code* are different. If a new variable
(e.g. wind) can reuse an existing generator's shape, it is added in the YAML file
with no new code. A new code file is only needed for a genuinely new *shape*
(generator) or a new *modification* (transform). See §8 for the full workflow.

---

## 4. Target directory structure

```
dsl/                              # repo root (currently empty)
├── pyproject.toml
├── README.md                     # copy §8 of this document into here
├── BUILD_PLAN.md                 # this file
├── examples/
│   └── basic_scenario.yaml       # the example from §7
├── src/
│   └── dsl/
│       ├── __init__.py
│       │
│       ├── core/                 # ===== LOCKED MACHINERY — do not edit =====
│       │   ├── __init__.py
│       │   ├── extension/        # how plugins attach (the extension system)
│       │   │   ├── __init__.py
│       │   │   ├── registry.py        # the shared Registry class (register / get)
│       │   │   ├── generator_base.py  # VariableGenerator ABC + generator registry
│       │   │   └── transform_base.py  # Transform ABC + transform registry
│       │   ├── config/           # read + validate the YAML
│       │   │   ├── __init__.py
│       │   │   ├── loader.py           # YAML file → dict
│       │   │   └── schema.py           # Pydantic models (edit ONLY for new top-level concepts)
│       │   └── pipeline/         # run the simulation + write output
│       │       ├── __init__.py
│       │       ├── periods.py         # period strings + periods-per-year
│       │       ├── disease.py         # builds the dependent disease signal
│       │       ├── engine.py          # orchestrates the pipeline
│       │       └── output.py          # CHAP formatting + train/test split
│       │
│       ├── generators/           # ===== EXTENSION ZONE — add one file per shape =====
│       │   ├── __init__.py       # AUTO-DISCOVERS every module here (do not touch)
│       │   ├── seasonal_spike.py
│       │   └── seasonal_smooth.py
│       │
│       ├── transforms/           # ===== EXTENSION ZONE — add one file per modification =====
│       │   ├── __init__.py       # auto-discovers (do not touch)
│       │   ├── lag.py
│       │   └── missing.py
│       │
│       └── cli.py                # `dsl run scenario.yaml -o out/`
└── tests/
    ├── conftest.py               # shared fixtures (seeded rng, sample config)
    ├── core/
    │   ├── test_registry.py
    │   ├── test_loader.py
    │   ├── test_schema.py
    │   ├── test_periods.py
    │   ├── test_disease.py
    │   ├── test_engine.py        # integration: config → DataFrame
    │   └── test_output.py
    ├── generators/
    │   ├── test_seasonal_spike.py
    │   └── test_seasonal_smooth.py
    ├── transforms/
    │   ├── test_lag.py
    │   └── test_missing.py
    ├── test_cli.py               # smoke test
    └── test_ground_truth.py      # recover the embedded relationship
```

(The `tests/` folder stays one level shallower than `core/` — mirroring the
extension/config/pipeline split in tests too would add nesting without much
benefit, so all core tests sit together in `tests/core/`.)

### What each `core/` module does

**`core/extension/` — the machinery that lets features plug in without editing the engine**

- `registry.py` — A lookup table mapping the name written in YAML (e.g.
  `"seasonal_spike"`) to the Python class that implements it. Provides the
  `@register(...)` decorator (records a class under a name) and `get(name)`
  (fetches it, raising a clear error listing available names if unknown). Written
  once as a reusable `Registry` class, instantiated once for generators and once
  for transforms. This is the single mechanism that keeps the engine ignorant of
  which features exist.
- `generator_base.py` — Defines `VariableGenerator`, the abstract base class
  every generator subclasses. An ABC declares a method (`generate(...)`) that
  subclasses are *forced* to implement, so the engine can call `.generate()` on
  any generator without knowing which one. Also creates the generator registry
  instance and exposes `register_generator` / `get_generator`.
- `transform_base.py` — The same for transforms: the `Transform` ABC with its
  required `apply(series, rng)` method, plus the transform registry. Generators
  create a series; transforms modify one.

**`core/config/` — read the YAML and check it is valid**

- `loader.py` — Reads the YAML file off disk into a plain dict (`yaml.safe_load`).
  Pure parsing/I/O; raises a clear error if the file is missing or malformed. It
  does not interpret field meanings — that is the schema's job.
- `schema.py` — The Pydantic models defining what a *valid* scenario looks like,
  turning the raw dict into a typed `ScenarioConfig` and rejecting bad input
  (wrong types, out-of-range values, unknown fields) with messages naming the
  offending field. Also enforces cross-section **hard errors** (referential
  integrity: every `depends_on.variable` must be a declared variable; lag sanity:
  `lag < n_total`) via a model validator, and provides a separate
  `validate_scenario()` that returns non-fatal **warnings** (orphan variables,
  extreme-but-legal values). Validates each variable's envelope
  (`name`/`generate`/`params`) but stays generic about generator-specific params,
  so it does not grow when generators are added. **This is the only core file ever
  edited afterward**, and only for a genuinely new top-level concept.

**`core/pipeline/` — actually run the simulation**

- `periods.py` — Time-axis helpers: `periods_per_year` (52 weekly, 12 monthly)
  and `format_period` (turns a row index into a CHAP period string like
  `2000-W01` or `2000-01`, handling year rollover). Used by generators (to scale
  seasonality) and output (to label rows).
- `disease.py` — Builds the dependent `disease_cases` series from the generated
  drivers: applies each dependency's lag, multiplies by its weight, sums them,
  adds noise, converts to non-negative integer counts, and applies the
  missing-rate. This is where the controlled cause→effect relationship — the known
  ground truth — is actually created.
- `engine.py` — The orchestrator. Takes the validated `ScenarioConfig`, creates
  the single seeded random generator, looks up and runs each variable generator
  through the registry, builds `disease_cases` via `disease.py`, and assembles a
  tidy pandas DataFrame. This is the "translate config into executable code" step.
- `output.py` — Takes the finished DataFrame and writes CHAP's format: always one
  `simulated_data.csv` (`index=False`) with columns `time_period, rainfall,
  mean_temperature, disease_cases, population`, `disease_cases` present for the whole
  series (CHAP hides test values itself during evaluation). If `train_fraction` is
  set, it *also* writes `train.csv`/`test.csv` as a row split for use outside CHAP.
  All CHAP-specific formatting is isolated here.

`cli.py` (outside `core/`) wires the chain together — loader → schema → engine →
output — as the `dsl run scenario.yaml -o out/` command.

---

## 5. Step-by-step implementation guide (TDD order)

Build in this order. Each phase: **write the listed tests first → run pytest →
implement → green → refactor → commit.** Each phase ends with a ready-to-use
**Commit** message in [Conventional Commits](https://www.conventionalcommits.org)
format — use it verbatim (adjust the body if the work differed).

### Phase 0 — Scaffolding
- Create the `src/` layout, empty `__init__.py` files, and `pyproject.toml`
  (see §6.1). Configure pytest to use `tests/` and the `src/` package.
- Create the venv and install with **uv**: `uv venv`, then `uv pip install -e
  ".[dev]"`. Run tests with `uv run pytest` (no need to activate the venv).
- Add a trivial `test_import.py` asserting `import dsl` works. Make it pass.

**Commit:**
```
chore: scaffold project structure, packaging, and tooling

Add src/ layout, pyproject.toml (pydantic, numpy, pandas; pytest dev extra),
the empty package skeleton, and a venv + editable install. Verify `import dsl`.
```

### Phase 1 — Config layer (in `core/config/`)
- `core/config/loader.py`: `load_yaml(path) -> dict` using `yaml.safe_load`. Raise a
  clear error if the file is missing or malformed.
- `core/config/schema.py`: Pydantic v2 models.
  - `VariableSpec`: `name: str`, `generate: str`, `params: dict = {}`. The `name`
    becomes the output column name, so CHAP-compatible scenarios name their
    variables `rainfall` and `mean_temperature`.
  - `DependencySpec`: `variable: str`, `lag: int = 0`, `weight: float = 1.0`.
  - `DiseaseSpec`: `depends_on: list[DependencySpec]`, `population: int`,
    `autoregressive: bool = False`, `missing_rate: float = 0.0`. (Optional
    incidence-model knobs with defaults matching the existing code: `max_rate:
    float = 0.3`, `median_rate: float = 0.1`.)
  - `ScenarioConfig`: `period: Literal["daily","weekly","monthly","yearly"]`,
    `n_total: int`, `seed: int = 0`, `train_fraction: float | None = None`,
    `variables: list[VariableSpec]`, `disease_cases: DiseaseSpec`.
  - `train_fraction` is **optional**. When `None`, the tool writes only the single
    full CSV. When set (0–1), it *also* writes `train.csv`/`test.csv` (a row split).
  - Set `model_config = ConfigDict(extra="forbid")` on every model so a typo'd
    field name (e.g. `peroid:`) is rejected instead of silently ignored.
  - Use `Field(...)` constraints for ranges (`train_fraction` and `missing_rate`
    in 0–1, `n_total` ≥ 1, `population` ≥ 1) so Pydantic reports them as
    field-level errors.
  - Add a `parse_config(data: dict) -> ScenarioConfig` helper.

  **Validation is two-tier — errors stop the run, warnings only inform.**

  *Hard errors (refuse to run).* Most come free from Pydantic (missing fields,
  wrong types, out-of-range values, unknown `period`, forbidden extra fields).
  Two cross-section checks need a `@model_validator(mode="after")` on
  `ScenarioConfig` because they involve relationships between fields:
  - **Referential integrity:** every `depends_on.variable` must match the `name`
    of a declared variable. On a mismatch, raise with a message that names the bad
    reference and lists the valid names, e.g. *"disease_cases depends on 'rainfal',
    which is not a defined variable. Defined variables: ['rainfall', 'mean_temperature']."*
  - **Lag sanity:** a dependency with `lag >= n_total` is a hard error (the delayed
    relationship cannot appear in the data).

  *Warnings (run, but flag).* Add a separate
  `validate_scenario(config: ScenarioConfig) -> list[str]` function (not inside
  Pydantic) that returns human-readable warning strings for suspicious-but-legal
  scenarios. It must NOT raise. Cases to detect:
  - An **orphan variable**: a declared variable that no `depends_on` entry uses
    (may be an intentional decoy/confounder, so warn — never block).
  - Extreme-but-legal values, e.g. `missing_rate >= 0.5`, or a set
    `train_fraction >= 0.95`.
  - `n_total` smaller than one seasonal cycle (`periods_per_year(period)`).

  The CLI (Phase 8) prints any returned warnings to stderr and proceeds; a raised
  error from `parse_config` stops execution before anything is generated.
- **Tests (`core/test_loader.py`, `core/test_schema.py`):**
  - valid config parses to a `ScenarioConfig`;
  - a missing required field raises with a message naming the field;
  - `train_fraction: 1.5` and `missing_rate: 2.0` are rejected;
  - `period: fortnightly` (unsupported) is rejected;
  - an unknown/typo'd field (`peroid:`) is rejected;
  - **referential integrity:** `depends_on` a variable that is not declared raises,
    and the message lists the valid variable names;
  - **lag sanity:** `lag >= n_total` raises;
  - **orphan warning:** a declared-but-unused variable makes `validate_scenario`
    return a warning string while `parse_config` still succeeds (does not raise).

**Commit:**
```
feat(config): add scenario schema, YAML loader, and two-tier validation

Pydantic models (ScenarioConfig, VariableSpec, DependencySpec, DiseaseSpec)
with extra=forbid and range constraints; a safe_load wrapper; referential-
integrity and lag-sanity hard errors; and a non-fatal validate_scenario pass
for orphan variables and extreme-but-legal values.
```

### Phase 2 — Periods (in `core/pipeline/`)
- `core/pipeline/periods.py`: `periods_per_year(period) -> int`
  (daily→365, weekly→52, monthly→12, yearly→1) and
  `format_period(index, period, start_year=2000) -> str` producing CHAP-compatible
  period strings. Match CHAP's exact formats (verified against `chap_core`):
  - daily → `2000-01-01` style is NOT used; CHAP daily is `YYYYMMDD`, e.g. `20000101`
  - weekly → `2000-W01` (ISO-like, Monday-start, zero-padded to 2 digits)
  - monthly → `2000-01`
  - yearly → `2000`
  Handle year rollover.
- **Tests:** weekly period 0 is `2000-W01`, period 52 rolls to `2001-W01`; monthly
  period 12 rolls to `2001-01`; daily produces `YYYYMMDD`; yearly produces `YYYY`.

**Commit:**
```
feat(periods): add period helpers for daily/weekly/monthly/yearly

periods_per_year plus CHAP-format period strings (YYYYMMDD, YYYY-Wnn,
YYYY-MM, YYYY) with correct year rollover.
```

### Phase 3 — Registry + Generators (the extensible heart)
- `core/extension/registry.py`: a small reusable `Registry` class with `.register(name)`
  (decorator that raises if `name` is already taken) and `.get(name)` (raises
  `KeyError` listing available names if missing). Used for both generators and
  transforms so the logic exists once. See §6.2.
- `core/extension/generator_base.py`:
  - Create one `Registry` instance for generators; expose its `register` and
    `get` as `register_generator` / `get_generator`.
  - `VariableGenerator(ABC)` with abstract
    `generate(self, n_periods: int, period: str, rng: np.random.Generator) -> np.ndarray`.
- `generators/__init__.py`: auto-discover all sibling modules (see §6.3) so a new
  file registers itself with zero further edits.
- `generators/seasonal_spike.py`: `@register_generator("seasonal_spike")`. Models a
  variable with a low baseline and a pronounced seasonal "rainy season" spike.
  Params (with defaults): `baseline`, `spike_height`, `spike_center` (period offset
  of peak), `spike_width`. Validate params in `__init__`. Output length must equal
  `n_periods`. The seasonal peak must scale to the period resolution (use
  `periods_per_year`).
- `generators/seasonal_smooth.py`: `@register_generator("seasonal_smooth")`. A smooth
  sine wave over the yearly cycle. Params: `mean`, `amplitude`, `phase`. Must scale
  correctly to the period resolution via `periods_per_year` (daily/weekly/monthly/yearly).
- **Tests (`core/test_registry.py`, `generators/test_*`):** registering a duplicate
  name raises; `get_generator("nope")` raises and lists available names; each
  generator returns an array of length `n_periods`; **same seed → identical output,
  different seed → different output**; `seasonal_spike` has its maximum at the
  configured `spike_center`; `seasonal_smooth` is approximately periodic with the
  configured period.

**Commits (two — registry/base first, then the generators):**
```
feat(core): add plugin registry and generator base class

Reusable Registry (register/get), the VariableGenerator ABC, and
auto-discovery of generator modules so new generators self-register on import.
```
```
feat(generators): add seasonal_spike and seasonal_smooth generators

A seasonal rainy-season spike and a smooth yearly sine wave, each scaled to
the period resolution via periods_per_year.
```

### Phase 4 — Transforms
- `core/extension/transform_base.py`: a second `Registry` instance for transforms; expose
  `register_transform` / `get_transform`. `Transform(ABC)` with
  `apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray`.
- `transforms/__init__.py`: auto-discovery (same as generators).
- `transforms/lag.py`: `@register_transform("lag")`. Shifts a series forward by `n`
  periods so the driver's effect on disease is delayed.
  **Decision — use causal shift, not circular.** Shift forward and leave the first
  `n` positions undefined (these become the `max_lag` warm-up that the disease model
  blanks). Do NOT use `np.roll` (circular): the existing code used `np.roll`, which
  wraps the end of the series onto the start and so leaks future values into the
  past — wrong for a causal forecasting benchmark. (If you ever need to reproduce
  the old behaviour exactly, that is the one place to switch back, knowingly.)
- `transforms/missing.py`: `@register_transform("missing")`. Replaces a `rate`
  fraction of entries with `NaN`, selected via the passed `rng` (so it's
  reproducible).
- **Tests:** lag of 3 shifts the signal by exactly 3 positions; lag of 0 is
  identity; missing with rate 0 changes nothing; missing with rate 0.5 on a long
  series produces ~50% NaN; same seed → same mask.

**Commit:**
```
feat(transforms): add transform base, causal lag, and missing transforms

Transform ABC with its own registry; a causal (non-circular) lag shift; and
reproducible missing-value injection driven by the seeded RNG.
```

### Phase 5 — Disease model (in `core/pipeline/`)
- `core/pipeline/disease.py`: `build_disease_cases(drivers: dict[str, np.ndarray], spec: DiseaseSpec,
  rng, n_periods, period) -> np.ndarray`. This is a **population-relative incidence
  model** (matching the existing code's intent), NOT a plain linear sum:
  1. Start from a linear predictor `eta` (e.g. a seasonal baseline over the yearly cycle).
  2. For each dependency: take the named driver, apply its `lag` (causal), standardize
     it (z-score), multiply by `weight`, and add to `eta`.
  3. If `spec.autoregressive`: add a random-walk component, `cumsum` of white noise
     drawn from `rng`, to `eta`.
  4. Map `eta` to a rate with a sigmoid shifted so the median maps near `median_rate`,
     then scale: `rate * population * max_rate`.
  5. Draw counts with `rng.poisson(...)`, **cap at `population`**, and cast to `int`.
  6. Blank the first `max_lag` rows (set to NaN) — the lag warm-up has no valid signal.
  7. Apply `missing_rate` via the `missing` transform last.
  - Keep all randomness flowing from the passed `rng` (reproducibility). Note: the
    existing code used global `np.random` with no seed and was **not reproducible** —
    this seeded version is a deliberate fix.
- **Tests:** with a single driver, weight 1, lag 0, no AR, no missing, the disease
  signal tracks the (standardized) driver; a lag of `k` shifts the disease peak `k`
  periods after the driver peak; output is non-negative integers ≤ `population`
  (apart from the blanked warm-up and injected NaN); reproducible under a fixed seed.

**Commit:**
```
feat(disease): add population-relative Poisson incidence model

Lagged, standardized, weighted drivers plus an optional autoregressive walk →
sigmoid rate × population × max_rate → seeded Poisson draw capped at
population; blank the max_lag warm-up; apply missing_rate.
```

### Phase 6 — Engine (integration, in `core/pipeline/`)
- `core/pipeline/engine.py`: `run(config: ScenarioConfig) -> pandas.DataFrame`.
  1. Build one `np.random.default_rng(config.seed)` and thread it through.
  2. For each `VariableSpec`: `get_generator(spec.generate)(**spec.params)` then
     `.generate(...)`.
  3. Build `disease_cases` from the generated drivers via `disease.py`.
  4. Assemble a tidy DataFrame with a `time_period` column (from `periods.py`), one
     column per variable (named by `VariableSpec.name`), `disease_cases`, and a
     constant `population` column from `disease_cases.population`.
  - The engine does NOT hardcode CHAP column names — they come from the variable
    names in the YAML. A CHAP-ready scenario simply names its variables `rainfall`
    and `mean_temperature` and sets `population`.
- **Tests (`test_engine.py`):** running the §7 example config yields a DataFrame
  with columns `time_period, rainfall, mean_temperature, disease_cases, population`
  and `n_total` rows; column order is stable; rerunning with the same config is
  identical.

**Commit:**
```
feat(engine): assemble a scenario into a CHAP-formatted DataFrame

Run generators via the registry under one seeded RNG, build disease_cases,
and emit time_period + variable columns + disease_cases + population.
```

### Phase 7 — Output (CHAP formatting, in `core/pipeline/`)
- `core/pipeline/output.py`: `write_output(df, config, out_dir)`.
  - Always write one **`simulated_data.csv`** with `df.to_csv(..., index=False)`
    (no index column). Columns in CHAP order:
    `time_period, rainfall, mean_temperature, disease_cases, population`.
    `disease_cases` is filled in for the whole series (the `max_lag` warm-up rows
    are NaN). Do NOT remove `disease_cases` — CHAP needs the true values and does
    its own train/test hiding during evaluation.
  - If `config.train_fraction` is set: ALSO write `train.csv` (first
    `floor(n_total * train_fraction)` rows) and `test.csv` (the rest), each a plain
    row slice of the full DataFrame with all columns intact — for evaluation
    *outside* CHAP. If `train_fraction` is `None`, write only `simulated_data.csv`.
  - **Isolate all CHAP-specific naming/format decisions in this module** so they can
    be adjusted in one place. (Column names and period formats were verified against
    `chap_core`'s `ClimateHealthTimeSeries`/`FullData` and an example dataset.)
- **Tests:** with no `train_fraction`, only `simulated_data.csv` is written and has
  the CHAP columns with no index column; with `train_fraction: 0.8`, `train.csv` +
  `test.csv` are also written with row counts matching the split and all columns
  present in both.

**Commit:**
```
feat(output): write CHAP simulated_data.csv with optional train/test split

Always emit one index-free CSV in CHAP column order; when train_fraction is
set, also write train.csv/test.csv as a plain row split.
```

### Phase 8 — CLI
- `cli.py`: `dsl run <scenario.yaml> -o <out_dir>` wiring loader → `parse_config`
  → engine → output. Register a console entry point in `pyproject.toml`.
- After `parse_config` succeeds, call `validate_scenario(config)` and print any
  returned warnings to stderr (prefixed e.g. `warning:`), then continue. If
  `parse_config` raises (a hard error), print the error message clearly and exit
  with a non-zero status *before* generating anything.
- **Tests (`test_cli.py`):**
  - invoking the CLI on the example file writes the expected files into a temp dir
    and exits 0;
  - a config with an orphan variable still exits 0 but prints a warning;
  - a config with a dangling `depends_on` reference exits non-zero and writes no
    output files.

**Commit:**
```
feat(cli): add 'dsl run' command wiring loader→schema→engine→output

Print validate_scenario warnings to stderr and continue; on a hard validation
error, print it clearly and exit non-zero before generating anything.
```

### Phase 9 — Ground-truth recovery (validity check)
- `test_ground_truth.py`: generate a scenario where one driver causes disease with
  a known lag, then assert a simple check (cross-correlation peak between driver
  and disease, or a trivial regression) **recovers that lag**. This is the test
  that proves the simulator actually embeds the relationship it claims — keep it
  prominent.

**Commit:**
```
test(ground-truth): verify the simulator embeds a recoverable lag

Generate a known driver→disease lag and assert it is recovered, proving the
ground truth is real rather than assumed.
```

> **YAGNI reminder:** implement only `seasonal_spike` and `seasonal_smooth`, and
> only `lag` and `missing`. Do not pre-build a disease-model registry or extra
> generators. The registry pattern makes adding the *next* one a one-file change,
> so there is no cost to waiting until a real need appears.

---

## 6. Canonical reference code

Use these implementations for the shared scaffolding so the decoupling holds.
Fill in domain logic (the generator maths, disease formula) to satisfy the tests.

### 6.1 `pyproject.toml`

```toml
[project]
name = "dsl"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pydantic>=2", "pyyaml", "numpy", "pandas"]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[project.scripts]
dsl = "dsl.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

### 6.2 `core/extension/registry.py` (one reusable registry, used twice)

A registry is just a dictionary that maps a name (the string used in the YAML,
e.g. `"seasonal_spike"`) to the class that implements it. The decorator adds an
entry; the lookup retrieves it. Writing it once as a class avoids duplicating the
same logic in the generator and transform base files.

```python
"""A name -> class lookup used to plug features in without editing the engine."""

class Registry:
    """Maps DSL names (strings from YAML) to the classes that implement them."""

    def __init__(self, kind: str):
        # `kind` is only used to make error messages clearer ("generator"/"transform").
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        """Decorator: `@my_registry.register("foo")` stores the class under "foo".

        A decorator is a function that takes a class and returns it (here,
        unchanged) after running a side effect — in this case, recording it in
        the dictionary. That side effect is what lets a feature "announce itself".
        """
        def decorator(cls):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type:
        """Look up a registered class by name, with a helpful error if missing."""
        if name not in self._items:
            available = sorted(self._items)
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {available}"
            )
        return self._items[name]
```

### 6.3 `core/extension/generator_base.py` and `core/extension/transform_base.py`

Each base file creates one `Registry` instance and the abstract base class that
features subclass. (An *abstract base class* defines a method that subclasses are
*required* to implement — `generate`/`apply` — so every feature has a guaranteed
shape the engine can rely on.)

```python
# core/extension/generator_base.py
"""Defines what every variable generator must look like, plus their registry."""
from abc import ABC, abstractmethod
import numpy as np
from .registry import Registry      # same folder (core/extension/), so relative

# The single registry instance every generator file will register itself into.
generator_registry = Registry("generator")
register_generator = generator_registry.register   # convenience aliases
get_generator = generator_registry.get

class VariableGenerator(ABC):
    """Base class for anything that CREATES a variable's time series."""

    @abstractmethod
    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Return an array of length ``n_periods`` (the variable's values)."""
```

`core/extension/transform_base.py` is identical in shape: it creates
`transform_registry = Registry("transform")`, exposes `register_transform` /
`get_transform`, and defines `Transform(ABC)` with an abstract
`apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray` —
the base class for anything that MODIFIES an existing series.

### 6.4 `generators/__init__.py` (auto-discovery → zero-touch extension)

This makes every file in the folder import automatically, which is what triggers
the `@register_generator(...)` decorator on each one. Because of this, adding a
new generator file requires editing nothing else — not even this file.

```python
"""Auto-import every generator module so each one registers itself on startup."""
import importlib
import pkgutil

# pkgutil.iter_modules lists the .py files in this folder. Importing each one
# runs its @register_generator decorator, adding it to the registry. New files
# are picked up automatically — that is the whole point of this loop.
for _finder, module_name, _is_pkg in pkgutil.iter_modules(__path__):
    if not module_name.startswith("_"):
        importlib.import_module(f"{__name__}.{module_name}")
```

`transforms/__init__.py` is the same.

### 6.5 Example generator (the shape every generator file follows)

```python
"""A smooth seasonal sine wave — used for variables like temperature."""
import numpy as np
from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import periods_per_year

@register_generator("seasonal_smooth")   # this string is what you write in YAML
class SeasonalSmoothGenerator(VariableGenerator):
    def __init__(self, mean: float = 15.0, amplitude: float = 10.0, phase: float = 0.0):
        # These become the `params:` you can set per-variable in the YAML file.
        # Validate them here if needed (e.g. amplitude >= 0).
        self.mean = mean
        self.amplitude = amplitude
        self.phase = phase

    def generate(self, n_periods, period, rng):
        ppy = periods_per_year(period)       # 52 for weekly, 12 for monthly
        t = np.arange(n_periods)             # the time axis: 0, 1, 2, ...
        # One full sine cycle per year, scaled to the period resolution.
        return self.mean + self.amplitude * np.sin(2 * np.pi * t / ppy + self.phase)
```

---

## 7. Example scenario (`examples/basic_scenario.yaml`)

```yaml
period: weekly
n_total: 78
seed: 42
train_fraction: 0.8        # OPTIONAL — omit to write only the single full CSV
variables:
  - name: rainfall
    generate: seasonal_spike
  - name: mean_temperature  # CHAP's column name (not "temperature")
    generate: seasonal_smooth
disease_cases:
  population: 100000        # constant population; scales the incidence model
  depends_on:
    - variable: rainfall
      lag: 3
      weight: 2.0
    - variable: mean_temperature
      lag: 3
      weight: 1.0
  autoregressive: false     # if true, add a random-walk component to the signal
  missing_rate: 0.05
```

This describes a weekly dataset where `disease_cases` depends on `rainfall` and
`mean_temperature`, each delayed by 3 periods, with rainfall weighted more
heavily. The `lag` makes the delay explicit so a model can be tested on whether
it recovers it — something impossible with real data, where the true delay is
unknown. `population` scales the incidence model; `missing_rate` injects small
gaps. Variable names become the output column names, so naming them `rainfall`
and `mean_temperature` (plus `population`) makes the output directly CHAP-compatible.

`train_fraction` is optional: omit it and the tool writes one full
`simulated_data.csv`; set it (e.g. `0.8`) and the tool *also* writes `train.csv`
(first 80% of rows) and `test.csv` (last 20%), all columns intact.

---

## 8. README content (copy into `README.md`)

```markdown
# DSL

A YAML-based DSL for generating synthetic climate-health datasets with known
ground truth, for simulation-based evaluation of disease-forecasting models.
Output is formatted for CHAP.

## Install

Requires Python 3.11+. Using `uv` (recommended):

    uv venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"

Or with stock tooling:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

## Run

    dsl run examples/basic_scenario.yaml -o out/

This writes to `out/`:
- `simulated_data.csv` — the full dataset, CHAP-ready, with columns
  `time_period, rainfall, mean_temperature, disease_cases, population`. Feed this
  to CHAP; it does its own train/test splitting during evaluation.
- `train.csv` and `test.csv` — only written if the scenario sets `train_fraction`;
  a plain row split (all columns intact) for evaluating models outside CHAP.

## Git commits

This project uses [Conventional Commits](https://www.conventionalcommits.org).
Write each commit as:

    type(scope): short description

Common types: `feat` (new capability), `fix` (bug fix), `test` (tests only),
`docs`, `refactor`, `chore` (tooling/scaffolding). The scope is the area you
touched, e.g. `config`, `generators`, `output`. Example:

    feat(generators): add gusty wind generator

(The build plan lists a ready commit message for each phase.)

## Test

### How tests are organised

Tests live in `tests/` and mirror the package structure (`tests/core/`,
`tests/generators/`, `tests/transforms/`). Shared setup lives in
`tests/conftest.py` — most importantly a seeded random generator fixture, so
every test is deterministic:

    import numpy as np
    import pytest

    @pytest.fixture
    def rng():
        return np.random.default_rng(0)

### How to write a good test

Each test follows arrange → act → assert and checks one behaviour. Test the
**contract** (what the function promises), not the implementation details:

    def test_lag_shifts_series(rng):
        from dsl.transforms.lag import LagTransform
        series = np.array([0, 0, 0, 5, 0, 0])          # arrange
        result = LagTransform(n=2).apply(series, rng)  # act
        assert result[5] == 5                          # assert: the spike moved +2

There are four kinds of test in this project, in rough order of importance:

1. **Determinism** — the same seed produces identical output, a different seed
   produces different output. This underpins "known ground truth", so every
   generator and transform needs one.
2. **Ground-truth recovery** (`test_ground_truth.py`) — build a scenario with a
   known driver→disease lag, then assert a simple check recovers it. This proves
   the simulator embeds the relationship it claims.
3. **Validation** — feed deliberately broken YAML/config and assert it is
   rejected with a clear, field-specific message.
4. **Unit & integration** — each generator/transform returns the right shape and
   structure; the engine turns a full config into the expected DataFrame.

When you add a feature, add its tests in the same commit. A feature without a
test that would fail if it broke is not finished.

### Is test coverage important?

Yes — but the percentage is a means, not the goal. Coverage tells you what code
has *never run* during the tests, which is a useful way to spot gaps you forgot.
It does **not** tell you the tested code is correct. You can have 100% coverage
with tests that assert nothing meaningful, and you can have a rock-solid suite at
85%.

Aim for tests that would actually catch a real mistake. For this project the
highest-value coverage is on the logic that your results depend on: the
generators, transforms, disease model, and the ground-truth recovery test. Trivial
glue code matters less. A practical target is high coverage (≈85–95%) on the
logic modules, with `--cov-report=term-missing` used as a gap-finder rather than
a score to chase.

## Adding a new feature the DSL can model

There are two extension points: **generators** create a variable's values from
scratch; **transforms** modify a series that already exists. Everything lives in
the two extension folders — you never edit the `core/` machinery.

**First, decide whether you even need code.** Adding a *variable* is not the same
as adding *code*:

- If the variable can reuse an existing pattern, just declare it in the YAML —
  **no code at all**.
- You only write a new file for a genuinely new *shape* (generator) or new
  *modification* (transform).

### Case A: add a variable that reuses an existing pattern (e.g. wind)

If `wind` looks like a smooth seasonal curve (like temperature), add it to the
scenario file. No Python:

    variables:
      - name: wind
        generate: seasonal_smooth      # reuse an existing generator
        params:
          mean: 12
          amplitude: 4

To let disease depend on it, add an entry under `depends_on` (the `lag` and
`weight` already exist):

    disease_cases:
      depends_on:
        - variable: wind
          lag: 2
          weight: 0.5

### Case B: add a new variable *shape* (a new generator)

Only needed when no existing generator produces the shape you want (e.g. gusty,
spiky wind).

1. Create `src/dsl/generators/gusty.py`.
2. Subclass `VariableGenerator` and register it:

       """Gusty wind: a noisy series with occasional sharp spikes."""
       import numpy as np
       from dsl.core.extension.generator_base import VariableGenerator, register_generator

       @register_generator("gusty")     # the name you'll use in YAML
       class GustyGenerator(VariableGenerator):
           def __init__(self, base: float = 5.0, gust_chance: float = 0.1):
               self.base = base                 # these are the YAML `params:`
               self.gust_chance = gust_chance
           def generate(self, n_periods, period, rng):
               series = rng.normal(self.base, 1.0, size=n_periods)
               gusts = rng.random(n_periods) < self.gust_chance
               series[gusts] += rng.normal(10, 2, size=gusts.sum())
               return series

3. Use it in any scenario file:

       variables:
         - name: wind
           generate: gusty
           params:
             gust_chance: 0.15

That is the whole workflow. The new file is auto-discovered on import, the
generic schema passes `params` straight through, and the engine looks the name
up in the registry. **No `core/` file changes.**

### Case C: add a new transform (a new modification, e.g. a different gap model)

Same pattern under `src/dsl/transforms/`: create one file, subclass `Transform`,
register it with `@register_transform("my_transform")`, implement
`apply(self, series, rng)`.

### When you must touch the core

Only genuinely new *top-level* concepts require editing `core/config/schema.py` —
for example a new field on `disease_cases`, or a new global setting. Those changes
should be deliberate and accompanied by new schema tests. Nothing else in `core/`
should ever need editing once built.
```

---

## 9. Definition of done

- All tests pass (`pytest` green), including the ground-truth recovery test.
- `dsl run examples/basic_scenario.yaml -o out/` produces `simulated_data.csv`
  with CHAP columns (`time_period, rainfall, mean_temperature, disease_cases,
  population`), plus `train.csv`/`test.csv` because the example sets `train_fraction`.
- A new generator can be added by creating a single file under `generators/`,
  with **no edits to anything in `core/`**.
- Output is reproducible: rerunning the example with the same seed is identical.
- Validation is two-tier: a dangling `depends_on` reference (or `lag >= n_total`)
  raises a clear error and writes no output; an orphan variable only warns and
  still runs.
- Every module, class, and public function has a docstring, and non-obvious lines
  carry inline comments explaining the Python idiom and the *why* — the code
  should be readable by someone new to Python.
