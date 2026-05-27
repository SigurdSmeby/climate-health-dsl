# DSL — build log

> **Note:** this is a temporary development log, written phase by phase as the
> tool is built. It will be replaced by the real README (usage docs) once the
> build is complete. The full specification lives in `BUILD_PLAN.md`.

A YAML-based DSL for generating synthetic climate-health datasets with known
ground truth, formatted for [CHAP](https://chap.dhis2.org/). Built test-driven,
phase by phase, one Conventional Commit per phase.

**Status:** Phases 0–7 done. The full pipeline works from Python: scenario →
validation → generators → disease signal → DataFrame → CHAP-format CSVs on
disk (with the optional train/test split). Still pending: the `dsl run` CLI
and the ground-truth recovery test (Phases 8–9).

---

## Phase 0 — Scaffolding

**Features**
- `src/` layout package skeleton: `dsl/core/{extension,config,pipeline}/` (the
  locked machinery) and `dsl/{generators,transforms}/` (the extension zones).
- `pyproject.toml` with hatchling build, runtime deps (pydantic 2, pyyaml,
  numpy, pandas), a `dev` extra (pytest, pytest-cov), and the `dsl` console
  script entry point.
- Auto-discovery `__init__.py` in both extension zones: every module dropped
  into `generators/` or `transforms/` is imported automatically, which is what
  triggers its self-registration. Adding a feature = adding one file.
- Environment managed with `uv` (`uv venv`, `uv pip install -e ".[dev]"`,
  `uv run pytest`).

**Considerations**
- Added a `.gitignore` (venv, caches, `out/`) — not in the plan, but standard.
- Commits carry no Co-Authored-By trailer (maintainer preference).

## Phase 1 — Config layer

**Features**
- `core/config/loader.py` — `load_yaml(path)`: YAML file → plain dict via
  `yaml.safe_load`. Distinct, clear errors for a missing file, malformed YAML
  (filename + original parse error chained), and a top level that isn't a
  mapping. Pure I/O — interpreting fields is the schema's job.
- `core/config/schema.py` — Pydantic v2 models: `ScenarioConfig`,
  `VariableSpec` (the generic name/generate/params envelope), `DependencySpec`
  (variable/lag/weight), `DiseaseSpec` (depends_on, population, autoregressive,
  missing_rate, plus incidence knobs `max_rate=0.3`, `median_rate=0.1`).
- All models use `extra="forbid"`, so a typo'd field (`peroid:`) is rejected
  instead of silently ignored. Ranges enforced with `Field` constraints
  (`train_fraction`/`missing_rate` in 0–1, `n_total >= 1`, `population >= 1`).
- Two-tier validation:
  - **Hard errors** (raise, run stops): everything Pydantic catches, plus two
    cross-section checks in a `@model_validator` — referential integrity
    (every `depends_on.variable` must be a declared variable; the error names
    the bad reference *and* lists the valid names) and lag sanity
    (`lag >= n_total` is impossible).
  - **Warnings** (`validate_scenario(config) -> list[str]`, never raises):
    orphan/decoy variables, `missing_rate >= 0.5`, `train_fraction >= 0.95`,
    and `n_total` shorter than one seasonal cycle.
- `train_fraction` is optional (`None` → only the single full CSV later).

**Considerations**
- `periods_per_year` was created early (it belongs to Phase 2) because the
  "shorter than one seasonal cycle" warning needs it.
- `DependencySpec.lag` got a `ge=0` constraint not spelled out in the plan: a
  negative lag would mean disease precedes its cause.

## Phase 2 — Period helpers

**Features**
- `core/pipeline/periods.py` — `periods_per_year(period)`: daily→365,
  weekly→52, monthly→12, yearly→1.
- `format_period(index, period, start_year=2000)` producing CHAP-format labels
  with correct year rollover:
  - daily → `20000101` (compact `YYYYMMDD`, not ISO dashes)
  - weekly → `2000-W01` (zero-padded; index 52 rolls to `2001-W01`)
  - monthly → `2000-01` (index 12 rolls to `2001-01`)
  - yearly → `2000`

**Considerations**
- Daily labels use a real calendar (`datetime.date` + `timedelta`), so month
  lengths and leap years are correct — e.g. 2000 is a leap year, so index 366
  lands on `20010101`. Weekly stays a flat 52-weeks-per-year counter, matching
  the old reference code's convention (not true ISO weeks).

## Phase 3 — Registry + generators (the extensible heart)

**Features**
- `core/extension/registry.py` — the reusable `Registry` class:
  `@register(name)` decorator (raises on duplicate names) and `get(name)`
  (raises a `KeyError` listing all available names). One class, instantiated
  once for generators and once for transforms.
- `core/extension/generator_base.py` — the `VariableGenerator` ABC with the
  abstract `generate(n_periods, period, rng)` method, the generator registry
  instance, and the `register_generator` / `get_generator` aliases.
- `generators/seasonal_spike.py` — low baseline + a yearly Gaussian-shaped
  spike ("rainy season"). Params: `baseline`, `spike_height`, `spike_center`
  (period offset of the peak), `spike_width`, `noise`. Validated in
  `__init__`.
- `generators/seasonal_smooth.py` — a smooth yearly sine wave ("temperature").
  Params: `mean`, `amplitude`, `phase`, `noise`.
- Both scale to the period resolution via `periods_per_year` (one seasonal
  cycle per year whether data is daily, weekly, or monthly), and both are
  verified reproducible: same seed → bit-identical, different seed → different.

**Considerations**
- **Added a `noise` param the plan didn't list.** The plan requires
  "different seed → different output" for every generator, but the listed
  params describe purely deterministic curves — with no randomness, every
  seed gives identical output and that test can never pass. `noise` (additive
  Gaussian, default 0.5, `0` disables) resolves the contradiction and makes
  the series more realistic. Shape tests run with `noise: 0` so they're exact.
- `seasonal_spike` measures distance to the peak **circularly**, so a spike
  centred near week 1 correctly wraps across the year boundary (week 51 is 2
  weeks from a week-1 peak, not 50).
- The registry test re-imports `dsl.generators` and asserts both built-ins are
  found — that exercises the actual auto-discovery path, not just the class.

## Phase 4 — Transforms

**Features**
- `core/extension/transform_base.py` — the `Transform` ABC with the abstract
  `apply(series, rng)` method, plus a second `Registry` instance and the
  `register_transform` / `get_transform` aliases. Same plugin pattern as
  generators: drop a file in `transforms/`, it self-registers.
- `transforms/lag.py` — **causal** lag: shifts a series forward by `n`
  periods and fills the first `n` positions with NaN (the warm-up the
  disease model later blanks). Deliberately NOT `np.roll`: roll is circular
  and wraps the end of the series onto the start, leaking future values into
  the past — the old reference code did this, and it's wrong for a causal
  forecasting benchmark.
- `transforms/missing.py` — replaces a `rate` fraction of entries with NaN.
  The mask comes from the passed seeded rng, so the same seed always blanks
  the same entries.

**Considerations**
- Transforms never mutate their input: `apply` returns a modified **copy**
  (`astype(float)` both copies and makes the array NaN-capable, since integer
  arrays can't hold NaN). Tests pin this down for both transforms.
- Edge case decided: a lag longer than the series returns all-NaN (the whole
  series is warm-up) rather than raising — the schema's `lag < n_total` hard
  error already prevents this in real scenarios.
- `missing` blanks each entry independently with probability `rate`, so the
  realized NaN fraction is approximately (not exactly) `rate`.

## Phase 5 — Disease model

**Features**
- `core/pipeline/disease.py` — `build_disease_cases(drivers, spec, rng,
  n_periods, period)`: the population-relative Poisson incidence model,
  ported from the reference code (not a plain linear sum):
  1. seasonal baseline `eta` — one sine cycle per year;
  2. each dependency's driver is lagged (causally), z-score standardized,
     multiplied by its `weight`, and added to `eta`;
  3. optional autoregressive random walk (`cumsum` of white noise, σ=0.2);
  4. sigmoid shifted by `logit(median_rate / max_rate)` so a typical period
     maps near `median_rate`, then `rate × population × max_rate`;
  5. Poisson counts drawn from the seeded rng, capped at `population`;
  6. first `max_lag` rows blanked to NaN (the lag warm-up has no valid
     signal);
  7. `missing_rate` applied last via the `missing` transform.
- All randomness flows from the passed seeded rng — fixes the reference
  code's unseeded global `np.random`.

**Considerations**
- **New hard error in the schema:** `median_rate >= max_rate` is rejected.
  Discovered via TDD: the sigmoid shift is `logit(median/max)`, which is
  undefined (division by zero) at a ratio of 1. The plan didn't list this
  constraint; without it a legal-looking YAML would crash mid-run.
- Standardization uses `nanmean`/`nanstd` (the lagged series carries a NaN
  warm-up) and guards zero variance: a constant driver becomes zeros instead
  of NaN-ing the whole signal (tested).
- The warm-up rows are temporarily zero-filled before the Poisson draw (the
  sampler can't take NaN rates) and blanked afterwards.
- **Seasonal confounding, found while testing:** with a *seasonal* driver,
  cross-correlation recovers the lag one period off — the disease's own
  seasonal baseline correlates with the driver and biases the estimate. This
  is real epidemiology (seasonality confounds lag estimation), not a bug.
  The lag-mechanism test therefore uses an aperiodic (random) driver, which
  recovers the exact lag across every seed tried. Worth remembering when
  designing the Phase 9 ground-truth scenario.
- The old code int-cast `sigmoid × population` *before* multiplying by
  `max_rate`; that truncation quirk was not ported (float all the way to the
  Poisson draw).

## Phase 6 — Engine

**Features**
- `core/pipeline/engine.py` — `run(config: ScenarioConfig) -> pd.DataFrame`,
  the "translate config into executable code" step:
  1. creates the single `np.random.default_rng(config.seed)` and threads it
     through everything;
  2. for each YAML variable, looks the `generate:` string up in the registry
     and runs the generator with its `params:`;
  3. builds `disease_cases` from the generated drivers;
  4. assembles the tidy DataFrame: `time_period` (CHAP labels), one column
     per variable in YAML declaration order, `disease_cases`, and a constant
     `population` column.
- The engine hard-codes no variable names: columns come from the YAML, so a
  CHAP-ready scenario just names its variables `rainfall` and
  `mean_temperature` (verified by a test using a `wind` variable instead).

**Considerations**
- Importing `dsl.generators` / `dsl.transforms` inside `engine.py` is what
  triggers plugin auto-discovery — without those imports the registries
  would be empty at runtime. Tests import the engine, so they exercise this.
- Integration tests pin the full contract: exact CHAP column order, row
  count, period labels with year rollover, constant population, blanked
  warm-up rows, params pass-through (a `noise: 0` generator is identical
  across seeds), bit-identical reruns, and a helpful KeyError (listing real
  generators) for an unknown `generate:` name.

## Phase 7 — Output (CHAP formatting)

**Features**
- `core/pipeline/output.py` — `write_output(df, config, out_dir)`:
  - always writes `simulated_data.csv` via `to_csv(index=False)` (no
    leading unnamed index column), columns in CHAP order;
  - `disease_cases` is present for the whole series apart from the blanked
    lag warm-up — CHAP needs the true values and does its own train/test
    hiding during evaluation;
  - if `train_fraction` is set, ALSO writes `train.csv` (first
    `floor(n_total × train_fraction)` rows) and `test.csv` (the rest), a
    plain row split with all columns intact, for evaluation outside CHAP.
- All CHAP-specific naming/format decisions are isolated in this one module
  (filenames are module constants).

**Considerations**
- The output directory is created if missing (`mkdir(parents=True,
  exist_ok=True)`), so `dsl run scenario.yaml -o out/some/new/dir` will just
  work; rerunning overwrites existing files.
- A test verifies the split is a true row slice: `train.csv` + `test.csv`
  concatenated equals `simulated_data.csv` exactly.
- The no-index requirement is tested against the raw header line of the
  file, not through pandas — that's what CHAP's reader actually sees.

---

## Test suite

91 tests, all green (`uv run pytest`): import smoke test, loader errors, both
validation tiers, period formats and rollover, registry behaviour,
auto-discovery, generator shapes, parameter validation, transform behaviour
(causal shift, NaN warm-up, reproducible masks, no input mutation), and
determinism.
