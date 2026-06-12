# DSL — synthetic climate-health data

A YAML-based DSL for generating synthetic climate-health datasets. A scenario file declares how climate variables relate to disease (lags, weights, missing data), and the tool generates a dataset embedding those relationships — so you can check how well a forecasting model recovers a *known* ground truth.

Output is formatted for [CHAP](https://chap.dhis2.org/), but is plain CSV.

## Commands and options

| Command | What it does |
|---|---|
| `dsl new [path]` | Write a commented starter scenario to edit (default `scenario.yaml`). |
| `dsl run <scenario>` | Generate a dataset from a scenario YAML (or reproduce one from a `metadata.json`). |

**`dsl new [path]`**

| Option | Default | Meaning |
|---|---|---|
| `path` | `scenario.yaml` | Where to write the starter file. |
| `-f`, `--force` | off | Overwrite the file if it already exists. |

**`dsl run <scenario>`**

| Option | Default | Meaning |
|---|---|---|
| `scenario` | required | Path to a scenario YAML, or a `metadata.json` to reproduce a previous run. |
| `-o`, `--out-dir DIR` | auto-named | Directory to write into. If omitted, an auto-named folder under `out/` is used so previous runs are never overwritten. |
| `--plot` | off | Also write a plot of the dataset into the output directory. |
| `--plot-format FMT` | `html` | Plot format: `html` (interactive) or `png`/`svg`/`pdf`. |
| `--watch` | off | Re-run automatically whenever the scenario file is saved; serves a live-reloading plot when paired with `--plot`. |
| `--new` | off | Skip the continue-or-new prompt; always write a fresh auto-numbered folder. |

## Getting started

A hands-on path from install to a real-data experiment — one command or edit per step.

**0. Install.** Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"
```

(Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`, then drop the `uv run` prefix below.)

**1. Scaffold a starter.** Writes a small, commented scenario:

```bash
uv run dsl new my_scenario.yaml
```

(Prefer to start from a finished example? `examples/` has several ready to run, e.g. `uv run dsl run examples/basic_scenario.yaml`.)

**2. Run it and look.** `--plot` writes an interactive `plot.html`; `--watch` re-runs every time you save the file:

```bash
uv run dsl run my_scenario.yaml --plot --watch
```

Together they open a browser tab served from `localhost` that **reloads itself** whenever you save the scenario — edit, save, watch the plot update live, no manual refresh. (Drop `--watch` for a one-shot run; it just writes `plot.html`.)

When a scenario already has output, `dsl run` lists the earlier runs and asks whether to **continue one** (refine the same `out/` folder) or start fresh — so a second session doesn't silently spawn `out/my_scenario_1/`. Use `--new` to skip the prompt and always start fresh, or `-o DIR` to write somewhere specific.

**3. Read the output.** `out/my_scenario/simulated_data.csv` has one row per period — the climate columns, `disease_cases`, and `population`. The first `disease_cases` are blank on purpose: with `lag: 2` there's no driver signal yet to react to. That blanked warm-up is the lag made visible.

**4. Change one thing, watch it move.** In `my_scenario.yaml`, bump `lag: 2` to `lag: 6` and save. With `--watch` running, the dataset regenerates and the plot refreshes — the disease peak shifts later relative to rainfall. Adding a second driver is pure YAML, no code:

```yaml
variables:
  - name: rainfall
    generate: seasonal_spike
  - name: mean_temperature      # add a second climate variable
    generate: seasonal_smooth
disease_cases:
  depends_on:
    - { variable: rainfall, lag: 6 }
    - { variable: mean_temperature, lag: 2 }   # ...and a second driver
```

**5. Use real climate data.** Swap a synthetic generator for `from_csv` to drive disease off *real* climate (a bundled three-province Laos sample lives in `examples/data/`):

```yaml
variables:
  - name: rainfall
    generate: from_csv
    params: { file: examples/data/laos_subset.csv, column: rainfall }
```

**6. Explore the worked examples.** `examples/` has ready-to-run scenarios, and `examples/real_data_demo/` has five fuller ones (real, synthetic, and mixed) — each with pre-generated output and a `README` explaining what it shows. A good next step after your first scenario:

```bash
uv run dsl run examples/real_data_demo/01_vietnam_multiprovince.yaml --plot
```

The reference below documents every field.

## Output files

| File | When | Contents |
|---|---|---|
| `simulated_data.csv` | always | The full dataset: `time_period`, `location`, one column per variable, `disease_cases`, `population`. This is the file to give CHAP — it does its own train/test hiding. |
| `train.csv`, `test.csv` | only if `train_fraction` is set | A split in time (the first `train_fraction` of each location's periods vs the rest), all columns intact — for evaluation outside CHAP. |
| `metadata.json` | always | The ground truth behind the dataset: seed, lags, weights, rates, generators, tool version, and the full resolved scenario. Feed it back to `dsl run` to reproduce the data exactly — no original YAML needed; the result is byte-identical. |
| `plot.html` (or `.png`/`.svg`/`.pdf`) | only with `--plot` | A faceted plot of the covariates and `disease_cases` over time, one line per location, with the train/test boundary marked. |

Rerunning the same scenario produces identical files — all randomness comes from the `seed`. Where those files land is controlled by `-o` / `--new` / the continue prompt (see [Commands and options](#commands-and-options) and step 2 of the guide above).

Output is checked against CHAP's dataset rules (required columns, a parseable period format, consecutive periods, no NaN in covariates). Findings print as warnings; the run still writes output. In practice they only arise with `from_csv` data that has gaps — the synthetic generators always produce CHAP-valid output.

## Writing a scenario

A scenario is one YAML file. The bundled example (`examples/basic_scenario.yaml`):

```yaml
period: weekly
n_total: 78
seed: 42
train_fraction: 0.8
variables:
  - name: rainfall
    generate: seasonal_spike
  - name: mean_temperature
    generate: seasonal_smooth
disease_cases:
  population: 100000
  depends_on:
    - variable: rainfall
      lag: 3
      weight: 1.0
    - variable: mean_temperature
      lag: 3
      weight: 1.0
  autoregressive: false
  missing_rate: 0
```

### Top-level fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `period` | `daily` \| `weekly` \| `monthly` \| `yearly` | required | Time resolution. Sets the period labels (`20000101`, `2000-W01`, `2000-01`, `2000`) and the length of one seasonal cycle (365/52/12/1). |
| `n_total` | int ≥ 1 | required | Number of time periods to generate. |
| `seed` | int | `0` | Seed for all randomness. Same scenario + same seed → identical output. |
| `train_fraction` | float, 0 < x < 1 | unset | If set, also write `train.csv`/`test.csv`. |
| `start_period` | str | first period of 2000 | Where the series starts on the real calendar, in the scenario's resolution: `"2010-07"` (monthly), `"2015-W10"` (weekly), `"20100615"` (daily), `"2003"` (yearly). Relabels the output but does **not** shift the seasonal *phase* — a mid-year start still begins the seasonal cycle at index 0 (the run warns when this applies). |
| `locations` | list of str, or mapping | `["loc"]` | Named locations, each an independently drawn series of `n_total` periods, stacked in long format with a `location` column. Use the **list** form (`[oslo, bergen]`) for one shared population, or the **mapping** form to set a per-location population: `{Bokeo: {population: 75000}, ...}`. A per-location population can itself be a generator (its own growth trajectory). A location with no `population` falls back to `disease_cases.population`. |
| `variables` | list | required | The independent (climate) variables — see below. |
| `disease_cases` | mapping | required | How the dependent disease signal is built — see below. |

### `variables` entries

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required | Becomes the output column name. For CHAP datasets use CHAP's names: `rainfall`, `mean_temperature`. |
| `generate` | str | required | Which generator produces the series (see [generators](#generators)). |
| `params` | mapping | `{}` | Passed straight to that generator; each generator validates its own. |

A variable is just a named series — adding one more (e.g. a decoy the disease does *not* depend on) needs no code, only YAML.

### `disease_cases` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `population` | int ≥ 1, or a generator | required* | Population; scales the incidence model and caps the counts. A plain int is constant; a generator block (`{generate: linear_trend, params: {start: 70000, slope: 90}}`) makes it **change over time**. Shared across locations unless the `locations` mapping overrides it per location. *Optional when every location sets its own population. |
| `depends_on` | list | required | The drivers of disease (see below). |
| `autoregressive` | bool | `false` | Add a random-walk component, giving the signal memory of its own past. |
| `missing_rate` | float, 0–1 | `0` | Fraction of disease values randomly blanked to simulate reporting gaps. |
| `max_rate` | float, 0–1 | `0.3` | Maximum incidence: cases never exceed ~`max_rate × population`. |
| `median_rate` | float | `0.1` | Incidence in a typical period. Must be **smaller than** `max_rate`. |
| `count_distribution` | `poisson` \| `negative_binomial` | `poisson` | How counts are drawn from the rate. `poisson` has variance = mean; `negative_binomial` adds overdispersion (spikier, more realistic surveillance counts). |
| `overdispersion` | float > 0 | `10.0` | Only for `negative_binomial`: variance = mean + mean²/`overdispersion`, so **smaller = more variable**; large values approach Poisson. |

Each `depends_on` entry:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `variable` | str | required | Name of a declared variable. |
| `lag` | int ≥ 0 | `0` | Delay in periods between the driver and its effect on disease. |
| `weight` | float | `1.0` | Strength of the driver relative to the others (drivers are standardized first, so weights are comparable across units). |

## Generators

Each `variable` names a generator that produces its series. Built-ins:

### `seasonal_spike` — rainy-season shape

A low baseline with a pronounced, smooth Gaussian spike peaking at the same point every year (wrapping across the year boundary).

| Param | Default | Meaning |
|---|---|---|
| `baseline` | `2.0` | The value far from the spike (dry-season level). |
| `spike_height` | `20.0` | How far above the baseline the peak rises. |
| `spike_center` | `26` | Period offset of the peak within the year (26 ≈ mid-year for weekly data). |
| `spike_width` | `4.0` | Width of the spike, in periods (> 0). |
| `noise` | `0.5` | Std. dev. of added Gaussian noise; `0` makes the series fully deterministic. |
| `clamp_min` | unset | Floor for the values — set `0` for quantities like rainfall that can't be negative. |

### `seasonal_smooth` — temperature shape

A smooth sine wave, one full cycle per year at any resolution.

| Param | Default | Meaning |
|---|---|---|
| `mean` | `15.0` | The value the wave oscillates around. |
| `amplitude` | `10.0` | Swing above/below the mean (≥ 0). `0` gives pure noise — an aperiodic driver. |
| `phase` | `0.0` | Phase offset in radians (shifts where in the year the peak falls). |
| `noise` | `0.5` | Std. dev. of added Gaussian noise; `0` disables it. |
| `clamp_min` | unset | Floor for the values. |

### `flat` — non-seasonal control / decoy

A constant level plus noise, no seasonal structure. Useful as a control covariate to check a model doesn't latch onto an irrelevant variable.

| Param | Default | Meaning |
|---|---|---|
| `level` | `0.0` | The constant value the series sits at. |
| `noise` | `1.0` | Std. dev. of added Gaussian noise; `0` gives a flat line. |
| `clamp_min` | unset | Floor for the values. |

### `linear_trend` — steady drift

A straight line `start + slope · t`, optionally noisy. Models slow drift (population growth, warming, reporting changes) — a non-seasonal confounder.

| Param | Default | Meaning |
|---|---|---|
| `start` | `0.0` | Value at the first period. |
| `slope` | `1.0` | Change per period (negative falls). |
| `noise` | `0.0` | Std. dev. of added Gaussian noise. |
| `clamp_min` | unset | Floor for the values. |

`examples/confounders_and_controls.yaml` uses `flat` and `linear_trend` as decoys the disease ignores, to test whether a model finds the real driver.

### `from_csv` — real data

Reads the variable's values from a CHAP-format CSV instead of synthesizing them, for semi-synthetic experiments: real climate, synthetic disease with a controlled relationship. The data is used as-is — if the file holds fewer periods than `n_total`, the run fails rather than wrapping or extrapolating.

| Param | Default | Meaning |
|---|---|---|
| `file` | required | Path to the CSV (`time_period` column plus data columns; `location` column if multi-location). |
| `column` | required | Which column to use as this variable's values. |
| `source_location` | unset | Which location's rows to use. Set it to feed one CSV location to every output location. If unset and the CSV has several locations, each output location **auto-matches** the CSV rows of the same name (and errors if there's no match). |
| `start_period` | first row | A `time_period` label to start reading from, e.g. `"2011-01"`. |

A real multi-location sample is bundled at `examples/data/laos_subset.csv` (three Lao provinces, monthly 2010–2012, from CHAP), used by `examples/real_data_demo/laos_real_climate_from_csv.yaml`. To align the output's `time_period` labels with the source dates, set the scenario's `start_period` to the source's first period (the Laos example uses `"2010-01"`).

Note: reproducing a `from_csv` run from its `metadata.json` re-reads the source CSV by path, so byte-identical reproduction requires that file to be unchanged.

## How `disease_cases` is generated

The disease signal is a population-relative incidence model, not a plain weighted sum. It builds a per-period incidence *rate*, then draws integer counts from it (Poisson by default, or overdispersed negative binomial — your choice via `count_distribution`):

1. Start from a seasonal baseline (one sine cycle per year), so disease has its own seasonality even with no drivers.
2. For each `depends_on` entry: delay the driver by `lag` (causally — the warm-up becomes NaN; values never wrap around from the end), standardize it to a z-score, multiply by `weight`, and add.
3. If `autoregressive`, add a random walk (cumulative white noise).
4. Squash through a sigmoid shifted so a typical period lands near `median_rate`, then scale to a rate: `sigmoid × population × max_rate`. The sigmoid guarantees incidence stays below `max_rate` no matter how extreme the drivers get.
5. Draw integer counts from the rate (seeded, per the chosen distribution), capped at `population`.
6. Blank any period with no valid driver signal — the lag warm-up, plus rows where a driver value was itself missing — then apply `missing_rate` last.

See `examples/overdispersed_outbreaks.yaml` for the negative-binomial counts.

## Extending the DSL — generators and transforms

One mental model: **generators create a series; transforms modify one.** Both live in extension folders where every file registers itself — you never edit the core machinery.

First check whether you need code at all. A new *variable* that reuses an existing shape is pure YAML:

```yaml
variables:
  - name: wind
    generate: seasonal_smooth  # reuse
    params: { mean: 12, amplitude: 4 }
```

A new *shape* is one new file in `src/dsl/generators/`:

```python
"""Gusty wind: a noisy series with occasional sharp spikes."""
import numpy as np
from dsl.core.extension.generator_base import VariableGenerator, register_generator

@register_generator("gusty")  # the name you write in YAML
class GustyGenerator(VariableGenerator):
    def __init__(self, base: float = 5.0, gust_chance: float = 0.1):
        self.base = base  # these are the YAML `params:`
        self.gust_chance = gust_chance

    def generate(self, n_periods, period, rng):
        series = rng.normal(self.base, 1.0, size=n_periods)
        gusts = rng.random(n_periods) < self.gust_chance
        series[gusts] += rng.normal(10, 2, size=gusts.sum())
        return series
```

The file is auto-discovered on import, the schema passes `params` through, and the engine finds the name in the registry — no other file changes. **Transforms** work identically under `src/dsl/transforms/`: subclass `Transform`, implement `apply(series, rng)`, register with `@register_transform`.

The only core file ever edited after the initial build is `src/dsl/core/config/schema.py`, and only for a genuinely new top-level concept (a new `disease_cases` field, a new global setting) — with schema tests in the same commit.

## Project layout

```
src/dsl/
├── core/                  # locked machinery — not edited when adding features
│   ├── extension/         #   registry + the two abstract base classes
│   ├── config/            #   YAML loader + Pydantic schema/validation
│   └── pipeline/          #   periods, disease model, engine, CSV output
├── generators/            # extension zone: one file = one variable shape
├── transforms/            # extension zone: one file = one series modification
└── cli.py                 # the `dsl run` / `dsl new` commands
tests/                     # mirrors the package; conftest.py has shared fixtures
```

## Development

Run the tests with `uv run pytest`. The suite covers determinism (same seed → identical output), ground-truth recovery (`tests/test_ground_truth.py` proves a declared lag is recoverable), validation (broken scenarios give clear, field-specific errors), and the config→DataFrame pipeline. Add tests with each feature, in the same commit.

Commits follow [Conventional Commits](https://www.conventionalcommits.org): `type(scope): description`.
