# DSL — synthetic climate-health data

A YAML-based DSL for generating synthetic climate-health datasets. A scenario
file declares how climate variables relate to disease (lags, weights, missing
data), and the tool generates a dataset containing those relationships. The
data can then be used to check how well a forecasting model recovers them.

Output is formatted for [CHAP](https://chap.dhis2.org/), but is plain CSV.

## Quickstart

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv
uv pip install -e ".[dev]"
uv run dsl run examples/basic_scenario.yaml -o out/
```

Or with stock tooling:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
dsl run examples/basic_scenario.yaml -o out/
```

The result (`out/simulated_data.csv`):

```
time_period,location,rainfall,mean_temperature,disease_cases,population
2000-W01,loc,2.152358553260388,15.312795196983668,,100
2000-W02,loc,1.4800080127540343,16.05069353269311,,100
2000-W03,loc,2.3752259025028235,17.621544261654282,,100
2000-W04,loc,2.4702836813159346,18.21508589989203,8.0,100
...
```

Weekly rainfall and temperature, and disease cases that depend on both with a
3-week delay. The first 3 `disease_cases` values are empty on purpose: with a
lag of 3 there is no driver signal yet.

## Output files

| File | When | Contents |
|---|---|---|
| `simulated_data.csv` | always | The full dataset: `time_period`, `location`, one column per variable, `disease_cases`, `population`. This is the file to give CHAP — it does its own train/test hiding. |
| `train.csv`, `test.csv` | only if `train_fraction` is set | A split in time (the first `train_fraction` of each location's periods vs the rest), all columns intact — for evaluation outside CHAP. |

Rerunning the same scenario produces identical files: all randomness comes
from the `seed` in the YAML.

Output is also checked against CHAP's dataset rules (required columns,
standard covariate names, monthly/weekly period format, consecutive and
per-location-identical periods, no NaN in covariates). Findings are printed
as warnings; with `dsl run ... --strict-chap` they become errors and nothing
is written. Scenarios don't have to be CHAP-compatible — other variable names
or daily/yearly resolution are fine for use outside CHAP.

## Writing a scenario

A scenario is one YAML file. The bundled example
(`examples/basic_scenario.yaml`):

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
  population: 100
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
| `locations` | list of str | `["loc"]` | Named locations. Each gets its own independently drawn series of `n_total` periods, stacked in long format with a `location` column. |
| `variables` | list | required | The independent (climate) variables — see below. |
| `disease_cases` | mapping | required | How the dependent disease signal is built — see below. |

### `variables` entries

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required | Becomes the output column name. For CHAP datasets use CHAP's names: `rainfall`, `mean_temperature`. |
| `generate` | str | required | Which generator produces the series (see [built-in generators](#built-in-generators)). |
| `params` | mapping | `{}` | Passed straight to that generator; each generator validates its own. |

A variable is just a named series — adding one more (e.g. a decoy the disease
does *not* depend on) needs no code, only YAML.

### `disease_cases` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `population` | int ≥ 1 | required | Constant population; scales the incidence model and caps the counts. |
| `depends_on` | list | required | The drivers of disease (see below). |
| `autoregressive` | bool | `false` | Add a random-walk component, giving the signal memory of its own past. |
| `missing_rate` | float, 0–1 | `0` | Fraction of disease values randomly blanked to simulate reporting gaps. |
| `max_rate` | float, 0–1 | `0.3` | Maximum incidence: cases never exceed ~`max_rate × population`. |
| `median_rate` | float | `0.1` | Incidence in a typical period. Must be **smaller than** `max_rate`. |

Each `depends_on` entry:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `variable` | str | required | Name of a declared variable. |
| `lag` | int ≥ 0 | `0` | Delay in periods between the driver and its effect on disease. |
| `weight` | float | `1.0` | Strength of the driver relative to the others (drivers are standardized first, so weights are comparable across units). |

## Built-in generators

### `seasonal_spike` — rainy-season shape

A low baseline with a pronounced, smooth spike that peaks at the same point
every year (a Gaussian bump, wrapping correctly across the year boundary).

| Param | Default | Meaning |
|---|---|---|
| `baseline` | `2.0` | The value far from the spike (dry-season level). |
| `spike_height` | `20.0` | How far above the baseline the peak rises. |
| `spike_center` | `26` | Period offset of the peak within the year (26 ≈ mid-year for weekly data). |
| `spike_width` | `4.0` | Width of the spike, in periods (> 0). |
| `noise` | `0.5` | Std. dev. of added Gaussian noise; `0` makes the series fully deterministic. |

### `seasonal_smooth` — temperature shape

A smooth sine wave, one full cycle per year at any resolution.

| Param | Default | Meaning |
|---|---|---|
| `mean` | `15.0` | The value the wave oscillates around. |
| `amplitude` | `10.0` | Swing above/below the mean (≥ 0). `0` gives pure noise around the mean — useful as an aperiodic driver. |
| `phase` | `0.0` | Phase offset in radians (shifts where in the year the peak falls). |
| `noise` | `0.5` | Std. dev. of added Gaussian noise; `0` disables it. |

## How `disease_cases` is generated

The disease signal is a population-relative Poisson incidence model, not a
plain weighted sum:

1. Start from a seasonal baseline (one sine cycle per year), so disease has
   its own seasonality even with no drivers.
2. For each `depends_on` entry: delay the driver by `lag` (causally — the
   warm-up becomes NaN; values never wrap around from the end of the series),
   standardize it to a z-score, multiply by `weight`, and add.
3. If `autoregressive`, add a random walk (cumulative white noise).
4. Squash through a sigmoid shifted so a typical period lands near
   `median_rate`, then scale: `rate × population × max_rate`. The sigmoid
   guarantees incidence stays below `max_rate` no matter how extreme the
   drivers get.
5. Draw integer counts from a Poisson distribution (seeded), capped at
   `population`.
6. Blank the first `max(lag)` rows (no valid driver signal) and apply
   `missing_rate` last.

## Extending the DSL

Two extension points, one mental model: **generators create a series;
transforms modify one.** Both live in extension folders where every file
registers itself — you never edit the core machinery.

First, check whether you need code at all. A new *variable* that reuses an
existing shape is pure YAML:

```yaml
variables:
  - name: wind
    generate: seasonal_smooth # reuse
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

That's the whole workflow: the file is auto-discovered on import, the schema
passes `params` through, and the engine finds the name in the registry. No
other file changes. New transforms work the same way under
`src/dsl/transforms/` (subclass `Transform`, implement `apply(series, rng)`,
register with `@register_transform`).

The only core file ever edited after the initial build is
`src/dsl/core/config/schema.py`, and only for a genuinely new top-level
concept (a new `disease_cases` field, a new global setting) — with schema
tests in the same commit.

## Project layout

```
src/dsl/
├── core/                  # locked machinery — not edited when adding features
│   ├── extension/         #   registry + the two abstract base classes
│   ├── config/            #   YAML loader + Pydantic schema/validation
│   └── pipeline/          #   periods, disease model, engine, CSV output
├── generators/            # extension zone: one file = one variable shape
├── transforms/            # extension zone: one file = one series modification
└── cli.py                 # the `dsl run` command
tests/                     # mirrors the package; conftest.py has shared fixtures
```

## Development

Run the tests with:

```bash
uv run pytest
```

The suite is layered: **determinism** tests (same seed → identical output)
underpin the known-ground-truth claim; **ground-truth recovery**
(`tests/test_ground_truth.py`) proves a declared driver→disease lag is
actually recoverable from the output; **validation** tests feed broken
scenarios and assert clear, field-specific rejections; **unit/integration**
tests cover each component and the config→DataFrame pipeline. When adding a
feature, add its tests in the same commit — aim for tests that would fail if
the logic broke, not for a coverage percentage
(`--cov-report=term-missing` is a gap-finder, not a score).

Commits follow [Conventional Commits](https://www.conventionalcommits.org):
`type(scope): description`, e.g. `feat(generators): add gusty wind generator`.
