# Reference

Every field, generator, transform and emitter. For looking things up — see the
[tutorial](TUTORIAL.md) to learn the tool, and [concepts](CONCEPTS.md) for why
it works this way.

**Jump to:** [scenario](#scenario) · [`split`](#split) · [`events`](#events) ·
[`series`](#series) · [`depends_on`](#depends_on) · [`counts`](#counts) ·
[generators](#generators) · [transforms](#transforms) ·
[emitters](#emitters) · [CLI](#cli) · [output](#output)

## At a glance

```yaml
period: monthly                      # daily | weekly | monthly | yearly
n_total: 60
seed: 42
start_period: 2010-01                # optional
split: { kind: time, k: 5 }          # optional
events:                              # optional
  storm: { at: [14, 47] }
locations:
  loc: { population: 100000 }
series:
  - name: rainfall
    generate: seasonal_spike
    params: { spike_height: 25 }
  - name: disease_cases
    counts: { median_rate: 0.1 }
    depends_on:
      - { series: rainfall, lag: 2, weight: 1.5 }
```

## Scenario

| Field | Default | Meaning |
|---|---|---|
| `period` | required | `daily` \| `weekly` \| `monthly` \| `yearly`. Sets the label format (`20000101`, `2000-W01`, `2000-01`, `2000`) and the periods per year (365/52/12/1). |
| `n_total` | required | Number of periods to generate. |
| `seed` | `0` | Seeds all randomness. Same scenario + seed → identical output. |
| `start_period` | 2000 start | Calendar start, in this resolution: `"2010-07"`, `"2015-W10"`, `"20100615"`, `"2003"`. Relabels only — it does **not** shift the seasonal phase (the run warns if it would matter). |
| `locations` | `["loc"]` | Mapping of name → `{population}`. Required once any series counts. A bare list works only without a count series. Population may be a generator block. |
| `events` | `{}` | Named shocks — see below. |
| `split` | unset | Evaluation folds — see below. |
| `series` | required | Everything the scenario builds. |

Population may itself vary over time:

```yaml
locations:
  loc:
    population: { generate: linear_trend, params: { start: 70000, slope: 90 } }
```

## `split`

Writes `folds/fold_N/{train,test}.csv` plus a report. Omit for a single dataset.

| Field | Default | Meaning |
|---|---|---|
| `kind` | required | `time` (forecast forward) or `location` (generalise to unseen places). |
| `k` | 1, or the location count | Number of folds. `k: 1` is a plain holdout. |
| `scheme` | `expanding` | Time only. `expanding` trains on periods *before* the test block; `blocked` tests each period once but trains on later ones too, and warns it can leak the future. |
| `min_train` | unset | Time only. Floor on fold 0's training size. |

With `kind: time, k: 5` over 120 periods, folds train on 0–19, 0–39, 0–59,
0–79, 0–99, each testing the 20 periods that follow. An expanding split keeps
its first `n_total // (k + 1)` periods for training only.

## `events`

A shock that fires in given periods and can strike **several series at once** —
the one thing a generator cannot do.

```yaml
events:
  storm: { rate: 0.05 }        # or { at: [14, 47, 88] }, or { per_fold: 1 }

series:
  - name: rainfall
    events: { storm: { multiplier: 2.1 } }   # proportional
  - name: mean_temperature
    events: { storm: { add: -4.0 } }         # absolute
```

| Field | Meaning |
|---|---|
| `rate` | Probability per period, seeded. Exactly one of `rate`/`at`/`per_fold`. |
| `at` | Explicit period offsets, fully deterministic. |
| `per_fold` | Place this many in **every fold's test half**, derived from the `split:` — so the periods follow k and `n_total` instead of being written down. Time splits only. |
| `multiplier` | On a reacting series: scales it. Cannot move a series sitting at zero. |
| `add` | On a reacting series: shifts by a fixed amount, any level. |

Both effects are signed; each series gives one or the other. Events are
regional — one draw shared by every location.

## `series`

One list holds climate and disease alike. A series is a float column unless it
carries a `counts:` block.

| Field | Default | Meaning |
|---|---|---|
| `name` | required | The output column. Conventional names: `rainfall`, `mean_temperature`, `disease_cases`. |
| `generate` | optional | A [generator](#generators) for the base. May be omitted when `depends_on` is set — the base is then flat 0. A series with neither is an error. |
| `params` | `{}` | Passed to that generator, which validates its own. |
| `depends_on` | `[]` | Parent series — any number. See below. |
| `events` | `{}` | Which declared events hit this series. |
| `missing_rate` | `0` | Fraction of values blanked to NaN. |
| `autoregressive` | unset | `{phi, noise}` — AR(1) persistence. `0 ≤ phi < 1`; `1` is a non-stationary random walk and is rejected. |
| `counts` | unset | Makes it a count series. See below. |

## `depends_on`

| Field | Default | Meaning |
|---|---|---|
| `series` | required | Another declared series. Cycles of any length are rejected. |
| `lag` | `0` | Periods of delay. The warm-up becomes NaN; values never wrap. |
| `weight` | `1.0` | Strength relative to the other parents. Parents are standardized first, so weights compare across units. Negative is protective. |
| `transforms` | `[]` | [Transforms](#transforms) applied after the lag, before standardizing. |

## `counts`

The block's presence marks a disease signal; `counts: {}` uses the defaults.
Population is **not** here — it belongs to the location.

| Field | Default | Meaning |
|---|---|---|
| `median_rate` | `0.1` | Where a typical period sits. Must be < `max_rate`. |
| `max_rate` | `0.3` | Ceiling on the expected **rate**. Individual draws scatter above it; only `population` hard-caps a drawn value. |
| `distribution` | `poisson` | `poisson` (variance = mean) or `negative_binomial` (overdispersed). |
| `overdispersion` | `10.0` | Negative binomial only. `Var = rate + rate²/k`; **smaller = more variable**. |

## Generators

A series' `generate:` names one of these.

### `seasonal_spike` — rainy season

A low baseline with a Gaussian peak at the same point each year, wrapping the
year boundary.

| Param | Default | Meaning |
|---|---|---|
| `baseline` | `2.0` | Dry-season level. |
| `spike_height` | `20.0` | Rise above baseline at the peak. |
| `spike_center` | mid-year | Period offset of the peak within the year. |
| `spike_width` | `4.0` | Width in periods (> 0). |
| `noise` | `0.5` | Gaussian noise; `0` is fully deterministic. |
| `clamp_min` | unset | Floor — set `0` for rainfall. |

### `seasonal_smooth` — temperature

One sine cycle per year, at any resolution.

| Param | Default | Meaning |
|---|---|---|
| `mean` | `15.0` | Value it oscillates around. |
| `amplitude` | `10.0` | Swing above/below (≥ 0). `0` gives pure noise — an aperiodic driver. |
| `phase` | `0.0` | Offset in radians. |
| `noise` | `0.5` | Gaussian noise. |
| `clamp_min` | unset | Floor. |

### `flat` — control / decoy

A constant plus noise, no seasonality. A decoy checks that a model does not
latch onto an irrelevant series.

| Param | Default | Meaning |
|---|---|---|
| `level` | `0.0` | The constant. |
| `noise` | `1.0` | Gaussian noise; `0` is a flat line. |
| `clamp_min` | unset | Floor. |

### `linear_trend` — drift

`start + slope · t`. Models slow drift: growth, warming, changing reporting.

| Param | Default | Meaning |
|---|---|---|
| `start` | `0.0` | Value at period 0. |
| `slope` | `1.0` | Change per period; negative falls. |
| `noise` | `0.0` | Gaussian noise. |
| `clamp_min` | unset | Floor. |

### `outbreak_shocks` — rare extremes

Randomly timed (Poisson) sharp spikes, so no two years look alike — unlike the
regular `seasonal_spike`.

| Param | Default | Meaning |
|---|---|---|
| `baseline` | `0.0` | Quiet level. |
| `noise` | `0.0` | Gaussian noise on the baseline. |
| `rate` | `1.0` | Expected shocks per year. |
| `magnitude` | `20.0` | Rise above baseline per shock. |
| `duration` | `1` | Periods each shock stays elevated. |
| `clamp_min` | unset | Floor. |

### `from_csv` — real data

Reads values from a long-format CSV instead of synthesizing them: real climate,
synthetic disease, known relationship. Used as-is — a file shorter than
`n_total` fails rather than wrapping.

```yaml
- name: rainfall
  generate: from_csv
  params: { file: examples/data/laos_subset.csv, column: rainfall }
```

| Param | Default | Meaning |
|---|---|---|
| `file` | required | Path to the CSV (`time_period` + data columns, plus `location` if multi-location). |
| `column` | required | Which column to read. |
| `source_location` | unset | Which location's rows to use. Unset with a multi-location CSV, each output location **auto-matches** the CSV rows of its own name (and errors if there is none). |
| `start_period` | first row | A label to start reading from, e.g. `"2011-01"`. |

A three-province Lao sample is bundled at `examples/data/laos_subset.csv`, used
by [`from_real_climate.yaml`](../examples/from_real_climate.yaml). Set the
scenario's `start_period` to the source's first period to align labels.

Reproducing a `from_csv` run from `metadata.json` re-reads the file by path, so
byte-identical reproduction needs that file unchanged.

## Transforms

Reshape a parent inside a `depends_on` entry, after the lag and before
standardizing — this is how a nonlinear or distributed-lag relationship is
planted instead of a plain weight.

```yaml
depends_on:
  - series: rainfall
    weight: 2.0
    transforms:
      - { name: threshold, params: { mode: hinge, threshold: 5.0 } }
```

| Name | Effect | Params |
|---|---|---|
| `threshold` | Nonlinear response: `hinge` (effect only above the threshold), `step` (binary switch), `quadratic` (U-shape). | `mode`, `threshold` |
| `distributed_lag` | Spreads the effect across a kernel of lags instead of one delay. Index *i* weights lag *i*. | `weights` (list) |
| `lag` | Delays by N periods. The warm-up becomes NaN. | `n` |
| `missing` | Blanks values at random, imitating reporting gaps. | `rate` |
| `block_missing` | Blanks contiguous runs — a reporting outage, versus `missing`'s scattered gaps. | `n_blocks`, `block_len` |
| `heavy_tail` | Adds Student-t noise: fat tails and outliers, to stress a model that assumes Gaussian. | `scale`, `df` (low = heavier) |

## Emitters

An emitter turns a series' finished values into its output column. The **block
name** selects it, so a series carrying `counts:` is a count series. Without
one, the series is written as floats.

| Name | Effect |
|---|---|
| `counts` | Population-relative incidence: sigmoid, scale by the location's population, draw whole counts. Fields under [`counts`](#counts). |

A new output type is one self-registering file — see the [how-to](HOW_TO.md).
`dsl list` shows what is registered.

## CLI

| Command | What it does |
|---|---|
| `dsl new [path]` | Write a commented starter scenario. `-f` overwrites. |
| `dsl run <scenario>` | Generate from a scenario YAML, or reproduce from a `metadata.json`. |
| `dsl list` | List the registered generators, transforms and emitters. |

### `dsl run` options

| Option | Default | Meaning |
|---|---|---|
| `-o`, `--out-dir DIR` | auto-named | Where to write. Omitted, an auto-named folder under `out/` keeps earlier runs intact. |
| `--plot` | off | Also write a plot. |
| `--plot-format FMT` | `html` | `html` (interactive), `png`, `svg`, `pdf`. |
| `--watch` | off | Re-run on every save; live-reloading plot with `--plot`. |
| `--new` | off | Skip the continue-or-new prompt. |
| `--replicates`, `-n N` | `1` | N seeded replicates (`base, base+1, …`) into `rep_00/`, `rep_01/`, …, each reproducible. Not with `--watch`. |

## Output

| File | When | Contents |
|---|---|---|
| `simulated_data.csv` | always | `time_period`, `location`, one column per series, `population`. |
| `folds/fold_N/{train,test}.csv` | with `split:` | One folder per fold, ready to use unsliced. |
| `folds/report.{md,json}` | with `split:` | Per-fold sizes, coverage, missing values, and every planted feature per side — the count, the exact periods it sits at, and which locations drew it. Plus a warning for any fold whose test half lacks a kind the scenario plants. |
| `metadata.json` | always | The ground truth: seed, lags, weights, transforms, rates, generators, version, resolved scenario. Feed back to `dsl run` to reproduce byte-for-byte. |
| `plot.{html,png,svg,pdf}` | with `--plot` | Faceted series over time, one line per location, the evaluation boundary marked. |

Same scenario + seed → identical files. Output is checked against the dataset
rules a forecasting tool expects; findings print as `warning:` and the run
still completes. Synthetic output always satisfies them — findings arise from
`from_csv` data with gaps or an odd shape:

- Required columns `time_period`, `location`, `disease_cases` (`population`
  optional; driver columns may have any name).
- `time_period` in a standard, parseable format for its resolution.
- Periods consecutive and identical across locations (advisory).
- No NaN in driver columns beyond a leading lag warm-up. NaN in `disease_cases`
  is fine — a missing count is treated as masked, not zero.

## See also

- **[Tutorial](TUTORIAL.md)** — learn the tool by editing one scenario.
- **[Concepts](CONCEPTS.md)** — why the DAG, events and folds work this way.
- **[How-to](HOW_TO.md)** — add a generator, transform or emitter.
