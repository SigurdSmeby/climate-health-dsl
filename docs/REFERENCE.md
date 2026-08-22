# Reference: scenario fields, generators, transforms

Every YAML field, generator, and transform, with types, defaults, and meanings. New to the DSL? Start with the [tutorial](TUTORIAL.md) instead — this page is for looking things up. See also: [how-to guides](HOW_TO.md) (extend the DSL), [concepts](CONCEPTS.md) (how the disease model works).

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
| `locations` | list of str, or mapping | `["loc"]` | Named locations, each an independently drawn series of `n_total` periods, stacked in long format with a `location` column. Use the **list** form (`[oslo, bergen]`) for one shared population, or the **mapping** form to set a per-location population: `{Bokeo: {population: 75000}, ...}`. A per-location population can itself be a generator. A location with no `population` falls back to `disease_cases.population`. |
| `variables` | list | required | The independent (climate) variables — see below. |
| `disease_cases` | mapping | required | How the dependent disease signal is built — see below. |

### `variables` entries

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required | Becomes the output column name. For CHAP datasets use CHAP's names: `rainfall`, `mean_temperature`. |
| `generate` | str | required | Which generator produces the series (see [Generators](#generators)). |
| `params` | mapping | `{}` | Passed straight to that generator; each generator validates its own. |
| `shared` | float, 0–1 | unset | Fraction of this variable's signal shared across **all** locations — a latent regional driver (confounder). `0`/unset → each location independent; `1` → identical across locations; in between → correlated. Lets you test whether a model attributes cross-location co-movement to the shared cause rather than a spurious location-to-location link. (Variance-preserving mix; single-location scenarios are unaffected.) |

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
| `transforms` | list | `[]` | Optional list of `{name, params}` transforms applied to this driver **after** the lag and **before** standardize — the way to plant nonlinear or distributed-lag relationships. See [Transforms](#transforms). |

## Generators

Each `variable` names a generator that produces its series. Built-ins:

### `seasonal_spike` — rainy-season shape

A low baseline with a pronounced, smooth Gaussian spike peaking at the same point every year (wrapping across the year boundary).

| Param | Default | Meaning |
|---|---|---|
| `baseline` | `2.0` | The value far from the spike (dry-season level). |
| `spike_height` | `20.0` | How far above the baseline the peak rises. |
| `spike_center` | mid-year | Period offset of the peak within the year. Unset → the middle of the year for the resolution (26 for weekly, 6 for monthly, etc.). |
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

`examples/confounders_and_controls.yaml` uses `flat` and `linear_trend` as decoys the disease ignores.

### `outbreak_shocks` — rare extreme events

A low baseline punctuated by rare, randomly-timed (Poisson) sharp spikes — the extreme-event / outbreak shape, distinct from the smooth annual `seasonal_spike`. No two years look alike.

| Param | Default | Meaning |
|---|---|---|
| `baseline` | `0.0` | Quiet-period level. |
| `noise` | `0.0` | Std. dev. of Gaussian noise on the baseline. |
| `rate` | `1.0` | Expected number of shock events per year. |
| `magnitude` | `20.0` | How far above baseline each shock rises. |
| `duration` | `1` | How many consecutive periods each shock elevates. |
| `clamp_min` | unset | Floor for the values. |

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

**CHAP validation warnings.** Every run checks its output against CHAP's dataset rules before writing; a violation prints a `warning:` and the run still completes. The synthetic generators always produce CHAP-valid output — these only arise from `from_csv` data with gaps or an unexpected shape:

- Required columns: `time_period`, `location`, `disease_cases` (`population` is optional; covariate columns may have any name).
- `time_period` must be in a resolution CHAP's own parser accepts.
- Periods should be consecutive and identical across locations (advisory — CHAP can auto-fill a gap, but a mismatch often signals a real problem in the source data).
- No `NaN` in covariate columns (`NaN` in `disease_cases` is fine — CHAP treats those as masked/missing case counts).

## Transforms

A transform modifies a series that already exists. They're applied to a driver inside a `depends_on` entry, via its `transforms` list, **after** the causal lag and **before** standardization — this is how you plant a *nonlinear* or *distributed-lag* relationship instead of a plain linear weight:

```yaml
disease_cases:
  depends_on:
    - variable: rainfall
      weight: 2.0
      transforms:
        - { name: threshold, params: { mode: hinge, threshold: 5.0 } }
```

### `threshold` — nonlinear driver response

Reshapes a driver around a threshold, for effects that aren't linear.

| Param | Default | Meaning |
|---|---|---|
| `mode` | `hinge` | `hinge` (effect only above the threshold: `max(0, x−t)`), `step` (binary switch: 1 at/above, else 0), or `quadratic` (U-shape around the threshold: `(x−t)²`). |
| `threshold` | `0.0` | The threshold / center value. |

### `distributed_lag` — spread over several lags

Convolves a driver with a causal weight kernel over lags 0..N, so its effect smears across a window instead of a single delay (the distributed-lag-nonlinear-model shape). Warm-up periods with no full past are blanked; never wraps.

| Param | Default | Meaning |
|---|---|---|
| `weights` | required | List of weights; index `i` is the weight at lag `i`, e.g. `[0.5, 0.3, 0.2]`. |

### `heavy_tail` — fat-tailed noise

Adds Student-t noise so a series has occasional outliers Gaussian noise can't produce — for stressing a model's robustness.

| Param | Default | Meaning |
|---|---|---|
| `scale` | `1.0` | Noise scale. `0` disables it. |
| `df` | `3.0` | Degrees of freedom; low = heavier tails, large approaches a Gaussian. |

### `block_missing` — contiguous reporting outages

Blanks whole contiguous runs to NaN, modelling a reporting outage — unlike `missing_rate`, which drops individual points independently.

| Param | Default | Meaning |
|---|---|---|
| `n_blocks` | `1` | Number of outage runs. |
| `block_len` | `4` | Length of each run in periods. |

### `lag` and `missing`

These underlie the built-in `depends_on[].lag` and `disease_cases.missing_rate` shortcuts and can also be named explicitly in a `transforms` list: `lag` (`n`) delays a series causally, `missing` (`rate`) blanks a random fraction of points.

## See also

- **[Concepts](CONCEPTS.md)** — how these fields combine to build the disease signal.
- **[How-to guides](HOW_TO.md)** — add a new generator or transform of your own.
- **[Tutorial](TUTORIAL.md)** — a hands-on walkthrough if you're starting out.
