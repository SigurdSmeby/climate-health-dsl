# Reference: scenario fields, generators, transforms

Every YAML field, generator, and transform, with types, defaults, and meanings. New to the DSL? Start with the [tutorial](TUTORIAL.md) instead — this page is for looking things up. See also: [how-to guides](HOW_TO.md) (extend the DSL), [concepts](CONCEPTS.md) (how the disease model works).

## Writing a scenario

A scenario is one YAML file. The bundled example (`examples/basic_scenario.yaml`):

```yaml
period: weekly
n_total: 78
seed: 42
train_fraction: 0.8
locations:
  loc:
    population: 100000
series:
  - name: rainfall
    generate: seasonal_spike
  - name: mean_temperature
    generate: seasonal_smooth
  - name: disease_cases
    counts: {}
    depends_on:
      - series: rainfall
        lag: 3
        weight: 1.0
      - series: mean_temperature
        lag: 3
        weight: 1.0
```

### Top-level fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `period` | `daily` \| `weekly` \| `monthly` \| `yearly` | required | Time resolution. Sets the period labels (`20000101`, `2000-W01`, `2000-01`, `2000`) and the length of one seasonal cycle (365/52/12/1). |
| `n_total` | int ≥ 1 | required | Number of time periods to generate. |
| `seed` | int | `0` | Seed for all randomness. Same scenario + same seed → identical output. |
| `train_fraction` | float, 0 < x < 1 | unset | If set, also write `train.csv`/`test.csv`. |
| `start_period` | str | first period of 2000 | Where the series starts on the real calendar, in the scenario's resolution: `"2010-07"` (monthly), `"2015-W10"` (weekly), `"20100615"` (daily), `"2003"` (yearly). Relabels the output but does **not** shift the seasonal *phase* — a mid-year start still begins the seasonal cycle at index 0 (the run warns when this applies). |
| `locations` | mapping (or list of str) | `["loc"]` | Named locations, each an independently drawn series of `n_total` periods, stacked in long format with a `location` column. The **mapping** form declares each location's population: `{Bokeo: {population: 75000}, ...}`, which may itself be a generator. Required as soon as any series counts, since a count series draws against the population where it happens. The bare **list** form is only for scenarios with no count series. |
| `events` | mapping | `{}` | Named shocks a series can react to — see below. |
| `series` | list | required | Every series the scenario builds, climate and disease alike — see below. |

### `series` entries

One list holds everything. A series is a plain float column unless it carries a
`counts:` block, which makes it a disease signal instead.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required | Becomes the output column name. For CHAP datasets use CHAP's names: `rainfall`, `mean_temperature`, `disease_cases`. |
| `generate` | str | optional | Which generator produces the series' base (see [Generators](#generators)). May be omitted when the series has `depends_on`, in which case its base is flat 0 and its parents give it all of its shape. |
| `params` | mapping | `{}` | Passed straight to that generator; each generator validates its own. |
| `depends_on` | list | `[]` | Parents this series is built from — any number, each with its own lag, weight and transforms (see below). A series may depend on any other series, so `rain -> soil_moisture -> disease` is expressible directly. Cycles of any length are rejected. |
| `events` | mapping | `{}` | Which declared events this series reacts to, and how: `{storm: {multiplier: 2.1}}`. |
| `missing_rate` | float, 0–1 | `0` | Fraction of this series' values randomly blanked — a failed gauge as readily as a reporting gap. |
| `autoregressive` | mapping | unset | AR(1) persistence: `{phi: 0.85, noise: 0.3}`. `phi` is the fraction of each period carried into the next (0 ≤ phi < 1; 1 would be a non-stationary random walk and is rejected). Natural on a physical store that carries over, such as soil moisture or a dam. |
| `counts` | mapping | unset | Makes this a count series, and holds the incidence model's parameters — see below. |

Adding a series (e.g. a decoy the disease does *not* depend on) needs no code, only YAML.

### `counts` fields

The presence of the block is what marks a disease signal; an empty `counts: {}`
means "use the defaults". Population is deliberately **not** here — it belongs
to the location, so several diseases cannot disagree about the same place.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_rate` | float, 0–1 | `0.3` | Ceiling on the incidence **rate**: the expected count never exceeds `max_rate × population`. Individual draws still scatter around that rate — only the population is a hard cap on a drawn value. |
| `median_rate` | float | `0.1` | Incidence in a typical period. Must be **smaller than** `max_rate`. |
| `distribution` | `poisson` \| `negative_binomial` | `poisson` | How counts are drawn from the rate. `poisson` has variance = mean; `negative_binomial` adds overdispersion (spikier, more realistic surveillance counts). |
| `overdispersion` | float > 0 | `10.0` | Only for `negative_binomial`: variance = mean + mean²/`overdispersion`, so **smaller = more variable**; large values approach Poisson. |

### `events`

A named shock that fires in specific periods and can strike **several series at
once** — the one thing a generator cannot do, since a generator sees only its
own series. Events are regional: drawn once and shared by every location, so a
storm hits the whole region in the same periods.

```yaml
events:
  storm: { rate: 0.05 }        # or: { at: [14, 47, 88] } for exact periods

series:
  - name: rainfall
    generate: seasonal_spike
    events: { storm: { multiplier: 2.1 } }   # up, proportionally
  - name: mean_temperature
    generate: seasonal_smooth
    events: { storm: { add: -4.0 } }         # down, by a fixed amount
```

An event declares exactly one of `rate` (each period independently, seeded) or
`at` (explicit periods, fully deterministic). Each reacting series gives exactly
one of `multiplier` (scales the series, so the effect grows with its level and
cannot move a series sitting at zero) or `add` (shifts it by a fixed amount
regardless of level). Both are signed.

Events do not replace `seasonal_spike`: a generator makes a *recurring* shape,
an event is a *one-off* shock, and a scenario normally uses both.

### `depends_on` entries

| Field | Type | Default | Meaning |
|---|---|---|---|
| `series` | str | required | Name of another declared series. |
| `lag` | int ≥ 0 | `0` | Delay in periods between the parent and its effect. |
| `weight` | float | `1.0` | Strength of the driver relative to the others (drivers are standardized first, so weights are comparable across units). |
| `transforms` | list | `[]` | Optional list of `{name, params}` transforms applied to this driver **after** the lag and **before** standardize — the way to plant nonlinear or distributed-lag relationships. See [Transforms](#transforms). |

## Generators

A series' `generate:` names the generator that produces its base. Built-ins:

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
series:
  - name: disease_cases
    counts: {}
    depends_on:
      - series: rainfall
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
