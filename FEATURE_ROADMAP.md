# Feature Roadmap After Phase 9

Complete Phase 9 first: verify that a known driver-to-disease lag can be
recovered from generated data. Then add the features below one at a time.

All additions must preserve the new DSL's plugin architecture, validation,
causal lagging, CHAP-oriented output, and single seeded random generator.

## Current Implementation Audit

No roadmap item is fully implemented yet. Keep every checkbox below unchecked
until the complete feature and its tests are present.

| # | Status | What already exists |
|---|---|---|
| 1 | Implemented | `locations:` config field (default `["loc"]`); every output row carries a `location` column; multi-location data is stacked long-format with independent draws per location; train/test split is per location. Verified end-to-end against the minimalist CHAP example model. |
| 2 | Implemented | `chap_check.validate_chap(df)` checks the required trio (time_period, location, disease_cases), CHAP-parseable period formats (all four resolutions), NaN in covariates, and disease_cases sanity; per-location/consecutive-period mismatches are advisory. All findings are warnings (no hard-fail; synthetic output is always valid, so findings only arise from from_csv real-data gaps). Corrected after a chap-core source audit (was over-strict on population, covariate names, and daily/yearly periods). |
| 3 | Implemented | The `from_csv` generator backs a variable with real data; `examples/laos_real_climate_from_csv.yaml` uses bundled CHAP Laos data (rainfall, temperature) with synthetic disease on top. Real population is still out of scope (population is a config scalar). |
| 4 | Implemented | `from_csv` reads any CHAP-format CSV (`file`, `column`, `source_location`, `start_period` params); insufficient data is a hard error, never wrapped or extrapolated. |
| 5 | Implemented | `metadata.py` writes `metadata.json` beside every dataset: seed, lags, weights, rates, generators, tool version, and the full resolved scenario (round-trips through parse_config to reproduce the run). |
| 6 | Not implemented | Drivers enter the predictor linearly; the final sigmoid does not provide configurable nonlinear driver effects. |
| 7 | Not implemented | Each dependency supports one fixed lag, not an effect distributed across several lags. |
| 8 | Not implemented | Multiple drivers are added independently; they do not interact. |
| 9 | Implemented | `count_distribution: negative_binomial` (with `overdispersion`) draws overdispersed counts via a gamma-Poisson mixture; default stays `poisson` (byte-identical to before). |
| 10 | Partial | Random missing disease cases are supported, but climate gaps, consecutive gaps, and reporting delays are not. |
| 11 | Not implemented | There are no outbreak or regime-change components. |
| 12 | Partial | `seasonal_spike`, `seasonal_smooth`, plus the nonseasonal `flat` (control/decoy) and `linear_trend` (drift/confounder) generators exist; sparse-event and uniform-range generators still missing. |
| 13 | Implemented | Top-level `start_period` config field (e.g. `"2010-07"`, `"2015-W10"`, `"20100615"`, `"2003"`) sets where the series starts on the real calendar, for all four resolutions; validated against the scenario's resolution. Default remains the first period of 2000. |
| 14 | Partial | One scenario is reproducible from its seed, but there is no command for generating multiple replicates. |
| 15 | Not implemented | The CLI runs one scenario YAML at a time. |
| 16 | Implemented | `plot.py` + `dsl run --plot` writes a faceted plotly chart (one panel per variable, one line per location, train/test boundary marked); interactive HTML by default, or static png/svg/pdf via `--plot-format` (kaleido). |
| 17 | Not implemented | GeoJSON is neither read nor written. |
| 18 | Not implemented | Covariate data must be downloaded to a CSV manually; nothing pulls from chap-core datasets or a CHAP server. |
| 19a | Implemented | The `locations` mapping form (`{Bokeo: {population: 75000}, ...}`) sets per-location population; the list form keeps one shared population. Drives both the output column and the incidence model/cap per location. |
| 19b | Not implemented | Locations cannot override generator params (per-location climate shape). |
| 20 | Not implemented | Real seasonal temperature is asymmetric (fast rise to an April peak, slow decline); `seasonal_smooth` is a symmetric sine, so the minimum lands in the wrong season. |
| 21 | Implemented | `clamp_min` param on both synthetic generators floors the series (e.g. 0 for rainfall, which noise could otherwise push negative). |
| 22 | Not implemented | Generators add only independent Gaussian noise; no serial correlation (AR/ARIMA). |
| 23 | Not implemented | Seasonality is a single yearly cycle; no multi-year/ENSO component. |
| 24 | Not implemented | No tooling compares synthetic output to a real reference's statistics. |
| 25 | Not implemented | Covariates cannot be combined into interaction terms before the disease model. |
| 26 | Implemented | `dsl run` accepts a `metadata.json` to reproduce a previous dataset, and with no `-o` writes into an auto-named, non-overwriting folder under `out/` (`out/<scenario>/`, then `_1`, `_2`, …). |
| 27 | Not implemented | `from_csv` hard-stops when the CSV is shorter than `n_total`; there is no way to extend a short real series with synthetic periods. |
| 28 | Implemented | `population` can be a generator (`{generate: linear_trend, ...}`) producing a per-period series, at the top level and per-location, so population changes over time; a plain int stays constant. Plotted only when it varies. |

## Prioritized Features

- [x] **1. Add CHAP location support**

  Add a `location` column and support generating data for one or more locations.
  This is needed for current CHAP datasets and spatial evaluation.

  Source: [CHAP data preparation](https://chap.dhis2.org/chap-modeling-platform/external_models/prepare_data/)

- [x] **2. Validate CHAP output**

  Check required columns, period values, locations, data types, and missing
  values before writing output. This catches incompatible datasets early.

  Source: [CHAP data preparation](https://chap.dhis2.org/chap-modeling-platform/external_models/prepare_data/)

- [x] **3. Add real-data-backed covariates**

  Use existing CHAP rainfall, temperature, and population as covariates while
  generating fresh synthetic disease cases. This enables realistic
  semi-synthetic experiments.

  This feature must not claim to continue, extend, or forecast the source CHAP
  dataset beyond its available dates.

  Old-system references:
  [rainfall generator](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/rainfall/RealisticRainfallGenerator.py),
  [temperature generator](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/temperature/RealisticTemperatureGenerator.py),
  [population generator](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/population/RealisticPopulationGenerator.py)

- [x] **4. Add generic CSV-backed covariates**

  Load covariates from any compatible CSV instead of depending on one bundled
  dataset. This supports more locations and studies.

  Source: [CHAP data preparation](https://chap.dhis2.org/chap-modeling-platform/external_models/prepare_data/)

- [x] **5. Save ground-truth metadata**

  Save the seed, true lags, weights, rates, generators, and disease settings
  beside each dataset. This makes experiments reproducible and auditable.

  Old-system reference:
  [dataset-suite script](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/scripts/generate_dataset_suite.py)

- [ ] **6. Add nonlinear disease relationships**

  Support curved or optimum-range driver effects instead of only linear
  weighted effects. Many climate-health relationships are nonlinear.

  Old-system reference:
  [ClimateDependentDiseaseCases.py](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/diseases/ClimateDependentDiseaseCases.py)

- [ ] **7. Add distributed lags**

  Allow one driver to affect disease over several later periods. Exposure
  effects are often spread across time rather than fixed to one lag.

  New research feature.

- [ ] **8. Add interaction effects**

  Allow one driver's effect to depend on another, such as rainfall having a
  stronger effect at high temperatures. This represents combined climate
  effects.

  New research feature.

- [x] **9. Add overdispersed disease counts**

  Add a Negative Binomial option alongside Poisson generation. Surveillance
  counts often vary more than a Poisson model permits.

  New research feature.

- [ ] **10. Add realistic missing-data models**

  Support missing climate values, consecutive gaps, and reporting delays.
  Real missingness is not always independent and random.

  New research feature.

- [ ] **11. Add outbreaks and regime changes**

  Support sudden outbreaks, trend changes, and unusual seasons. This tests
  models when the data-generating process changes.

  New research feature.

- [ ] **12. Add more synthetic generators**

  Add sparse events, uniform-range values, and nonseasonal random series. These
  isolate effects that seasonal generators can hide.

  Old-system references:
  [sparse rainfall](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/rainfall/SyntheticRainfallGenerator.py),
  [uniform rainfall](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/rainfall/SyntheticRainfallUniformGenerator.py),
  [random temperature](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/temperature/SyntheticTemperatureGenerator.py)

- [x] **13. Add a configurable start period**

  Let scenarios start at a selected date instead of always starting in 2000.
  This aligns synthetic and real study periods.

  New CHAP usability feature.

- [ ] **14. Generate multiple seeded replicates**

  Generate repeated versions of a scenario using controlled seeds. This shows
  whether evaluation results are robust to random variation.

  Source: [CHAP evaluation workflow](https://chap.dhis2.org/chap-modeling-platform/chap-cli/evaluation-workflow/)

- [ ] **15. Generate scenario suites**

  Generate combinations of dependencies, lags, generators, and autoregressive
  settings. This avoids manually maintaining many scenario files.

  Old-system references:
  [ConfigGenerator.py](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/config/config_variations/ConfigGenerator.py),
  [ConfigParameters.py](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/config/config_variations/ConfigParameters.py)

- [x] **16. Add dataset visualization**

  Plot generated variables, disease cases, and train/test boundaries. Visual
  inspection makes incorrect scenarios easier to detect.

  Old-system reference:
  [ClimateHealth.py](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/ClimateHealth.py)

  CHAP source:
  [evaluation workflow](https://chap.dhis2.org/chap-modeling-platform/chap-cli/evaluation-workflow/)

- [ ] **17. Support matching GeoJSON**

  Accept or write region geometry matching generated locations. This enables
  spatial maps and location-aware inspection.

  Source: [CHAP data preparation](https://chap.dhis2.org/chap-modeling-platform/external_models/prepare_data/)

- [ ] **18. Auto-pull covariate data from CHAP**

  Fetch real covariate data directly from a CHAP source (the chap-core
  bundled datasets or a CHAP server API) instead of requiring a manually
  downloaded CSV. Builds on the CSV-backed covariates (#3/#4): a fetch step
  that produces the CSV, keeping the generator itself file-based and
  chap-core out of the core dependencies (optional extra or separate
  command, e.g. `dsl fetch`).

- [x] **19a. Per-location population**

  Each location can set its own `population` via the mapping form of
  `locations:` (`locations: {Bokeo: {population: 75000}, ...}`); the plain
  list form keeps using the top-level population. The mapping structure is
  built so #19b extends it (more override keys) rather than replacing it.

- [ ] **19b. Per-location generator params**

  Let a location also override generator params (e.g. a wetter rainfall
  spike for one province), by allowing more keys inside each location's
  override block introduced in #19a. Lower priority than population, and
  `from_csv` with `source_location` already supplies real per-location
  climate.

- [ ] **20. Asymmetric seasonal generator**

  A seasonal shape with a fast rise and slow decline (or vice versa). Real
  Laos temperature climbs steeply Jan→April, then falls slowly to a
  December minimum; the symmetric sine of `seasonal_smooth` cannot place
  peak and trough independently.

- [x] **21. Clamp generator output ranges**

  A `clamp_min` param on the synthetic generators flooring the series.
  Found while mimicking the Laos subset: rainfall noise produced negative
  millimetres. Set `clamp_min: 0` for any quantity that cannot go below
  zero.

- [ ] **22. Autocorrelated climate noise (AR generator)**

  A generator producing a serially-correlated series (AR(1)/ARIMA:
  `X_t = phi·X_{t-1} + e_t`), optionally over a seasonal baseline. Real
  rainfall/temperature have week-to-week persistence that the current
  independent Gaussian noise on `seasonal_smooth`/`seasonal_spike` lacks.
  New file in `generators/`.

- [ ] **23. Multi-year (ENSO-like) cycle**

  An optional low-frequency component (period ~3–7 years) on the seasonal
  generators, for interannual variability in long datasets. Either a param
  on the existing generators or a composable second cycle.

- [ ] **24. Realism-validation tooling**

  A command/report comparing a generated dataset against a real reference
  (e.g. the bundled Laos data): mean/variance ratio (overdispersion),
  lag-1 autocorrelation, seasonal peak-to-trough asymmetry. Tells a user
  whether their synthetic data is realistic. Overlaps with #16 (plotting)
  and reuses the kind of stats already written ad hoc for the Laos mimicry.

- [ ] **25. Covariate-interaction transform**

  A transform multiplying two covariate series pointwise (e.g.
  `rainfall × mean_temperature`) so the product can itself be a driver.
  This is the *generation-side* framing of interactions; #8 covers
  interactions inside the disease model. Decide which framing to keep.

## External review (deep-research report, 2026-06)

A literature/CHAP deep-research pass (`deep-research-report.md`) reviewed the
disease model and covariate generation. Its findings, reconciled against the
items above:

- **Confirms the priority of** #9 (negative-binomial overdispersion), #6
  (nonlinear/threshold climate effects, e.g. thermal-suitability curves),
  #7 (distributed lags / DLNM), and #10 (realistic missingness) as the
  literature-backed high-value tier — do these before the spatial/ENSO work.
- **Maps onto existing items:** report's interactions → #8/#25, outbreaks →
  #11, asymmetric/multi-harmonic seasonality → #20, spatial correlation +
  population heterogeneity → #19.
- **Genuinely new → added above:** #22 (AR climate noise), #23 (ENSO),
  #24 (realism validation), #25 (interaction transform).
- **Validator audit (acted on):** the report flagged `chap_check` as
  over-strict; verified against chap-core source (`time_period/
  date_util_wrapper.py`, `datatypes.py`, `validators.py`) and corrected —
  population is optional, covariate names are free-form, and all four
  resolutions (daily/weekly/monthly/yearly, plus CHAP's weekly variants)
  are accepted. Per-location/consecutive-period checks kept as advisory.
- **Report caveat:** it never saw this roadmap (it assumed an empty
  placeholder), so its "no conflicts" conclusion was unreliable; the
  reconciliation above was done by hand.

- [x] **26. Reproduce from metadata + non-overwriting output folders**

  `dsl run` accepts a `metadata.json` as input and regenerates the dataset
  from its embedded scenario (no original YAML needed). When `-o` is omitted,
  output goes to an auto-named folder under `out/` (first `out/<scenario>/`,
  then `_1`, `_2`, …) so earlier runs are never silently overwritten;
  explicit `-o` still writes directly to the given directory.

- [ ] **27. Extend short real data with synthetic periods (real+synthetic mix)**

  When a `from_csv` covariate has fewer periods than `n_total` (e.g. 10 real
  years, scenario needs 12), optionally fill the remainder with generated
  data instead of hard-stopping.

  **Considered, deliberately deferred** — this is methodologically loaded, not
  just an engineering task:
  - Appending synthetic to real creates a **seam**: the real years carry real
    autocorrelation/seasonality/variance; the synthetic tail won't match, so
    there's a discontinuity at the join that a model is then partly evaluated
    on (an artifact we introduced). This can be worse than pure-real or
    pure-synthetic.
  - The whole value proposition is *known ground truth*; mixing weakens the
    honesty of "this is real climate" vs "this is controlled synthetic."
  - Three possible approaches, increasing soundness: (a) tile/repeat real
    years — cheap but duplicates and is memorizable; (b) concatenate a
    generator's output — easiest, worst seam; (c) fit a model to the real
    years and simulate forward (cf. chap-core `climate_predictor.py`) —
    principled, smooth join, but the extension is *modelled*, not measured.
  - If built, the right shape is a **transform that combines two series**, and
    metadata MUST record which periods are real vs extrapolated so the dataset
    stays self-describing.
  - Usual better answers: generate fewer periods, use a longer real dataset,
    or go fully synthetic tuned to resemble the real one
    (`laos_fully_synthetic.yaml` already does this).

- [x] **28. Time-varying population (population via a generator)**

  Let `population` be produced by any generator (e.g. `linear_trend` for
  growth, `from_csv` for a real trajectory) instead of only a fixed scalar,
  so population can change over time. Backward compatible: a plain int still
  means a constant population. Per-location (extends #19a): each location may
  carry its own population generator, so e.g. urban provinces grow faster
  than rural ones.

  Key points: population is resolved to a length-`n_total` array and threaded
  through the disease model (its `× population` and the count cap already
  broadcast element-wise) and the output column. A constant population — or a
  noise-free generator like `linear_trend` — must consume no RNG, keeping
  existing scalar scenarios byte-identical.

## Rules For Every Feature

- Add tests in the same change as the feature.
- Use the run's seeded `numpy.random.Generator` for every random draw.
- Add ground-truth recovery tests when disease relationships change.
- Keep generator and transform additions inside their extension zones.
- Preserve CHAP-compatible output and existing reproducibility guarantees.

## Do Not Copy From The Old System

- Factory classes and fixed variable-type enums
- Global `numpy.random` calls
- Circular lagging with `numpy.roll`
- Redundant `week` or `month` output columns
- Hard-coded Brazil-only behavior
- Unused or partially implemented configuration flags
