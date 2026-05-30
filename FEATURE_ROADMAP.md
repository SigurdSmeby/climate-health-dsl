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
| 1 | Not implemented | Output has no `location` column or multi-location model. |
| 2 | Partial | Pydantic validates scenarios and tests check the current CSV header, but there is no complete CHAP dataset validator. |
| 3 | Not implemented | Both current generators create fresh synthetic covariates. |
| 4 | Not implemented | The loader reads scenario YAML, not covariate CSV files. |
| 5 | Partial | The scenario contains seed, lag, weight, and rate settings, but output does not save them as metadata. |
| 6 | Not implemented | Drivers enter the predictor linearly; the final sigmoid does not provide configurable nonlinear driver effects. |
| 7 | Not implemented | Each dependency supports one fixed lag, not an effect distributed across several lags. |
| 8 | Not implemented | Multiple drivers are added independently; they do not interact. |
| 9 | Not implemented | Disease counts use Poisson sampling only. |
| 10 | Partial | Random missing disease cases are supported, but climate gaps, consecutive gaps, and reporting delays are not. |
| 11 | Not implemented | There are no outbreak or regime-change components. |
| 12 | Partial | `seasonal_spike` and `seasonal_smooth` exist, but the listed sparse, uniform-range, and nonseasonal generators do not. |
| 13 | Partial | `format_period` accepts `start_year`, but scenarios and the engine always use the default year 2000. |
| 14 | Partial | One scenario is reproducible from its seed, but there is no command for generating multiple replicates. |
| 15 | Not implemented | The CLI runs one scenario YAML at a time. |
| 16 | Not implemented | Train/test CSV files exist, but no visualization is produced. |
| 17 | Not implemented | GeoJSON is neither read nor written. |

## Prioritized Features

- [ ] **1. Add CHAP location support**

  Add a `location` column and support generating data for one or more locations.
  This is needed for current CHAP datasets and spatial evaluation.

  Source: [CHAP data preparation](https://chap.dhis2.org/chap-modeling-platform/external_models/prepare_data/)

- [ ] **2. Validate CHAP output**

  Check required columns, period values, locations, data types, and missing
  values before writing output. This catches incompatible datasets early.

  Source: [CHAP data preparation](https://chap.dhis2.org/chap-modeling-platform/external_models/prepare_data/)

- [ ] **3. Add real-data-backed covariates**

  Use existing CHAP rainfall, temperature, and population as covariates while
  generating fresh synthetic disease cases. This enables realistic
  semi-synthetic experiments.

  This feature must not claim to continue, extend, or forecast the source CHAP
  dataset beyond its available dates.

  Old-system references:
  [rainfall generator](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/rainfall/RealisticRainfallGenerator.py),
  [temperature generator](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/temperature/RealisticTemperatureGenerator.py),
  [population generator](https://github.com/SigurdSmeby/climate_health_simulations/blob/9e8877637b1b7e50a4493adf4bd7a978ad1538a5/src/climate_health_simulations/simulator/population/RealisticPopulationGenerator.py)

- [ ] **4. Add generic CSV-backed covariates**

  Load covariates from any compatible CSV instead of depending on one bundled
  dataset. This supports more locations and studies.

  Source: [CHAP data preparation](https://chap.dhis2.org/chap-modeling-platform/external_models/prepare_data/)

- [ ] **5. Save ground-truth metadata**

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

- [ ] **9. Add overdispersed disease counts**

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

- [ ] **13. Add a configurable start period**

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

- [ ] **16. Add dataset visualization**

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
