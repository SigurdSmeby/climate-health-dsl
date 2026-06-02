# Recommended Synthetic-Data Features

Below we list recommended features to add to the DSL, each with epidemiological rationale (with citations), mathematical formulation, and where it would fit in the plugin architecture (generator, transform, or disease-model option). We then prioritize the features by utility and effort.

- **1. Overdispersion (Negative Binomial counts).** Real disease counts often show variance above the Poisson mean (due to unobserved heterogeneity or superspreading).  *Rationale:* Modeling counts as **Negative Binomial (NB)** allows an extra “dispersion” parameter \(k\) so that \(\mathrm{Var}(Y)=\mu + \mu^2/k\) instead of \(\mu\). This better captures outbreaks or heavy-tailed incidence seen in epidemics.  *Formulation:*  If the Poisson rate is \(\lambda_{t,\ell}\), use NB: 
  \[
    P(Y=y) = \frac{\Gamma(y + k)}{y!\,\Gamma(k)}\Big(\frac{\lambda}{\lambda + k}\Big)^y \Big(\frac{k}{\lambda + k}\Big)^k,\quad \mathrm{Var}(Y)=\lambda + \lambda^2/k.
  \]
  *Plugin fit:* Add a disease-model option (e.g. `count_distribution: negative_binomial`) that draws cases from NB with user-specified or default dispersion \(k\) (instead of Poisson).  

- **2. Nonlinear/Threshold Climate Effects.** Many vector-borne diseases peak at certain “optimal” temperatures or rainfall levels rather than increasing linearly.  *Rationale:* For example, mosquito-borne transmission often follows a unimodal (“thermal response”) curve in temperature. Likewise, extreme rainfall may wash out breeding sites or cause nonlinear effects.  *Formulation:* Apply a nonlinear link or transform to covariates: e.g. logistic or Gaussian “bell-curve” response. For a covariate \(X_{t-\ell}\) (lagged climate), use a transformed effect \(f(X)\) such as 
  \[
    f(X) = \frac{1}{1 + \exp(-a (X - X_0))}, 
    \quad\text{or}\quad f(X) = \exp\Big(-\frac{(X - \mu)^2}{2\sigma^2}\Big).
  \]
  These capture thresholds or optima.  *Plugin fit:* Implement as a **transform** that replaces raw covariate \(X\) by \(f(X)\). The YAML could specify e.g. 
  ```
  transforms:
    - type: logistic_threshold
      input: mean_temperature
      midpoint: 25.0
      steepness: 0.5
  ```
  to yield a sigmoidal response around 25 °C.  *Citation:* Numerous studies (e.g. Mordecai *et al.*, *Science* 2019) document nonlinear thermal responses of dengue/malaria transmission.

- **3. Distributed Lags (Time-Smoothed Effects).** Climate effects on disease often accrue over many weeks (incubation, vector development, herd immunity) rather than a single lag.  *Rationale:* Distributed Lag Nonlinear Models (DLNMs) are standard in environmental epidemiology. For example, rainfall two months ago and three months ago both contribute to current cases.  *Formulation:* Convolve each climate series with a lag-weight kernel. If covariate \(X\) has lags \(X_{t-\ell}\) with weights \(w_\ell\), the total effect is \(\sum_\ell w_\ell X_{t-\ell}\). One can allow \(w_\ell\) to follow a shape (e.g. decaying exponential or Gaussian).  *Plugin fit:* Extend the disease model to allow **multiple lagged inputs** with a smooth weighting (or add a transform that computes a moving average). For example,  
  ```
  disease_model:
    type: poisson
    lag_weights:
      mean_temperature: {lags: [1,2,3,4], weights: [0.1, 0.3, 0.4, 0.2]}
  ``` 
  or a `lag_distribution` transform that auto-generates \(w_\ell\) (e.g. geometric decay).  *Citation:* Gasparrini *et al.* (2010, *EHP*) and others outline DLNMs for climate–disease links.

- **4. Climate Interactions.** Climate variables can interact. For instance, high temperature may amplify the effect of rainfall on mosquito breeding.  *Rationale:* Interaction (e.g. “rainfall × temperature”) can capture synergistic effects not seen in additive models.  *Formulation:* Include product terms: e.g. \(X_{t-\ell}\times Y_{t-\ell}\) or a more complex function.  *Plugin fit:* A **transform** that multiplies two covariate series pointwise. For example:
  ```
  transforms:
    - type: multiply
      inputs: [rainfall, mean_temperature]
      output: rainfall_times_temp
  ```
  and then `rainfall_times_temp` can be a covariate in the disease model. This lets users specify interactions without hardcoding in the model.  

- **5. Outbreak or Regime-Change Dynamics.** Real disease time series sometimes have abrupt outbreaks or phase shifts (e.g. epidemic vs interepidemic period).  *Rationale:* CHAP’s naive simulator only adds random noise, but real epidemics can spike above seasonal baseline due to nonlinear transmission dynamics or external introduction.  *Formulation:* Superpose a rare “outbreak” component: e.g. add to \(\lambda_t\) a term that is zero most times and occasionally large (a spike). This could be a Poisson(λ_outbreak) added at random times or a piecewise change-point that raises baseline for a period.  *Plugin fit:* Implement as either (a) a **generator** of occasional outbreak signals added to disease rate, or (b) a **transform** on the disease series (e.g. after generating baseline Poisson cases, occasionally multiply by a factor). For example:
  ```
  disease_model:
    outbreak:
      probability: 0.05
      magnitude: {distribution: lognormal, mean: 2, sd: 0.5}
  ```
  which injects outbreaks with probability 0.05. *Citation:* Standard epidemiological models (e.g. compartmental models) can exhibit regime changes; see Kermack–McKendrick (1927) or modern reviews.  

- **6. Seasonal Baseline Shape (Multiple Harmonics).** Instead of a pure sine, real disease seasonality can be asymmetric or multi-peaked (e.g. bimodal rainy-season peaks).  *Rationale:* CHAP’s simulator has only a single sinusoid. Empirical data often require higher-order Fourier terms or flexible seasonal curves.  *Formulation:* Use a Fourier series or custom seasonal generator: 
  \[
    S(t) = a_1 \sin(2\pi t/T) + b_1\cos(2\pi t/T)
         + a_2 \sin(4\pi t/T) + b_2\cos(4\pi t/T) + \dots
  \]
  *Plugin fit:* A **generator** (or transform) for seasonality allowing multiple harmonics or skewness. For example, let the user specify additional sine terms or an arbitrary seasonal template (perhaps loaded from real data).  

- **7. Autocorrelated Climate Noise.** Real climate covariates (rainfall, temperature) have autocorrelation (e.g. rain one week increases chance of rain next week).  *Rationale:* CHAP’s simple sine+spike ignores realistic weather patterns.  *Formulation:* Model covariates as AR(1) or ARIMA processes. E.g. 
  \[
    X_{t} = \phi X_{t-1} + \varepsilon_t, \quad \varepsilon_t\sim N(0,\sigma^2).
  \]
  *Plugin fit:* A **generator** for each climate series (beyond fixed sine) that generates an AR(1)/ARIMA series. YAML example:
  ```
  generators:
    - name: temperature_series
      type: AR1
      phi: 0.8
      sigma: 2.0
      baseline: {seasonal_sine: {amplitude: 5}}
  ```
  This maintains seasonal baseline with realistic “weather” variability. *Citation:* Time-series modeling of climate (e.g. Shumway–Stoffer) and many climate–disease studies use AR(1) for covariate generation.

- **8. Multi-Year Cycles (e.g. ENSO).** Some diseases show multi-year oscillations (e.g. El Niño impacts malaria/dengue).  *Rationale:* Including interannual variability increases realism, especially for long datasets.  *Formulation:* Add a low-frequency sine or AR term (e.g. period 4–7 years).  *Plugin fit:* Extend climate generators to include an optional low-frequency cycle:
  ```
  generators:
    - name: precipitation_series
      type: sine
      period: 52   # weeks (yearly)
      ...
      plus: 
         sine:
           period: 208  # 4-year ENSO cycle
           amplitude: 10
  ```
  or allow chaining two sine generators.  

- **9. Spatial Correlation / Population Heterogeneity.** In reality, neighboring locations have correlated climate and possibly spillover of infection.  *Rationale:* CHAP’s DSL currently generates each location’s climate and population independently. Real data show spatial correlation (e.g. a regional monsoon) and differing populations by region size.  *Formulation:* One can draw climate anomalies from a multivariate normal with spatial covariance, or share a base climate + independent noise. Similarly, sample populations from a distribution reflecting urban/rural differences.  *Plugin fit:* For climate, allow generators to specify that some covariates are common plus a random effect. For population, use a **generator** that assigns population values per location (e.g. drawn from a lognormal or provided list).  This might be a separate metadata input (locations with populations).  

- **10. Reporting Artifacts & Missing Data.** Surveillance data often have structured missingness (weekends, holidays), reporting delays, and under-reporting.  *Rationale:* Random NaNs are unrealistic: e.g. clinics may close on weekends or districts may systematically under-report.  *Formulation:* Introduce blocks of missing values and scaling factors. For example, let each location have a “reporting probability” \(p_\ell\) so that observed cases = Binomial(cases, \(p_\ell\)). Or impose that 1–2 weeks per year have no reports (set cases to NA or 0).  *Plugin fit:* A **transform** on the disease series to mask or down-weight values. For instance:
  ```
  transforms:
    - type: reporting_bias
      probability_missing: 0.1
      cluster_months: [8,9]   # wetter months with worse reporting
      underreport_factor: 0.5
  ```
  This would randomly drop 10% of weeks, more often in certain months, and halved counts to simulate under-reporting.  *Citation:* Reporting bias is well documented in infectious disease surveillance (WHO reports on dengue/malaria often note underreporting rates).

- **11. Benchmark Metadata & Replicates.** To evaluate model recovery of ground truth, it’s useful to generate multiple realizations and record true parameters.  *Rationale:* Forecasting-benchmark literature (e.g. epidemic forecast challenges) uses ensembles of scenarios.  *Formulation:* Save the scenario YAML plus random seed, and produce multiple replicates. Also record the true coefficients/lag-weights in metadata.  *Plugin fit:* Not a data transform but a *tooling feature*: automatically output a manifest (e.g. JSON) of the true DAG of effects, plus allow generating multiple datasets per scenario with different seeds.  

- **12. Realism Validation Tools.** Provide automated checks comparing synthetic vs real data (distribution, autocorrelation, seasonal profile) using CHAP’s plotting.  *Rationale:* To avoid unrealistic data, we can measure e.g. histograms of counts, ACFs, seasonal shapes and compare to reference (CHAP example_data).  *Formulation:* Compute summary stats (mean/variance ratio, autocorrelation at lag 1, peak-to-trough seasonal asymmetry).  *Plugin fit:* Possibly reuse CHAP’s plotting (e.g. periodograms) as a **validation step** after generation, but not part of data generation itself.  

Finally, we re-examine the DSL’s existing roadmap items. The attached `FEATURE_ROADMAP.md` currently has only a placeholder (no items yet implemented). None of the above suggestions appear to be in conflict, but the user’s planning should ensure we don’t duplicate or mis-order. (If “reporting delays” or “negative binomial” were already tentatively listed, our analysis confirms their importance.) 

**Prioritization (impact vs. effort):** High-priority features are (1) overdispersion, (2) nonlinear climate effects, (3) distributed lags, and (4) realistic missing-data mechanisms, since these strongly affect model recovery and are often seen in real data. Adding negative-binomial disease modeling and a flexible lag structure directly addresses common model weaknesses. Medium priority are (5) outbreak spikes/regime change and (6) richer seasonal shapes, which improve realism but may be less critical for routine forecasts. Lower priority (more complex) are (7) spatial correlation and (8) multi-year ENSO cycles, which are nice but can be overkill for a basic benchmark. The reporting/bias and population heterogeneity should be considered early if the target models are sensitive to those (especially multi-location comparisons).  

# CHAP Data Format Rules (Validator vs. Canonical) 

The table below compares each rule our validator currently enforces against the CHAP code/docs. Rows marked *Stricter* or *Looser* indicate differences:

| **Rule**                         | **Our Validator**                | **CHAP Requirement** (cited)                                                                                                                    |
|----------------------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **time_period format**           | YYYY-MM (monthly) or YYYY-Wnn (weekly) only. | CHAP *accepts* ISO weekly/monthly by default.  In fact CHAP’s parser also supports date-range weeks (“YYYY-MM-DD/YYYY-MM-DD” for weekly), daily (`YYYY-MM-DD`) and annual (`YYYY`) formats via `TimePeriod.parse`. Our validator is **stricter** than CHAP by disallowing those. |
| **location column present**      | Required.                        | Required by CHAP.  Every CSV row must have a `location`.                                                                                                                                         |
| **disease_cases column present**| Required.                        | Required.  CHAP mandates a `disease_cases` column for the target variable.                                                                                                                       |
| **population column**           | Required (we enforce it).        | Optional. CHAP’s docs state population is optional.  Indeed, CHAP will auto-create a dataclass without population if it’s absent. Our validator is **stricter** than necessary.             |
| **Covariate columns**            | We require (by default) `rainfall`, `mean_temperature`. | Flexible. CHAP uses whatever covariate names are present. (The CLI example shows `rainfall, mean_temperature` by convention, but CHAP’s `from_csv` will accept any extra columns as covariates.) Our validator should allow arbitrary covariate names or use a mapping. |
| **Consecutive periods (per loc)**| Enforced across each location (no gaps). | Enforced *locally*: by default `fill_missing=False`, CHAP requires each location’s series be contiguous. Our rule matches this.  (If `fill_missing=True` were used, CHAP would allow gaps by filling them.) |
| **Identical time range across locs** | Required (we enforced same start/end). | Not required. CHAP will **auto-fill** each location to the full global range.  Different start/end among locations is allowed. Our rule is **stricter** than CHAP’s behavior. |
| **No NaN in covariates**        | Enforced (we disallow).         | Not strictly required. CHAP’s parser can produce NaNs (if `fill_missing=True`, missing slots get NaN). It tolerates NaNs but models may not. CHAP does not state a rule forbidding NaN covariates, so our validator is **stricter** than CHAP. |
| **Missing disease_cases (forecast horizon)** | Allowed only at end (we permit NaNs in last periods). | CHAP generally expects disease history to evaluate a model; missing values after the training window (prediction horizon) are allowed. CHAP does not forbid NaNs in `disease_cases` for future periods (and in fact will align data with NaNs). This matches our assumption. |
| **Column naming (`rainfall`/`mean_temperature`)** | Fixed: we require exactly these names. | Free-form. CHAP allows any column names and can map them via a data-source mapping. Our rule is **stricter**; better to accept arbitrary covariate columns or allow a mapping file. |

Each row cites CHAP’s code or docs. Any rule marked *Stricter* means our validator disallows cases CHAP would accept; *Looser* (none above) would mean we allow something CHAP forbids (we have none). This table highlights where we should relax our rules (daily/weekly formats, population optional, covariate naming, missingness handling, and per-location time range). All changes should be aligned with the **CHAP canonical data model** (per `spatio_temporal_data` and docs), not just our example data quirks.

