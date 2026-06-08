# Bug log

Found during a QA bug-hunt on v1.4.0 (branch `qa/bug-hunt`). **All six are now
fixed**, each with a regression test so they can't return. Severity:
🔴 data corruption / silent wrong output, 🟡 missing guard / misleading,
⚪ cosmetic.

| # | Severity | Status | Regression test |
|---|---|---|---|
| 1 | 🔴 | Fixed | `test_variable_named_like_reserved_column_rejected` |
| 2 | 🔴 | Fixed | `test_start_period_aligns_from_csv_data` |
| 3 | 🟡 | Fixed | `test_from_csv_fixed_source_with_multi_location_warns` |
| 4 | 🟡 | Fixed | `test_duplicate_variable_names_rejected` |
| 5 | 🟡 | Fixed | `test_parse_period_date_range_week_gives_helpful_error` |
| 6 | ⚪ | Fixed | `test_extreme_weight_no_overflow_warning` |

## 🔴 1. A variable named `disease_cases` / `population` / `time_period` / `location` silently overwrites the built-in column

**What:** Variable names are not checked against the reserved output column
names. A variable named `disease_cases` replaces the real disease signal in
the output; one named `population` overwrites the population column; etc. The
real data is silently lost — no error, no warning.

**Reproduce:**
```python
from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
df = run(parse_config({
    "period": "weekly", "n_total": 6,
    "variables": [{"name": "disease_cases", "generate": "seasonal_spike"}],
    "disease_cases": {"population": 1000,
                      "depends_on": [{"variable": "disease_cases", "lag": 1}]},
}))
# df["disease_cases"] holds the GENERATOR output, not the disease counts —
# the actual disease signal is gone; columns are only 4 wide.
```

**Fix:** In `ScenarioConfig._check_cross_section` (schema.py), reject any
variable whose `name` is in the reserved set
`{"time_period", "location", "disease_cases", "population"}` with a clear
error. Add tests for each reserved name.

**Fixed:** `_check_cross_section` now rejects variable names that match reserved columns.

## 🔴 2. `start_period` + `from_csv` misalign labels and real data

**What:** When a scenario sets `start_period` (e.g. `2010-04`) and a variable
uses `from_csv`, the output rows are LABELLED from `start_period` but the
`from_csv` values are read from the CSV's row 0. So a row labelled `2010-04`
can contain the CSV's January value — the time label and the real measurement
disagree. A data-integrity problem for semi-synthetic datasets.

**Reproduce:** CSV `q.csv` with months 2010-01..2010-06, rainfall 1..6 for
location A. Scenario `start_period: 2010-04`, `n_total: 3`, rainfall via
`from_csv(source_location=A)`. Output `time_period` = [2010-04, 2010-05,
2010-06] but `rainfall` = [1, 2, 3] (the CSV's Jan–Mar).

**Fix:** Make `from_csv` honor the scenario's `start_period`: when set, slice
the CSV from the matching label (it already supports a per-generator
`start_period` param — wire the scenario-level one through, or have the engine
pass it). Alternatively, error if `start_period` is set but a `from_csv`
column can't be aligned to it. Decide one behavior and test the alignment.

**Fixed:** the engine injects the scenario `start_period` into a `from_csv` variable's params (unless it sets its own), so real values align with the labels.

## 🟡 3. Multi-location `from_csv` silently duplicates one source across locations

**What:** With several DSL `locations` but a `from_csv` variable pinned to one
`source_location`, every location receives the SAME (duplicated) real series,
with no warning. The user likely expects each DSL location to map to a
different CSV source location (or to be told this isn't supported).

**Reproduce:** CSV with locations A and B (different values). Scenario
`locations: [X, Y]`, rainfall via `from_csv(source_location=A)`. Output X and
Y both get A's identical values.

**Fix:** Either (a) support mapping each DSL location to a CSV source location
(e.g. `source_location` per location, or auto-match by name), or (b) at
minimum emit a `validate_scenario` warning when a `from_csv` variable has a
fixed `source_location` while the scenario has >1 location. Option (b) is the
small, safe fix; (a) is the feature. Test the warning.

**Fixed:** `validate_scenario` warns when a `from_csv` variable has a fixed `source_location` while the scenario has >1 location.

## 🟡 4. Duplicate variable names are silently accepted

**What:** Declaring two variables with the same `name` is not rejected; the
second overwrites the first in the output (dict collision), losing one series.

**Reproduce:**
```python
run(parse_config({
    "period": "weekly", "n_total": 4,
    "variables": [
        {"name": "rainfall", "generate": "seasonal_spike"},
        {"name": "rainfall", "generate": "seasonal_smooth"},
    ],
    "disease_cases": {"population": 1000,
                      "depends_on": [{"variable": "rainfall", "lag": 1}]},
}))  # accepted; only one 'rainfall' column survives
```

**Fix:** In `_check_cross_section`, raise if `[v.name for v in variables]` has
duplicates (same style as the existing duplicate-location-names check). Test.

**Fixed:** `_check_cross_section` now rejects duplicate variable names.

## 🟡 5. `parse_period` rejects CHAP's date-range weekly form that `chap_check` accepts

**What:** `chap_check.validate_chap` treats `YYYY-MM-DD/YYYY-MM-DD` (CHAP's
date-range weekly label) as a valid period format, but
`periods.parse_period` raises on it. So a `start_period` in that form would be
rejected even though the validator blesses the format — an internal
inconsistency. (The DSL never *generates* that form, so impact is low.)

**Reproduce:**
```python
from dsl.core.pipeline.periods import parse_period
parse_period("2003-12-29/2004-01-04", "weekly")  # raises ValueError
```

**Fix:** Either teach `parse_period` to parse the date-range weekly form
(compute the year + week offset from the start date), or document that the
DSL's own period labels are the canonical set and the date-range form is
accepted only on ingested data, not as `start_period`. Pick one; add a test.

**Fixed:** `parse_period` gives an explicit error for the date-range week form, pointing to the canonical `YYYY-Wnn`.

## ⚪ 6. `exp` overflow RuntimeWarning with extreme driver weights

**What:** A very large `weight` (e.g. 1000) on a dependency makes the sigmoid
input huge, so `np.exp(-shifted)` overflows and numpy emits
`RuntimeWarning: overflow encountered in exp`. The OUTPUT is still correct
(the sigmoid saturates to ~1, counts stay valid and capped), but the warning
leaks to the user and looks like something broke.

**Reproduce:**
```python
import warnings; warnings.simplefilter("error")
run(parse_config({
    "period": "weekly", "n_total": 52,
    "variables": [{"name": "rainfall", "generate": "seasonal_spike"}],
    "disease_cases": {"population": 1000,
        "depends_on": [{"variable": "rainfall", "lag": 1, "weight": 1000.0}]},
}))  # RuntimeWarning: overflow encountered in exp
```

**Fix:** In `disease.py`, compute the sigmoid in an overflow-safe way — e.g.
`np.clip(shifted, -700, 700)` before `np.exp`, or a numerically-stable
sigmoid (`np.where(shifted >= 0, 1/(1+exp(-s)), exp(s)/(1+exp(s)))`). Clipping
is the smallest fix. Test that an extreme weight produces no warning and valid
counts.

**Fixed:** the sigmoid input is clipped to ±700 before `np.exp` (no numerical change, no warning).

## Checked and OK (non-bugs)

These were probed and behaved correctly — recorded so they aren't re-checked:

- `n_total=1` → correctly rejected when `lag >= n_total`.
- `population=1`, `missing_rate=1.0`, `yearly` with `n_total=2`, `daily` with
  `n_total=400`, `autoregressive` on `n_total=2`, `negative_binomial` with
  tiny `overdispersion` → all produce valid, non-negative, integer counts
  capped at population.
- Location name containing a comma → CSV quoting handles it; re-reads fine.
- `train_fraction=0.9` with `n_total=3` → train=2 / test=1, not empty.
- Period formatting and round-trip across all four resolutions → correct,
  including the 2000 leap-year daily boundary.
