# Bug log

QA bug-hunt on v1.4.0 (branch `qa/bug-hunt`). Bugs #1-12 are fixed, each with
a regression test. Rounds 4-5 found 26 additional bugs (#13-38), reported by Codex; all 26 were
**independently verified to reproduce**, then fixed (24) or documented as a
known limitation (#16 seasonal phase, #26 from_csv reproducibility). Severity: 🔴 data corruption / silent wrong output, 🟡 missing guard /
misleading, ⚪ cosmetic.

Round 1 (#1–6) — general input-validation hunt. Round 2 (#7–12) — a deeper
re-hunt plus a focused `from_csv` malformed-CSV stress test.

| # | Severity | Status | Regression test |
|---|---|---|---|
| 1 | 🔴 | Fixed | `test_variable_named_like_reserved_column_rejected` |
| 2 | 🔴 | Fixed | `test_start_period_aligns_from_csv_data` |
| 3 | 🟡 | Fixed | `test_from_csv_fixed_source_with_multi_location_warns` |
| 4 | 🟡 | Fixed | `test_duplicate_variable_names_rejected` |
| 5 | 🟡 | Fixed | `test_parse_period_date_range_week_gives_helpful_error` |
| 6 | ⚪ | Fixed | `test_extreme_weight_no_overflow_warning` |
| 7 | 🔴 | Fixed | `test_nan_covariate_blanks_disease_cases` |
| 8 | ⚪ | Fixed | `test_unknown_generator_param_clear_error` |
| 9 | 🔴 | Fixed | `test_from_csv_unsorted_periods_are_sorted`, `test_from_csv_duplicate_periods_rejected` |
| 10 | 🟡 | Fixed | `test_from_csv_requires_time_period_for_start` |
| 11 | 🟡 | Fixed | `test_from_csv_empty_file`, `test_from_csv_header_only` |
| 12 | ⚪ | Fixed | `test_from_csv_non_numeric_clear_error` |
| 13 | 🔴 | Fixed |
| 14 | 🔴 | Fixed |
| 15 | 🔴 | Fixed |
| 16 | 🔴 | Documented |
| 17 | 🟡 | Fixed |
| 18 | 🟡 | Fixed |
| 19 | 🟡 | Fixed |
| 20 | 🟡 | Fixed |
| 21 | 🟡 | Fixed |
| 22 | 🟡 | Fixed |
| 23 | 🔴 | Fixed |
| 24 | 🟡 | Fixed |
| 25 | 🔴 | Fixed |
| 26 | 🔴 | Documented |
| 27 | 🔴 | Fixed |
| 28 | 🟡 | Fixed |
| 29 | 🟡 | Fixed |
| 30 | 🟡 | Fixed |
| 31 | 🔴 | Fixed |
| 32 | 🟡 | Fixed |
| 33 | 🟡 | Fixed |
| 34 | 🟡 | Fixed |
| 35 | 🟡 | Fixed |
| 36 | 🟡 | Fixed |
| 37 | ⚪ | Fixed |
| 38 | ⚪ | Fixed |

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

---

# Round 2 (#7–12) — deeper re-hunt + `from_csv` stress test

## 🔴 7. A NaN covariate value produces a fabricated `disease_cases` count

**What:** If a `from_csv` covariate has a missing value (NaN) at some period,
that NaN flows into the disease model, where `build_disease_cases` does
`np.nan_to_num(eta, nan=0.0)` — turning the missing driver into eta=0, i.e.
the *average* predictor. So at a period whose real driver is **unknown**, the
model still emits a confident `disease_cases` count as if the covariate were
exactly average. The output row then shows a blank covariate but a definite
disease number built from invented data — a silent ground-truth violation.
(`nan_to_num` is also what zeroes the lag warm-up, which IS re-blanked at the
end; the covariate-NaN rows are not.)

**Reproduce:**
```python
import pandas as pd, numpy as np
from dsl.core.config.schema import parse_config
from dsl.core.pipeline.engine import run
pd.DataFrame({"time_period":[f"2010-{m:02d}" for m in range(1,13)],
              "rainfall":[1,2,np.nan,4,5,6,7,8,9,10,11,12]}).to_csv("withnan.csv", index=False)
df = run(parse_config({"period":"monthly","n_total":12,
    "variables":[{"name":"rainfall","generate":"from_csv",
                  "params":{"file":"withnan.csv","column":"rainfall"}}],
    "disease_cases":{"population":1000,"depends_on":[{"variable":"rainfall","lag":1}]}}))
# row 3 (the NaN month, after lag 1): rainfall is NaN but disease_cases is a
# normal number — computed as if rainfall were average.
```

**Fix:** In `build_disease_cases`, record which rows have NaN in `eta` BEFORE
`nan_to_num` (i.e. any period where a lagged driver is missing), and blank
those `disease_cases` rows to NaN at the end — same treatment as the lag
warm-up. A period with a missing input cannot have a known-ground-truth
output. Test: a from_csv covariate with a NaN yields NaN disease_cases at the
affected (post-lag) row, while other rows are unaffected.

**Fixed:** `build_disease_cases` records which `eta` rows are NaN (warm-up + missing driver) before `nan_to_num`, and blanks `disease_cases` at those rows. A period with a missing input now gets NaN, not a fabricated count.

## ⚪ 8. Unknown generator param raises a raw `TypeError`

**What:** Passing a param a generator doesn't accept (e.g.
`seasonal_spike` with `bogus_param: 99`) surfaces as a bare Python
`TypeError: __init__() got an unexpected keyword argument 'bogus_param'`
rather than a clear, user-facing message. It IS rejected (not silently
ignored), so this is cosmetic — but inconsistent with the friendly errors
elsewhere (bad *values* like `spike_width: -5` give a nice message).

**Reproduce:**
```python
run(parse_config({"period":"weekly","n_total":10,
    "variables":[{"name":"rainfall","generate":"seasonal_spike",
                  "params":{"bogus_param":99}}],
    "disease_cases":{"population":1000,"depends_on":[{"variable":"rainfall","lag":1}]}}))
# TypeError: SeasonalSpikeGenerator.__init__() got an unexpected keyword argument 'bogus_param'
```

**Fix:** In the engine, wrap the generator instantiation
(`get_generator(spec.generate)(**params)`) and re-raise an unexpected-keyword
`TypeError` as a clear error naming the variable, the generator, and the bad
param. Test that a bogus param gives the friendly message.

**Fixed:** the engine wraps generator instantiation (`_build_generator`) and re-raises an unexpected-keyword `TypeError` as a clear ValueError naming the variable, generator, and bad param.

## 🔴 9. Unsorted or duplicate `time_period` rows silently map real data to the WRONG periods

**What:** `from_csv` reads rows in FILE order and never sorts or deduplicates
by `time_period`. With `start_period`, it finds the start label's row
*position* and slices the next N rows in file order. So if the CSV is not
sorted by time (real exports often aren't), values are assigned to the wrong
periods — silently.

**Reproduce (off-by-a-month):**
```python
# CSV rows out of order: Mar, Jan, Feb, Apr
open("uns.csv","w").write("time_period,rainfall\n2010-03,30\n2010-01,10\n2010-02,20\n2010-04,40\n")
from dsl.generators.from_csv import FromCsvGenerator
import numpy as np
out = FromCsvGenerator(file="uns.csv", column="rainfall",
                       start_period="2010-01").generate(3, "monthly", np.random.default_rng(0))
# Asked for Jan,Feb,Mar -> got [10, 20, 40] = Jan, Feb, APRIL (March skipped!)
```
Duplicate periods are equally bad: a repeated `2010-01` row shifts everything
after it by one (`[10, 999, 20]` instead of `[10, 20, 30]`).

**Fix:** In `from_csv.generate`, after selecting the location and before
slicing, **sort by `time_period`** and **reject (or warn on) duplicate
periods**. Sorting must use period order, not string order — for the DSL's
label formats string sort is correct for monthly/weekly/yearly but NOT for
daily mixed widths; safest is to map labels through `parse_period`. Tests:
unsorted CSV yields correctly-ordered values; duplicate period raises.

**Fixed:** `from_csv` now sorts rows by `time_period` before slicing and rejects duplicate periods, so values always map to the correct period regardless of file order.

## 🟡 10. A CSV with no `time_period` column silently skips all alignment

**What:** If the CSV has no `time_period` column, `from_csv` skips the
resolution check AND `start_period` entirely (the `if "time_period" in
df.columns:` guard), then just takes the first N rows. So `start_period` is
silently ignored and a weekly scenario can read monthly data with no error.
Real CHAP data always has `time_period`; silently accepting data without it
removes every alignment safeguard.

**Reproduce:**
```python
open("notp.csv","w").write("date,rainfall\n2010-01,1\n2010-02,2\n2010-03,3\n2010-04,4\n")
FromCsvGenerator(file="notp.csv", column="rainfall",
                 start_period="2010-03").generate(2, "monthly", np.random.default_rng(0))
# returns [1, 2] — start_period 2010-03 silently ignored
```

**Fix:** Require a `time_period` column (error if missing), OR at least error
when `start_period` is set but there's no `time_period` to align to. Decide
whether headerless-time CSVs are supported at all; if yes, document that
alignment is skipped. Test the error.

**Fixed:** `from_csv` raises if `start_period` is set but the CSV has no `time_period` column to align to.

## 🟡 11. Empty / header-only CSV crashes with a raw pandas error

**What:** A completely empty file raises `pandas.errors.EmptyDataError`
("No columns to parse"); a header-only file (no data rows) raises
`IndexError: single positional indexer is out-of-bounds` from inside
`_check_resolution` (`df["time_period"].iloc[0]`). Neither is a clear
from_csv message.

**Reproduce:**
```python
open("empty.csv","w").write("")
FromCsvGenerator(file="empty.csv", column="rainfall").generate(3,"monthly",rng)  # EmptyDataError
open("ho.csv","w").write("time_period,rainfall\n")
FromCsvGenerator(file="ho.csv", column="rainfall").generate(3,"monthly",rng)     # IndexError
```

**Fix:** After `pd.read_csv` (wrap it to catch `EmptyDataError`), check the
frame is non-empty and raise a clear `from_csv: <file> has no data rows`.
Test both empty and header-only files.

**Fixed:** `from_csv` catches `EmptyDataError` and checks for zero data rows, raising a clear `has no data rows` error.

## ⚪ 12. Unparseable cell values raise raw numpy/pandas errors

**What:** Non-numeric content in the target column — text (`"heavy"`),
thousands separators (`"1,000"`), or a wrong delimiter (semicolon CSV, so the
column isn't found) — raises raw `ValueError: could not convert string to
float` / column-not-found, rather than a friendly "column X has non-numeric
value at period Y". Functionally safe (it does raise), just unpolished.

**Fix:** Coerce with `pd.to_numeric(..., errors="coerce")` and, if that
introduces NaNs that weren't blank cells, raise a clear error naming the bad
value/period. (Decide intended behavior for thousands separators — probably
reject.) Lower priority. Test a friendly message for text in the column.

**Fixed:** `from_csv` coerces with `pd.to_numeric` and raises a clear error naming the non-numeric value and column.

## Round 3 — `from_csv` inputs that behaved correctly

These malformed CSVs were handled fine: extra unrelated columns, reordered
columns, integer-typed column, surrounding whitespace, quoted numbers, empty
cells (→ NaN, but see bug #7), BOM prefix, trailing blank lines, negative
values, scientific notation. Resolution mismatch and too-short data are
correctly rejected (existing behavior).

## Round 2 — checked and OK (non-bugs)

Probed and correct:
- Reproducibility: same config run twice, and fresh re-parses, are identical;
  a complex everything-at-once scenario (per-location pop generator,
  start_period, AR, negative_binomial, decoy, clamp, missing) round-trips
  byte-identically through metadata.
- **Ground truth:** changing a dependency's `weight` changes `disease_cases`
  but leaves the covariates identical (same seed); changing `seed` changes
  output; adding a covariate does not shift another covariate's draws.
- Transforms: `lag`/`missing` don't mutate input; pre-existing NaN preserved;
  rate 0.0/1.0 correct; negative lag rejected.
- Population generator reaching 0 → 0 cases (correct); negative weights and
  negative `linear_trend` covariates produce valid counts.
- Train/test split per location is time-ordered, leak-free, counts add up,
  column order consistent across all three CSVs.
- CLI: spaces in scenario names, auto-numbered folders, nested `-o` paths,
  plain-JSON-as-scenario, bad subcommand all handled.
- Plot with `n_total=2` and with an all-NaN disease column don't crash.
- `from_csv` with a constant (zero-variance) column → no div-by-zero;
  resolution mismatch rejected; daily `n_total=5000` fast and valid.

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

---

# Round 4 (#13-22) - post-fix re-hunt

## 🔴 13. Missing `from_csv` periods are silently shifted onto the wrong labels

**What:** `from_csv` sorts and deduplicates `time_period`, but never checks
that the selected rows are consecutive. The engine then discards the source
labels and creates its own consecutive labels. If February is absent from the
CSV, March's value is silently written on the row labelled February, April's
value is written as March, and so on. `validate_chap` cannot detect this
afterward because the generated labels themselves are consecutive.

**Reproduce:**
```python
# gaps.csv:
# time_period,rainfall
# 2010-01,10
# 2010-03,30
# 2010-04,40
config = parse_config({
    "period": "monthly", "n_total": 3, "start_period": "2010-01",
    "variables": [{"name": "rainfall", "generate": "from_csv",
                   "params": {"file": "gaps.csv", "column": "rainfall"}}],
    "disease_cases": {"population": 1000,
                      "depends_on": [{"variable": "rainfall"}]},
})
run(config)[["time_period", "rainfall"]]
# 2010-01 -> 10, 2010-02 -> 30, 2010-03 -> 40
# The real March and April values have been relabelled as February and March.
```

**Fix:** After sorting and applying `start_period`, parse the selected source
labels and require every adjacent pair to be exactly one scenario period
apart. Raise a clear error naming the first gap. Add monthly, weekly, daily,
and yearly gap tests.

## 🔴 14. A missing value in an otherwise constant driver still fabricates cases

**What:** Bug #7's fix is bypassed for a constant covariate. `_standardize`
returns `np.zeros_like(series)` when the non-missing values have zero
variance. That also replaces the original NaN with zero, so `eta` no longer
records the missing input and a definite `disease_cases` count is generated
for the unknown period.

**Reproduce:**
```python
# constant_nan.csv has rainfall [5, NaN, 5, 5].
config = parse_config({
    "period": "monthly", "n_total": 4, "start_period": "2010-01",
    "variables": [{"name": "rainfall", "generate": "from_csv",
                   "params": {"file": "constant_nan.csv",
                              "column": "rainfall"}}],
    "disease_cases": {"population": 1000,
                      "depends_on": [{"variable": "rainfall", "lag": 0}]},
})
df = run(config)
# df.loc[1, "rainfall"] is NaN, but df.loc[1, "disease_cases"] is a number.
```

**Fix:** In `_standardize`, preserve the input NaN mask in the zero-variance
branch: return zeros only at valid positions and NaN at missing positions.
Add regression tests for constant drivers with missing values at lag 0 and at
a positive lag.

## 🔴 15. `from_csv.start_period` can disagree with output labels, including for population

**What:** Calendar alignment is only injected for ordinary `from_csv`
variables that do not already set their own `start_period`.

Two advertised configurations therefore silently mislabel real data:

- A generated population using `from_csv` never receives the scenario
  `start_period`. A scenario labelled March-April can use January-February
  population values.
- A variable-level `params.start_period` can differ from the scenario
  `start_period`, or be used without a scenario `start_period`. The selected
  real values are then written under unrelated generated labels.

**Reproduce:**
```python
# pop.csv contains 2010-01..2010-04 with pop [100, 200, 300, 400].
config = parse_config({
    "period": "monthly", "n_total": 2, "start_period": "2010-03",
    "variables": [{"name": "x", "generate": "flat",
                   "params": {"level": 1, "noise": 0}}],
    "disease_cases": {
        "population": {"generate": "from_csv",
                       "params": {"file": "pop.csv", "column": "pop"}},
        "depends_on": [{"variable": "x"}],
    },
})
run(config)[["time_period", "population"]]
# 2010-03 -> 100, 2010-04 -> 200 (January/February values).
```

Also, `params: {start_period: "2010-02"}` with no scenario `start_period`
selects February data but labels it `2000-01`.

**Fix:** Make one calendar start authoritative for the entire run. Apply the
scenario start to population generators too, and reject a generator-level
`start_period` that differs from it. If only the generator specifies a start,
either derive output labels from it or require the scenario field explicitly.
Test all three cases.

## 🔴 16. Synthetic seasonality ignores the calendar position of `start_period`

**What:** `start_period` offsets output labels, but all synthetic generators
and the disease baseline still receive a time axis beginning at zero. A
monthly scenario starting in July therefore generates the same first seasonal
value as one starting in January. `seasonal_spike(spike_center=0)` peaks on
the first output row regardless of whether that row is January, July, or any
other month. The disease baseline has the same phase error.

Daily seasonality also uses a fixed 365-row cycle while output labels use the
real leap-year calendar, so its phase shifts by one day after February 29.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 2, "start_period": "2010-07",
    "variables": [{"name": "rainfall", "generate": "seasonal_spike",
                   "params": {"baseline": 0, "spike_height": 1,
                              "spike_center": 0, "spike_width": 0.1,
                              "noise": 0}}],
    "disease_cases": {"population": 100,
                      "depends_on": [{"variable": "rainfall", "weight": 0}]},
})
run(config)[["time_period", "rainfall"]]
# 2010-07 -> 1.0: a center-0 (start-of-year) spike incorrectly peaks in July.
```

**Fix:** Build one calendar-aware period axis and pass its within-year offsets
to seasonal generators and the disease baseline. For daily data, derive phase
from the actual date rather than `row_index % 365`. Add mid-year start tests
for monthly, weekly, and daily scenarios plus a leap-year boundary test.

## 🟡 17. Non-finite numeric configuration values are accepted

**What:** Float fields and generator params accept NaN and infinity. Examples
include dependency `weight: NaN`, `weight: Infinity`, and generator
`level: NaN`. These values produce runtime warnings and all-NaN disease
output instead of a validation error. Because generator params are untyped
dicts, their constructors also need explicit finite checks.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 4,
    "variables": [{"name": "x", "generate": "flat",
                   "params": {"level": "NaN", "noise": 0}}],
    "disease_cases": {"population": 1000,
                      "depends_on": [{"variable": "x",
                                      "weight": "Infinity"}]},
})
run(config)  # accepted; emits RuntimeWarnings and invalid/all-NaN output
```

**Fix:** Require finite values on all schema floats (`weight`, rates,
overdispersion, train fraction) and validate every numeric generator param
with `math.isfinite`/`np.isfinite`. Raise a field-specific error before
generation. Add NaN, positive-infinity, and negative-infinity tests.

## 🟡 18. A missing generated-population value crashes in NumPy

**What:** `from_csv` intentionally permits blank numeric cells as NaN. When
used as a population generator, `_resolve_population` rounds and casts that
NaN to an integer, emitting `RuntimeWarning: invalid value encountered in
cast`. The resulting invalid population reaches the Poisson sampler, which
crashes with `ValueError: lam < 0 or lam contains NaNs`.

**Reproduce:**
```python
# popnan.csv contains pop [100, NaN, 100].
config = parse_config({
    "period": "monthly", "n_total": 3, "start_period": "2010-01",
    "variables": [{"name": "x", "generate": "flat",
                   "params": {"level": 1, "noise": 0}}],
    "disease_cases": {
        "population": {"generate": "from_csv",
                       "params": {"file": "popnan.csv", "column": "pop",
                                  "start_period": "2010-01"}},
        "depends_on": [{"variable": "x"}],
    },
})
run(config)  # RuntimeWarning, then ValueError from rng.poisson
```

**Fix:** Validate generated population before rounding/casting: every value
must be finite and positive (or explicitly define supported missing-population
semantics). Raise a clear population-generator error naming the bad period.

## 🟡 19. CLI generation errors escape as tracebacks

**What:** The CLI catches loader/schema errors only. Generator validation and
runtime errors occur later in `run_engine(config)` outside the `try` block, so
normal user mistakes such as an unknown generator param or missing
`from_csv` column raise an uncaught traceback instead of printing `error: ...`
and returning exit code 1. This also weakens bug #8's friendly error handling:
the final message is clearer, but it is still wrapped in a traceback.

**Reproduce:**
```yaml
period: monthly
n_total: 3
variables:
  - name: x
    generate: seasonal_spike
    params: {bogus: 1}
disease_cases:
  population: 100
  depends_on: [{variable: x}]
```

```bash
uv run dsl run bad.yaml
# Uncaught ValueError traceback, rather than a clean one-line CLI error.
```

**Fix:** Wrap generation and output/plot validation failures at the CLI
boundary, print a concise `error:` message to stderr, and return 1. Do not
swallow unexpected programming exceptions. Add CLI tests for invalid
generator params and malformed `from_csv` inputs.

## 🟡 20. Empty variable and location names are accepted as CHAP-valid

**What:** `VariableSpec.name` and location strings have no non-empty/whitespace
validation. An empty variable name creates an unnamed CSV header that pandas
later reads as `Unnamed: 2`. An empty location writes blank cells that pandas
reads back as NaN. `validate_chap` reports no findings for either generated
DataFrame, so the CLI claims success without warning.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 3, "locations": [""],
    "variables": [{"name": "", "generate": "flat"}],
    "disease_cases": {"population": 100,
                      "depends_on": [{"variable": ""}]},
})
df = run(config)
validate_chap(df)  # []
```

**Fix:** Strip and reject empty/whitespace-only variable and location names in
the schema. Independently make `validate_chap` flag null/blank location values
and blank column names. Test YAML empty strings and whitespace-only names.

## 🟡 21. `validate_chap` does not check daily or yearly consecutiveness

**What:** The validator's documentation says periods are checked for
consecutiveness, but steppers exist only for monthly and weekly data. Daily
`20000101, 20000103` and yearly `2000, 2002` sequences both return no
findings, even though each skips a period.

**Reproduce:**
```python
df = pd.DataFrame({
    "time_period": ["20000101", "20000103"],
    "location": ["x", "x"],
    "rainfall": [1.0, 2.0],
    "disease_cases": [1.0, 2.0],
})
validate_chap(df)  # []
```

**Fix:** Add calendar-based daily and integer yearly steppers and apply the
same adjacent-period check used for monthly/weekly data. Add rollover,
leap-day, and skipped-period tests.

## 🟡 22. Valid `train_fraction` values can produce an empty train split

**What:** Schema validation checks only `0 < train_fraction < 1`, while output
uses `floor(n_total * train_fraction)`. Small datasets or small fractions can
therefore produce `n_train == 0`; `train.csv` contains headers but no rows.
No warning mentions the empty split.

**Reproduce:**
```python
config = parse_config({
    "period": "yearly", "n_total": 2, "train_fraction": 0.1,
    "variables": [{"name": "x", "generate": "flat"}],
    "disease_cases": {"population": 100,
                      "depends_on": [{"variable": "x"}]},
})
write_output(run(config), config, "out")
# train.csv: 0 rows; test.csv: 2 rows.
```

**Fix:** Cross-validate `n_total` and `train_fraction` so both
`floor(n_total * fraction)` and the remainder are at least one, or make the
split rule guarantee non-empty partitions. Add boundary tests for one- and
two-period scenarios.

---

# Round 5 (#23-38) - cross-feature and output-integrity hunt

## 🔴 23. A zero-weight dependency still blanks disease rows

**What:** A dependency with `weight: 0` should have no effect, but its lag and
missing values still propagate NaN into `eta`. This blanks the dependency's
warm-up rows (and any rows where that driver is missing) even though the
driver contributes exactly zero to the disease model. Setting a weight to
zero therefore does not actually disable that dependency.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 12, "seed": 7,
    "variables": [{"name": "x", "generate": "linear_trend"}],
    "disease_cases": {
        "population": 1000,
        "depends_on": [{"variable": "x", "lag": 3, "weight": 0}],
    },
})
df = run(config)
# df["disease_cases"].iloc[:3] is all NaN solely because the zero-weight
# dependency has lag 3. With lag 0, those same seeded rows are valid counts.
```

**Fix:** Skip dependencies whose weight is exactly zero before lagging and
standardizing, or compute missingness only from dependencies that can affect
the predictor. Add tests for zero-weight lag warm-up and zero-weight missing
driver values.

## 🟡 24. RNG streams are coupled to variable and location declaration order

**What:** One sequential RNG is shared by all variables, locations, and the
disease draw. Semantically harmless YAML changes therefore alter unrelated
data:

- Reordering variables changes each named variable's generated values.
- Reordering locations changes the series assigned to each location name.
- Appending an unused decoy leaves the real driver unchanged but consumes
  random draws before disease generation, changing almost every case count.

This confounds controlled experiments: adding a covariate that disease does
not depend on changes the target being evaluated.

**Reproduce:**
```python
base = {
    "period": "monthly", "n_total": 24, "seed": 42,
    "variables": [{"name": "a", "generate": "flat"}],
    "disease_cases": {"population": 1000,
                      "depends_on": [{"variable": "a", "lag": 1}]},
}
with_decoy = {
    **base,
    "variables": base["variables"] + [{"name": "decoy", "generate": "flat"}],
}
x, y = run(parse_config(base)), run(parse_config(with_decoy))
np.array_equal(x["a"], y["a"])  # True
np.array_equal(x["disease_cases"], y["disease_cases"], equal_nan=True)  # False
```

**Fix:** Derive stable child RNG streams from the scenario seed plus a
component key (location name, variable name, population, disease, missingness)
using `SeedSequence`, so declaration order and unrelated components cannot
shift each other's draws. Add reorder and decoy-invariance tests.

## 🔴 25. Reusing an output directory leaves stale optional files

**What:** `write_output` overwrites files it writes but never removes files
that no longer belong to the run. If a directory first receives a scenario
with `train_fraction` and is then reused for a scenario without it, the old
`train.csv` and `test.csv` remain beside the new full dataset and metadata.
The folder now presents incompatible files as one run. An old plot similarly
remains when rerunning without `--plot`.

**Reproduce:**
```python
base = {
    "period": "monthly", "n_total": 4,
    "variables": [{"name": "x", "generate": "flat"}],
    "disease_cases": {"population": 100,
                      "depends_on": [{"variable": "x"}]},
}
split = parse_config({**base, "train_fraction": 0.5})
plain = parse_config(base)  # same output directory, no train_fraction
write_output(run(split), split, "out")
write_metadata(split, "out")
write_output(run(plain), plain, "out")
write_metadata(plain, "out")
# out/train.csv and out/test.csv still exist, but metadata describes `plain`.
```

**Fix:** Before writing an explicit output directory, remove known optional
artifacts that the new run will not produce, or write into a temporary
directory and atomically replace the run directory. Add split-to-no-split and
plot-to-no-plot rerun tests.

## 🔴 26. `metadata.json` cannot reproduce a `from_csv` run after its source changes

**What:** Metadata stores only the external CSV path, not the source content or
even a checksum. If that file is edited, replaced, moved, or deleted, feeding
the same `metadata.json` back to the CLI produces different data or fails.
This breaks the documented guarantee that every dataset can be regenerated
byte-identically from its metadata.

**Reproduce:**
```python
# real.csv initially contains x = [1, 2, 3].
config = parse_config({
    "period": "monthly", "n_total": 3, "start_period": "2010-01",
    "variables": [{"name": "x", "generate": "from_csv",
                   "params": {"file": "real.csv", "column": "x"}}],
    "disease_cases": {"population": 100,
                      "depends_on": [{"variable": "x"}]},
})
first = run(config)
meta = build_metadata(config)

# Replace the source values after the run.
# real.csv now contains x = [10, 20, 30].
second = run(parse_config(meta["scenario"]))
first.equals(second)  # False, despite using the original metadata.
```

**Fix:** Snapshot the selected real source data into the output directory and
make the reproducible scenario point to that immutable copy. Also record a
cryptographic hash and fail clearly if an external source no longer matches.
Test changed and deleted source files.

## 🔴 27. `from_csv` accepts impossible calendar labels

**What:** Resolution checking uses shape-only regular expressions, so labels
such as `2010-00`, `2010-99`, `2010-W00`, `2010-W99`, `20100230`, and
`20101340` are accepted as monthly, weekly, or daily data. Without a scenario
`start_period`, their values are then silently relabelled onto the engine's
valid `2000-*` timeline, hiding the invalid source dates.

**Reproduce:**
```python
# bad.csv:
# time_period,x
# 2010-00,1
# 2010-99,2
gen = FromCsvGenerator(file="bad.csv", column="x")
gen.generate(2, "monthly", np.random.default_rng(0))  # returns [1.0, 2.0]
```

**Fix:** Parse and validate every source label with real calendar logic, not
only a regex. For weekly data, enforce the exact week convention supported by
the engine. Add impossible month/week/date tests.

## 🟡 28. Multi-location duplication warning misses single-source CSVs

**What:** Bug #3's warning only fires when `params.source_location` is set.
With several output locations and a CSV that has exactly one location (or no
`location` column), `from_csv` still duplicates the same real series into
every output location, but `validate_scenario` emits no duplication warning.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 2, "start_period": "2010-01",
    "locations": ["X", "Y"],
    "variables": [{"name": "x", "generate": "from_csv",
                   "params": {"file": "one_location.csv", "column": "x"}}],
    "disease_cases": {"population": 100,
                      "depends_on": [{"variable": "x"}]},
})
validate_scenario(config)  # no from_csv warning
run(config).groupby("location")["x"].apply(list)
# X and Y receive identical source values.
```

**Fix:** Warn for every `from_csv` variable in a multi-location scenario
unless there is an explicit supported per-location mapping. The warning
should not depend on `source_location` being present.

## 🟡 29. Relative `from_csv` paths resolve against the process working directory

**What:** A path inside a scenario is interpreted relative to wherever
`dsl run` was launched, not relative to the scenario file. A portable folder
containing `scenario.yaml` and sibling `data.csv` fails when invoked from its
parent or another directory. The same scenario works only if the caller first
changes into the scenario folder.

**Reproduce:**
```text
experiment/
  scenario.yaml   # params.file: data.csv
  data.csv
```

```bash
uv run dsl run experiment/scenario.yaml
# ValueError: from_csv: file not found: data.csv
```

**Fix:** Resolve relative resource paths against the input scenario's parent
directory before generator construction. Metadata reproduction should apply
the same rule relative to the metadata file (or use the snapshot from #26).
Add CLI tests launched from a different working directory.

## 🟡 30. Negative seeds pass schema validation but crash the engine

**What:** `seed` accepts any integer, but `np.random.default_rng` requires a
non-negative integer. A negative seed is accepted as a valid
`ScenarioConfig`, then fails at the first engine line with a raw
`ValueError: expected non-negative integer`.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 2, "seed": -1,
    "variables": [],
    "disease_cases": {"population": 10, "depends_on": []},
})  # accepted
run(config)  # ValueError from NumPy
```

**Fix:** Add `ge=0` to the schema's seed field and test `-1` plus a valid zero
seed.

## 🔴 31. Infinite CSV values pass both ingestion and CHAP validation

**What:** `pd.to_numeric` treats `inf` and `-inf` as numeric, so `from_csv`
accepts them. `validate_chap` checks only dtype and NaN, not finiteness, and
therefore reports no findings. The CLI can write a dataset containing
infinite covariates as if it were CHAP-compatible. If the column drives
disease, standardization also emits runtime warnings and can blank the target.

**Reproduce:**
```python
# inf.csv has x = [1, inf, 3].
config = parse_config({
    "period": "monthly", "n_total": 3, "start_period": "2010-01",
    "variables": [{"name": "x", "generate": "from_csv",
                   "params": {"file": "inf.csv", "column": "x"}}],
    "disease_cases": {"population": 100, "depends_on": []},
})
df = run(config)
validate_chap(df)  # []
# df.loc[1, "x"] is inf.
```

**Fix:** Reject non-finite values in `from_csv` with a message naming the
column and period. Independently make `validate_chap` flag non-finite numeric
covariates and target values.

## 🟡 32. `validate_chap` crashes on non-numeric `disease_cases`

**What:** `validate_chap` promises never to raise, but it compares the target
directly with zero without checking its dtype. A string target column raises
`TypeError` instead of returning a finding. Numeric-looking CSV strings fail
the same way.

**Reproduce:**
```python
df = pd.DataFrame({
    "time_period": ["2000-01", "2000-02"],
    "location": ["x", "x"],
    "rainfall": [1.0, 2.0],
    "disease_cases": ["1", "2"],
})
validate_chap(df)
# TypeError: '<' not supported between instances of 'str' and 'int'
```

**Fix:** Check target numeric dtype before value comparisons and return a
clear finding when it is not numeric. Add numeric-string and arbitrary-text
tests that assert the validator does not raise.

## 🟡 33. Valid Sunday weeks and week 53 get false consecutiveness warnings

**What:** CHAP validation accepts both Sunday-start labels (`YYYY-Snn`) and
week 53, but `_next_week` always emits `W` labels and rolls over after week
52. Consequently:

- `2000-S01, 2000-S02` is flagged because the expected value is built as
  `2000-W02`.
- `2020-W52, 2020-W53, 2021-W01` is flagged at week 53.

These are formats the validator itself declares valid.

**Reproduce:**
```python
def frame(periods):
    return pd.DataFrame({
        "time_period": periods,
        "location": ["x"] * len(periods),
        "rainfall": [1.0] * len(periods),
        "disease_cases": [1.0] * len(periods),
    })

validate_chap(frame(["2000-S01", "2000-S02"]))
# "not consecutive"
validate_chap(frame(["2020-W52", "2020-W53", "2021-W01"]))
# "not consecutive"
```

**Fix:** Preserve the input week prefix and use calendar-aware week rollover,
including valid week-53 years. Add W, S, year-boundary, and week-53 tests.

## 🟡 34. A non-empty train split can contain zero observed targets

**What:** Bug #22 covers zero train rows, but a larger train file can still be
unusable. If the maximum disease lag is at least the number of training
periods, every training target is warm-up NaN while all valid cases land in
test. Schema validation only requires `lag < n_total`, and no warning checks
the split boundary.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 10, "train_fraction": 0.8,
    "variables": [{"name": "x", "generate": "linear_trend"}],
    "disease_cases": {"population": 100,
                      "depends_on": [{"variable": "x", "lag": 8}]},
})
write_output(run(config), config, "out")
# train.csv has 8 rows and zero non-NaN disease_cases.
# test.csv has the only 2 observed disease_cases.
```

**Fix:** Cross-check `floor(n_total * train_fraction)` against the maximum
effective lag and reject or clearly warn when training contains no known
target. Add boundary tests around `max_lag == n_train`.

## 🟡 35. `seasonal_spike` has weekly-biased defaults and broken distant centers

**What:** The default `spike_center=26` is sensible for weekly data but lies
more than two full cycles beyond a 12-month year. The circular-distance
formula only wraps once, so a default monthly series never reaches its
configured peak: it rises toward December and tops out below
`baseline + spike_height`. Larger explicit centers behave even worse
(`spike_center=24` should be equivalent to January but produces almost no
spike). Yearly data with the default is effectively flat.

**Reproduce:**
```python
x = SeasonalSpikeGenerator(noise=0).generate(
    12, "monthly", np.random.default_rng(0)
)
int(np.argmax(x))  # 11 (December)
x.max()            # ~17.1, not the configured peak 22.0

y = SeasonalSpikeGenerator(
    baseline=0, spike_height=1, spike_center=24,
    spike_width=0.2, noise=0,
).generate(12, "monthly", np.random.default_rng(0))
y.max()  # ~0.0000037 instead of 1.0
```

**Fix:** Normalize explicit centers modulo `periods_per_year` before computing
distance, and make the omitted default resolution-aware (for example,
mid-cycle rather than a hard-coded week number). Test defaults and multi-cycle
centers at every resolution.

## 🟡 36. Valid configurations can cross year 9999 and crash or emit invalid labels

**What:** `start_period` accepts year 9999 without checking whether the whole
requested range is representable. A daily scenario starting `99991231` with
two rows crashes with `OverflowError`. Monthly, weekly, and yearly scenarios
continue into five-digit year labels such as `10000-01` / `10000-W01` /
`10000`, which fail the project's own CHAP validator but are still written.

**Reproduce:**
```python
run(parse_config({
    "period": "daily", "n_total": 2, "start_period": "99991231",
    "variables": [],
    "disease_cases": {"population": 10, "depends_on": []},
}))  # OverflowError: date value out of range
```

**Fix:** Validate the full requested period range during schema parsing and
reject scenarios whose final period exceeds the supported four-digit calendar.
Add end-of-year boundary tests for all resolutions.

## ⚪ 37. Constant per-location populations are plotted as time-varying

**What:** Population is included in the plot whenever the entire column has
more than one unique value. Two locations with different but individually
constant populations therefore get a population panel, despite the documented
rule that population is plotted only when it varies over time (growth).

**Reproduce:**
```python
df = pd.DataFrame({
    "time_period": ["2000-01", "2000-02"] * 2,
    "location": ["A", "A", "B", "B"],
    "x": [1, 2, 3, 4],
    "disease_cases": [1, 2, 3, 4],
    "population": [100, 100, 200, 200],
})
_series_columns(df)
# ["x", "disease_cases", "population"]
```

**Fix:** Include population only when it has more than one value within at
least one location group. Test heterogeneous constant populations separately
from actual growth.

## ⚪ 38. Flattened metadata omits the count distribution ground truth

**What:** The full `scenario` block records `count_distribution` and
`overdispersion`, but the convenient top-level `metadata["disease_cases"]`
summary omits both. A negative-binomial run therefore looks like it has no
distribution setting unless the reader knows to inspect the nested scenario.
This conflicts with metadata's role as a quick ground-truth summary of disease
settings.

**Reproduce:**
```python
config = parse_config({
    "period": "monthly", "n_total": 3, "variables": [],
    "disease_cases": {
        "population": 100, "depends_on": [],
        "count_distribution": "negative_binomial",
        "overdispersion": 2.5,
    },
})
meta = build_metadata(config)
meta["scenario"]["disease_cases"]["count_distribution"]  # "negative_binomial"
meta["disease_cases"].get("count_distribution")          # None
```

**Fix:** Add `count_distribution` and `overdispersion` to the flattened
disease metadata and test both Poisson defaults and negative-binomial values.
