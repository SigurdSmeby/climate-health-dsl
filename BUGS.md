# Bug log

QA bug-hunt on v1.4.0 (branch `qa/bug-hunt`). **All 12 found are fixed**, each
with a regression test so they can't return. Severity: 🔴 data corruption /
silent wrong output, 🟡 missing guard / misleading, ⚪ cosmetic.

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
