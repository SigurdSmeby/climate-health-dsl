# Tutorial: your first scenario

A hands-on path through the file `dsl new` writes for you. Every step edits one
thing in that file and shows you what changed in the data.

New here? Start at the top and work down. Looking something up instead? See the
[reference](REFERENCE.md). Want to know *why* the disease signal is built the
way it is? [Concepts](CONCEPTS.md).

## 1. Write a scenario

```bash
uv run dsl new my_scenario.yaml
```

Open it. It is a complete, working scenario — two climate series driving a
disease, split into folds for evaluation — and every field is commented. The
rest of this tutorial walks through it block by block.

## 2. Run it and look

```bash
uv run dsl run my_scenario.yaml --plot --watch
```

A browser tab opens with one panel per series. `--watch` re-runs and reloads
the plot every time you save, so keep it running while you read on.

If no tab opens, open `out/my_scenario/plot.html` yourself — `--watch` only
controls auto-reload, not whether the file is written.

## 3. The shape of the run

```yaml
period: monthly       # daily | weekly | monthly | yearly
n_total: 60           # how many periods (here: 5 years)
seed: 42              # same scenario + same seed -> identical data
```

`seed` is what makes an experiment repeatable: the same scenario and seed give
byte-identical output, every time and on any machine. Change it to draw a
different sample of the *same* relationship.

## 4. Where the disease happens

```yaml
locations:
  loc:   { population: 100000 }
```

A disease is counted against the people who live somewhere, so population
belongs to the location, not to the disease. Delete the `#` on the `north:`
line to add a second place — it draws its own climate, and the output stacks
both with a `location` column.

Every commented option in the file works this way: **delete the `#` and
nothing else.** The indentation after it is already the indentation the line
needs. Two options say in the file that they come as a set — an event and the
series that react to it — and the comment above them tells you which.

## 5. The series list

```yaml
series:
  - name: rainfall
    unit: mm
    generate: seasonal_spike
    params:
      spike_center: 7
      spike_height: 25
      spike_width: 2
```

Everything the scenario builds lives in this one list — climate and disease
alike. Each entry names a column in the output. `generate:` picks a shape from
the [generators](REFERENCE.md#generators), and `params:` tunes it.

`unit: mm` is a label, not a conversion — the generator produces numbers and
only you know they stand for millimetres. It goes on the plot's y-axis so the
panel reads `rainfall (mm)`. Leave it out for a series that measures nothing
physical, like a moisture index built from other series.

`spike_center: 7` is July — month numbers, counting from 1. `spike_width: 2`
is how many months the season spreads over: raise it and the wet season
swells until it never really ends, so the yearly rhythm stops being visible.

**Try it:** change `spike_center` to `1` and save. The wet season moves to
January, and the disease peak follows it.

## 6. The disease, and the lag you are planting

```yaml
  - name: disease_cases
    counts:
      median_rate: 0.1
      max_rate: 0.3
    depends_on:
      - series: rainfall
        lag: 2
        weight: 1.5
```

A `counts:` block is what makes a series a disease signal: its values become
whole case counts drawn against the location's population, rather than a plain
number. [Concepts](CONCEPTS.md) explains that model.

`lag: 2` is **the ground truth you are planting** — the disease reacts two
months after the rain, so a July rainfall peak shows up as a September peak in
`disease_cases`. That offset is the whole point of the tool: a forecasting
model claiming to have found a two-month lag can be checked against the two you
wrote.

**Try it:** change `lag` to `6` and save. The disease peak slides six months
after the rainfall peak. The first six `disease_cases` go blank, because there
is no rainfall yet for them to react to — that blanked warm-up *is* the lag,
made visible.

## 7. Read the output

`out/my_scenario/simulated_data.csv` has one row per location and period.
Beside it:

- **`metadata.json`** — the ground truth behind the data: every lag, weight and
  generator, plus the resolved scenario. Feed it back to `dsl run` to reproduce
  the dataset byte-for-byte, with no YAML needed.
- **`folds/`** — one folder per evaluation fold, from the `split:` block.

## 8. The folds, and whether they test anything

```yaml
split:
  kind: time
  k: 5
```

`expanding` (the default) trains each fold only on the periods *before* its
test block, so a model is never shown the future it is asked to forecast.

```bash
cat out/my_scenario/folds/report.md
```

That report answers the question a split cannot: **does every fold still test
what you planted?** It counts each planted feature on both sides of every fold,
and warns about any fold whose test half holds none — a fold that would score
well on recovering something it never contained.

## 9. Link series together

Drop the `#` from the `soil_moisture` block at the bottom of the file, then
point the disease at it instead of at rainfall:

```yaml
    depends_on:
      - { series: soil_moisture, lag: 1, weight: 2.0 }
```

Rain now fills a moisture store that heat drains and that carries over month to
month, and the disease responds to the *store*. The chain
rain → soil_moisture → disease is a real column in the output, not a hidden
weight — so you can see the intermediate step you planted.

## 10. Shape the response, beyond a straight line

`weight` scales a driver, so on its own it can only say "more rain, more
disease, in proportion". Real climate-health links rarely work that way: an
effect may appear only past a threshold, or peak at an optimum and fall away
either side.

A `transforms:` list on a dependency reshapes that driver before it is
weighted, so the relationship becomes a function of the driver's value. Delete
the `#` from the `transforms:` line under `depends_on`:

```yaml
    depends_on:
      - series: rainfall
        lag: 2
        weight: 1.5
        transforms: [{ name: threshold, params: { mode: hinge, threshold: 20 } }]
```

`hinge` passes only what exceeds 20 mm, so the disease reacts to heavy rain
alone and light months contribute nothing.

Other shapes are available for the same driver. For a temperature optimum —
transmission best around 25 °C and worse when it is too cold *or* too hot —
use `peak`:

```yaml
        transforms: [{ name: threshold, params: { mode: peak, threshold: 25 } }]
```

`quadratic` is that curve upside down: it puts the *minimum* at the threshold,
so the extremes contribute most. That is a real shape too ("extremes are bad"),
just not the one temperature usually has. `log` gives diminishing returns — the
first 100 mm of rain matters more than the second — and `power` sweeps the
curve's shape with one exponent. The [reference](REFERENCE.md#transforms) lists
every transform.

**The threshold has to sit inside the driver's own range.** Set it below the
series minimum and `hinge` becomes a constant subtraction, which standardizing
then removes — the option looks enabled and changes nothing. Check the
rainfall column before picking a number.

**Try it:** run with and without that line and compare the wet-season peaks.
The planted relationship is still exact ground truth; it is simply no longer a
straight line.

## 11. Go further

Any value in the scenario can say which values to try, and the run writes one
dataset per combination:

```yaml
seed: { sweep: { from: 42, to: 61 } }
```

```bash
uv run dsl run my_scenario.yaml -o out/study
```

Twenty datasets in `out/study/seed_42/`, `seed_43/`, …, each independently
reproducible. Run a forecaster over all of them and report the spread, so a
result is not a fluke of one draw.

Each run directory holds its own `metadata.json`, and a `sweep.json` beside
them maps every directory to the values that made it — needed because a
directory called `weight_0p5` cannot be parsed back into `0.5` reliably, and
evaluation tools identify rows by filename.

The same trick answers a different question when the swept value is a planted
relationship rather than the seed:

```yaml
depends_on:
  - series: rainfall
    lag: { sweep: [1, 2, 3] }
    weight: { sweep: { from: 0.2, to: 1.0, step: 0.2 } }
```

Fifteen datasets covering three lags at five effect strengths — enough to ask
at what point a model stops recovering what you planted. See
[REFERENCE.md](REFERENCE.md#sweeps-one-scenario-many-datasets) for the rules.

### Skip the typing: presets

Everything in this tutorial was written by hand so you could see what each
field does. Once you know, `preset:` gives you a whole scenario and your file
says only the difference:

```yaml
preset: evaluation
n_total: 240
```

That is two climate drivers, two locations, a counted disease with planted
lags, five expanding folds and a storm in every fold's test half — over 240
periods instead of the preset's 120. `dsl list` names the three that ship
(`minimal`, `climate`, `evaluation`).

Overriding is per field, not per block, so this changes one parameter and
keeps the rest of the preset's rainfall intact:

```yaml
preset: climate
series:
  - name: rainfall
    params: { spike_height: 40 }
```

A preset is an ordinary scenario file, so `dsl run` one directly to see what it
produces. See [REFERENCE.md](REFERENCE.md#presets-start-from-a-scenario-write-only-the-difference)
for the merge rules.

The four bundled [examples](../examples/) each show one thing in full:
[`minimal`](../examples/minimal.yaml),
[`from_real_climate`](../examples/from_real_climate.yaml) (real CSV climate),
[`linked_series`](../examples/linked_series.yaml) (chains and shared shocks),
and [`cross_validation`](../examples/cross_validation.yaml).

## Where to next

- **[Reference](REFERENCE.md)** — every field, generator, transform and emitter.
- **[Concepts](CONCEPTS.md)** — how the disease signal is built, and why.
- **[How-to](HOW_TO.md)** — add a new generator, transform or emitter.
