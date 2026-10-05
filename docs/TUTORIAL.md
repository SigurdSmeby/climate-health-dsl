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
  loc:
    population: 100000
```

A disease is counted against the people who live somewhere, so population
belongs to the location, not to the disease. Drop the `#` on the `north:` line
to add a second place — it draws its own climate, and the output stacks both
with a `location` column.

## 5. The series list

```yaml
series:
  - name: rainfall
    generate: seasonal_spike
    params:
      spike_center: 7
      spike_height: 25
```

Everything the scenario builds lives in this one list — climate and disease
alike. Each entry names a column in the output. `generate:` picks a shape from
the [generators](REFERENCE.md#generators), and `params:` tunes it.

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
months after the rain. That is the whole point of the tool: a forecasting model
claiming to have found a two-month lag can be checked against the two you wrote.

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

## 10. Go further

```bash
uv run dsl run my_scenario.yaml -o out/study --replicates 20
```

Twenty datasets with seeds `42, 43, …`, each independently reproducible. Run a
forecaster over all of them and report the spread, so a result is not a fluke
of one draw.

The four bundled [examples](../examples/) each show one thing in full:
[`minimal`](../examples/minimal.yaml),
[`from_real_climate`](../examples/from_real_climate.yaml) (real CSV climate),
[`linked_series`](../examples/linked_series.yaml) (chains and shared shocks),
and [`cross_validation`](../examples/cross_validation.yaml).

## Where to next

- **[Reference](REFERENCE.md)** — every field, generator, transform and emitter.
- **[Concepts](CONCEPTS.md)** — how the disease signal is built, and why.
- **[How-to](HOW_TO.md)** — add a new generator, transform or emitter.
