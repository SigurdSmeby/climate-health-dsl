# Tutorial: your first scenario

A hands-on path from install to a real-data experiment — one command or edit per step.

New to the DSL? Start here. Looking something up instead? See the [reference](REFERENCE.md) (every field, generator, transform and emitter), [how-to guides](HOW_TO.md) (extend the DSL), or [concepts](CONCEPTS.md) (how the disease model works). Quick install/commands: the [README](../README.md).

**0. Install.** Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"
```

(Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`, then drop the `uv run` prefix below.)

**1. Scaffold a starter.** Writes a small, commented scenario:

```bash
uv run dsl new my_scenario.yaml
```

(Prefer to start from a finished one? `uv run dsl run examples/minimal.yaml`.)

Open `my_scenario.yaml` now and skim it — every line is commented, and the comments explain the file as you read. You'll see a `series:` list holding two entries: `rainfall`, and a `disease_cases` that depends on it with `lag: 2` — so the disease responds 2 periods after rainfall moves. Everything lives in that one list; a series becomes a disease signal by carrying a `counts:` block. Keep this file open; the next steps refer back to it.

**2. Run it and look.** `--plot` writes an interactive `plot.html`; `--watch` re-runs every time you save the file:

```bash
uv run dsl run my_scenario.yaml --plot --watch
```

Together they open a browser tab served from `localhost` that **reloads itself** whenever you save the scenario — edit, save, watch the plot update live, no manual refresh. (Drop `--watch` for a one-shot run; it just writes `plot.html`.)

**What you should see:** the terminal prints `Wrote 36 rows to out/my_scenario/` and `Wrote plot to out/my_scenario/plot.html`, and a browser tab opens showing two stacked panels — `rainfall` on top, `disease_cases` below — each with a line rising and falling over 36 months. If nothing opens, open `out/my_scenario/plot.html` yourself; `--watch` only affects auto-reload, not whether the file gets written.

When a scenario already has output, `dsl run` lists the earlier runs and asks whether to **continue one** (refine the same `out/` folder) or start fresh — so a second session doesn't silently spawn `out/my_scenario_1/`. Use `--new` to skip the prompt and always start fresh, or `-o DIR` to write somewhere specific.

**3. Read the output.** `out/my_scenario/simulated_data.csv` has one row per period — the climate columns, `disease_cases`, and `population`. The first `disease_cases` are blank on purpose: with `lag: 2` (the line you saw in the YAML in step 1) there's no driver signal yet to react to. That blanked warm-up is the lag made visible.

**4. Change one thing, watch it move.** In `my_scenario.yaml`, bump `lag: 2` to `lag: 6` and save. With `--watch` running, the dataset regenerates and the plot refreshes — the disease peak shifts later relative to rainfall. Adding a second driver is pure YAML, no code:

```yaml
series:
  - name: rainfall
    generate: seasonal_spike
  - name: mean_temperature      # add a second climate series
    generate: seasonal_smooth
  - name: disease_cases
    counts: {}
    depends_on:
      - { series: rainfall, lag: 6 }
      - { series: mean_temperature, lag: 2 }   # ...and a second driver
```

**5. Use real climate data.** Swap a synthetic generator for `from_csv` to drive disease off *real* climate (a bundled three-province Laos sample lives in `examples/data/`). In `my_scenario.yaml`, replace just the `rainfall` series' `generate:`/`params:` lines with these (leave the rest as-is):

```yaml
series:
  - name: rainfall
    generate: from_csv
    # laos_subset.csv holds three provinces; source_location picks one. To use
    # several, name them at the top instead, each with its own population:
    #   locations: { Bokeo: {population: 75000}, ... }
    # and drop source_location so each output location reads its own rows.
    params: { file: examples/data/laos_subset.csv, column: rainfall, source_location: Bokeo }
```

Save, and (with `--watch` still running) the plot updates to show real Laos rainfall driving a still-synthetic disease signal.

**6. Generate replicates.** To show your evaluation isn't a fluke of one seed, generate several with `--replicates`:

```bash
uv run dsl run my_scenario.yaml -o out/study --replicates 20
```

This writes `out/study/rep_00/`, `rep_01/`, … each a full dataset+metadata with seed `base, base+1, …`. Run your forecaster over all of them and report the spread.

**7. Link series, and shock several at once.** A series can be built from other
series, so a chain like rain → soil moisture → disease is a real column rather
than a hidden weight. A named event then strikes several series in the *same*
period — a storm raising rainfall while cooling the air, which no generator can
do, since a generator sees only its own series:

```bash
uv run dsl run examples/linked_series.yaml --plot
```

**8. Split it for evaluation.** `split:` writes one folder per
cross-validation fold, plus a report of what each fold actually contains — so a
fold whose test half holds no outbreak is visible, instead of quietly scoring
well:

```bash
uv run dsl run examples/cross_validation.yaml
cat out/cross_validation/folds/report.md
```

## Where to next

- **[Reference](REFERENCE.md)** — every scenario field, generator, and transform, with defaults and meanings.
- **[How-to guides](HOW_TO.md)** — add a new generator or transform.
- **[Concepts](CONCEPTS.md)** — how the disease signal is actually built, step by step.
