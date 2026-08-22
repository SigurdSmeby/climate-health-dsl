# Tutorial: your first scenario

A hands-on path from install to a real-data experiment — one command or edit per step.

New to the DSL? Start here. Looking something up instead? See the [reference](REFERENCE.md) (every field, generator, and transform), [how-to guides](HOW_TO.md) (extend the DSL), or [concepts](CONCEPTS.md) (how the disease model works). Quick install/commands: the [README](../README.md).

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

(Prefer to start from a finished example? `examples/` has several ready to run, e.g. `uv run dsl run examples/basic_scenario.yaml`.)

**2. Run it and look.** `--plot` writes an interactive `plot.html`; `--watch` re-runs every time you save the file:

```bash
uv run dsl run my_scenario.yaml --plot --watch
```

Together they open a browser tab served from `localhost` that **reloads itself** whenever you save the scenario — edit, save, watch the plot update live, no manual refresh. (Drop `--watch` for a one-shot run; it just writes `plot.html`.)

When a scenario already has output, `dsl run` lists the earlier runs and asks whether to **continue one** (refine the same `out/` folder) or start fresh — so a second session doesn't silently spawn `out/my_scenario_1/`. Use `--new` to skip the prompt and always start fresh, or `-o DIR` to write somewhere specific.

**3. Read the output.** `out/my_scenario/simulated_data.csv` has one row per period — the climate columns, `disease_cases`, and `population`. The first `disease_cases` are blank on purpose: with `lag: 2` there's no driver signal yet to react to. That blanked warm-up is the lag made visible.

**4. Change one thing, watch it move.** In `my_scenario.yaml`, bump `lag: 2` to `lag: 6` and save. With `--watch` running, the dataset regenerates and the plot refreshes — the disease peak shifts later relative to rainfall. Adding a second driver is pure YAML, no code:

```yaml
variables:
  - name: rainfall
    generate: seasonal_spike
  - name: mean_temperature      # add a second climate variable
    generate: seasonal_smooth
disease_cases:
  depends_on:
    - { variable: rainfall, lag: 6 }
    - { variable: mean_temperature, lag: 2 }   # ...and a second driver
```

**5. Use real climate data.** Swap a synthetic generator for `from_csv` to drive disease off *real* climate (a bundled three-province Laos sample lives in `examples/data/`):

```yaml
variables:
  - name: rainfall
    generate: from_csv
    # laos_subset.csv holds three provinces; source_location picks one (or set
    # locations: [Bokeo, ...] at the top to match the CSV names).
    params: { file: examples/data/laos_subset.csv, column: rainfall, source_location: Bokeo }
```

**6. Generate replicates.** To show your evaluation isn't a fluke of one seed, generate several with `--replicates`:

```bash
uv run dsl run my_scenario.yaml -o out/study --replicates 20
```

This writes `out/study/rep_00/`, `rep_01/`, … each a full dataset+metadata with seed `base, base+1, …`. Run your forecaster over all of them and report the spread.

**7. Explore the worked examples.** `examples/real_data_demo/` has five fuller scenarios (real, synthetic, and mixed) — each with pre-generated output and a `README`:

```bash
uv run dsl run examples/real_data_demo/01_vietnam_multiprovince.yaml --plot
```

## Where to next

- **[Reference](REFERENCE.md)** — every scenario field, generator, and transform, with defaults and meanings.
- **[How-to guides](HOW_TO.md)** — add a new generator or transform.
- **[Concepts](CONCEPTS.md)** — how the disease signal is actually built, step by step.
