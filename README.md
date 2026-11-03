# DSL — synthetic climate-health data

Generate climate-health datasets whose relationships you **already know**.

A scenario file says how climate drives disease — which series depend on which,
how many periods later, how strongly. The tool emits a dataset that embeds
exactly that. So when a forecasting model claims to have found a two-month
rainfall lag, you can check it against the lag you planted, rather than against
an unknown real-world one.

Output is plain CSV in long format, so anything that reads a CSV can use it.

## Install

Python 3.11+, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[dev]"
```

## Run

```bash
uv run dsl new my_scenario.yaml                 # a commented starter to edit
uv run dsl run my_scenario.yaml --plot          # generate a dataset + a plot
uv run dsl run examples/minimal.yaml            # or run a bundled example
```

`dsl run` writes `simulated_data.csv` plus a `metadata.json` that reproduces
the run byte-for-byte. Add `--watch` to re-run on every save with a
live-reloading plot. `dsl list` shows the available building blocks.

### Start from a preset

A whole scenario you did not have to write. `preset:` loads one and your file
says only what differs:

```yaml
preset: climate
n_total: 120
```

Three ship — `minimal`, `climate` and `evaluation` (five folds plus the report
that says what each one contains). A series you name is merged into the
preset's, so changing one parameter does not drop the rest.

### One scenario, many datasets

Any value can say which values to try, and the run writes one dataset per
combination — no flag, the `sweep:` is the trigger:

```yaml
depends_on:
  - series: rainfall
    lag: { sweep: [1, 2, 3] }
    weight: { sweep: { from: 0.2, to: 1.0, step: 0.2 } }
```

Fifteen datasets in `out/study/lag_1__weight_0p2/`, …, each reproducible, plus
a `sweep.json` mapping every directory to the values that made it — so a
model's scores can be joined back to the relationship it was asked to recover.
Sweep `seed:` instead to vary only the draw.

## Learn more

- **[Tutorial](docs/TUTORIAL.md)** — write and read your first scenario. Start here.
- **[Reference](docs/REFERENCE.md)** — every field, generator, transform and emitter.
- **[Concepts](docs/CONCEPTS.md)** — how the disease signal is built, and why.
- **[How-to](docs/HOW_TO.md)** — extend the DSL with a new building block.

## Examples

Four scenarios, each showing one thing:

| File | Shows |
|---|---|
| [`minimal.yaml`](examples/minimal.yaml) | the smallest scenario that runs |
| [`from_real_climate.yaml`](examples/from_real_climate.yaml) | real climate from a CSV, synthetic disease |
| [`linked_series.yaml`](examples/linked_series.yaml) | series driving each other, and a shock hitting several at once |
| [`cross_validation.yaml`](examples/cross_validation.yaml) | folds for evaluation, with a report of what each contains |

## Development

`uv run pytest` (833 tests) covers determinism, ground-truth recovery,
validation and the full CLI. `uv run ruff check .` lints. Add tests with every
feature, in the same commit; commits follow
[Conventional Commits](https://www.conventionalcommits.org).
