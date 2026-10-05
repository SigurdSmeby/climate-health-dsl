# DSL — synthetic climate-health data

A YAML-based DSL for generating synthetic climate-health datasets. A scenario file declares one list of series — climate and disease alike — and how they relate: how many periods later one reacts to another, how strongly, and how noisy or incomplete the data is. The tool then generates a dataset that actually embeds those relationships, so a forecasting model's output can be checked against the *exact*, known relationship you wrote — the "ground truth" — instead of an unknown real-world one.

Any series can drive any other, so a chain like rain → soil moisture → disease is expressible directly; named events couple series in the *same* period (one storm raising rainfall while dropping temperature); and a `split:` block writes cross-validation folds with a report of what each fold actually contains.

Output is plain CSV in long format — one row per location and period — so anything that reads a CSV can use it. The column conventions match what disease-forecasting tools expect; [CHAP](https://chap.dhis2.org/) is one such consumer.

**New here?** The [tutorial](docs/TUTORIAL.md) walks from install to a real-data experiment. This page is the quick reference — see [Learn more](#learn-more) for the full docs.

## Install

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"
```

(Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`, then drop the `uv run` prefix below.)

## Quick start

```bash
uv run dsl new my_scenario.yaml                    # write a commented starter
uv run dsl run my_scenario.yaml --plot --watch     # generate + live-reloading plot
uv run dsl run examples/basic_scenario.yaml        # or run a bundled example
```

`--watch` re-runs on every save and reloads the browser plot. Drop it for a one-shot run. See the [tutorial](docs/TUTORIAL.md) for a hands-on walkthrough.

## Commands and options

| Command | What it does |
|---|---|
| `dsl new [path]` | Write a commented starter scenario to edit (default `scenario.yaml`). |
| `dsl run <scenario>` | Generate a dataset from a scenario YAML (or reproduce one from a `metadata.json`). |
| `dsl list` | List the registered generators, transforms and emitters. |

**`dsl new [path]`** — `-f`, `--force`: overwrite the file if it already exists.

**`dsl run <scenario>`**

| Option | Default | Meaning |
|---|---|---|
| `scenario` | required | Path to a scenario YAML, or a `metadata.json` to reproduce a previous run. |
| `-o`, `--out-dir DIR` | auto-named | Directory to write into. If omitted, an auto-named folder under `out/` is used so previous runs are never overwritten. |
| `--plot` | off | Also write a plot of the dataset into the output directory. |
| `--plot-format FMT` | `html` | Plot format: `html` (interactive) or `png`/`svg`/`pdf`. |
| `--watch` | off | Re-run automatically whenever the scenario file is saved; serves a live-reloading plot when paired with `--plot`. |
| `--new` | off | Skip the continue-or-new prompt; always write a fresh auto-numbered folder. |
| `--replicates`, `-n N` | `1` | Generate N seeded replicates (seeds base, base+1, …) into `rep_00/`, `rep_01/`, … each independently reproducible — for showing evaluation robustness to seed. Default 1 writes a single run directly. Incompatible with `--watch`. |

## Output files

| File | When | Contents |
|---|---|---|
| `simulated_data.csv` | always | The full dataset: `time_period`, `location`, one column per series, `population`. Give this to a forecasting tool — most do their own train/test hiding. |
| `folds/fold_N/train.csv`, `test.csv` | only if `split:` is set | One directory per cross-validation fold, each ready to hand to a model unsliced. |
| `metadata.json` | always | The ground truth behind the dataset: seed, lags, weights, transforms, rates, generators, tool version, and the full resolved scenario. Feed it back to `dsl run` to reproduce the data byte-for-byte — no original YAML needed. |
| `plot.html` (or `.png`/`.svg`/`.pdf`) | only with `--plot` | A faceted plot of the covariates and `disease_cases` over time, one line per location, train/test boundary marked. |

Rerunning the same scenario produces identical files — all randomness comes from the `seed`. Output is checked against the dataset rules a forecasting tool expects; findings print as warnings and the run still writes output (they only arise with `from_csv` data that has gaps — synthetic output always satisfies them).

## Learn more

- **[Tutorial](docs/TUTORIAL.md)** — a hands-on walkthrough from install to a real-data experiment. Start here if you're new.
- **[Reference](docs/REFERENCE.md)** — every scenario field, generator, and transform, with defaults and meanings. Look things up here.
- **[How-to guides](docs/HOW_TO.md)** — add a new generator or transform.
- **[Concepts](docs/CONCEPTS.md)** — how the disease signal is actually built, step by step.
- **`examples/`** — ready-to-run scenarios (`basic_scenario`, `confounders_and_controls`, `overdispersed_outbreaks`, …). `examples/real_data_demo/` has five fuller ones (real / synthetic / mixed), each with pre-generated output and its own README.

## Project layout

```
src/dsl/
├── core/                  # locked machinery — not edited when adding features
│   ├── extension/         #   registry + the three abstract base classes
│   ├── config/            #   YAML loader + Pydantic schema/validation
│   └── pipeline/          #   periods, signal, engine, CSV output, fold report
├── generators/            # extension zone: one file = one series shape
├── transforms/            # extension zone: one file = one series modification
├── emitters/              # extension zone: one file = one output type
└── cli.py                 # the `dsl run` / `dsl new` commands
tests/                     # mirrors the package; conftest.py has shared fixtures
docs/                      # tutorial, reference, how-to guides, concepts
```

## Development

Run the tests with `uv run pytest`. The suite covers determinism (same seed → identical output), ground-truth recovery (`tests/test_ground_truth.py` proves a declared relationship is recoverable), validation (broken scenarios give clear, field-specific errors), and the config→DataFrame pipeline. Add tests with each feature, in the same commit. Commits follow [Conventional Commits](https://www.conventionalcommits.org).
