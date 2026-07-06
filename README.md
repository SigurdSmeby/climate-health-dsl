# DSL — synthetic climate-health data

A YAML-based DSL for generating synthetic climate-health datasets. A scenario file declares how climate variables relate to disease (lags, weights, nonlinearities, missing data), and the tool generates a dataset embedding those relationships — so you can check how well a forecasting model recovers a *known* ground truth.

Output is formatted for [CHAP](https://chap.dhis2.org/), but is plain CSV.

**New here?** The [full guide](docs/GUIDE.md) walks from install to a real-data experiment and documents every field, generator, and transform. This page is the quick reference.

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

`--watch` re-runs on every save and reloads the browser plot. Drop it for a one-shot run. See the [guide](docs/GUIDE.md#getting-started) for a hands-on walkthrough.

## Commands and options

| Command | What it does |
|---|---|
| `dsl new [path]` | Write a commented starter scenario to edit (default `scenario.yaml`). |
| `dsl run <scenario>` | Generate a dataset from a scenario YAML (or reproduce one from a `metadata.json`). |
| `dsl list` | List the registered generators and transforms. |

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
| `simulated_data.csv` | always | The full dataset: `time_period`, `location`, one column per variable, `disease_cases`, `population`. Give this to CHAP — it does its own train/test hiding. |
| `train.csv`, `test.csv` | only if `train_fraction` is set | A split in time (first `train_fraction` of each location's periods vs the rest) for evaluation outside CHAP. |
| `metadata.json` | always | The ground truth behind the dataset: seed, lags, weights, transforms, rates, generators, tool version, and the full resolved scenario. Feed it back to `dsl run` to reproduce the data byte-for-byte — no original YAML needed. |
| `plot.html` (or `.png`/`.svg`/`.pdf`) | only with `--plot` | A faceted plot of the covariates and `disease_cases` over time, one line per location, train/test boundary marked. |

Rerunning the same scenario produces identical files — all randomness comes from the `seed`. Output is checked against CHAP's dataset rules; findings print as warnings and the run still writes output (they only arise with `from_csv` data that has gaps — synthetic output is always CHAP-valid).

## Learn more

- **[Full guide](docs/GUIDE.md)** — getting-started walkthrough, the complete scenario reference (every field), all generators and transforms with their params, how the disease model works, and how to extend the DSL with a new generator or transform.
- **`examples/`** — ready-to-run scenarios (`basic_scenario`, `confounders_and_controls`, `overdispersed_outbreaks`, …). `examples/real_data_demo/` has five fuller ones (real / synthetic / mixed), each with pre-generated output and its own README.

## Project layout

```
src/dsl/
├── core/                  # locked machinery — not edited when adding features
│   ├── extension/         #   registry + the two abstract base classes
│   ├── config/            #   YAML loader + Pydantic schema/validation
│   └── pipeline/          #   periods, disease model, engine, CSV output
├── generators/            # extension zone: one file = one variable shape
├── transforms/            # extension zone: one file = one series modification
└── cli.py                 # the `dsl run` / `dsl new` commands
tests/                     # mirrors the package; conftest.py has shared fixtures
docs/GUIDE.md              # the full guide
```

## Development

Run the tests with `uv run pytest`. The suite covers determinism (same seed → identical output), ground-truth recovery (`tests/test_ground_truth.py` proves a declared relationship is recoverable), validation (broken scenarios give clear, field-specific errors), and the config→DataFrame pipeline. Add tests with each feature, in the same commit. Commits follow [Conventional Commits](https://www.conventionalcommits.org).
