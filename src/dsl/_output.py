"""Output directory management and starter scenario template.

This module handles deciding where a run writes its files (with
auto-numbering so a previous run is never silently overwritten) and
writing the commented starter scenario `dsl new` produces.
"""
import sys
from pathlib import Path

# The starter scenario `dsl new` writes for the user to edit. Two climate
# drivers, a counted disease and a split, so a first run produces everything
# the tool makes — commented so the file itself teaches, with pointers to the
# docs rather than an exhaustive menu. It must parse and run with no warnings,
# and every commented-out block must be valid once uncommented (there is a
# test for both).
STARTER_TEMPLATE = """\
# A starter scenario: two climate series driving a disease, split into folds
# for evaluation. Everything here is live — run it, read the plot, then change
# one value and run again.
#
#   dsl run scenario.yaml --plot --watch     (re-runs on every save)
#
# The comments explain each field as you read. For the full list of fields,
# generators and transforms see docs/REFERENCE.md; for WHY the disease signal
# is built the way it is, docs/CONCEPTS.md.

period: monthly       # daily | weekly | monthly | yearly
n_total: 60           # how many periods to generate (here: 5 years)
seed: 42              # same scenario + same seed -> identical data, every time

# A disease is counted against the people who live somewhere, so population
# belongs to the location rather than to the disease. Add more locations to
# stack several places in one dataset; each draws its own climate.
locations:
  loc:
    population: 100000
  # north: { population: 300000 }

# Divide the data for evaluation. `expanding` trains each fold only on the
# periods BEFORE its test block, which is what judging a forecaster requires.
# Writes folds/fold_N/{train,test}.csv plus a report of what each fold holds.
# (`kind: location` holds whole places out instead — see REFERENCE.)
split:
  kind: time
  k: 5

# A named shock that strikes SEVERAL series in the same period — the one thing
# a generator cannot do, since a generator only ever sees its own series.
# `per_fold: 1` puts one in every fold's test half, worked out from the split
# above, so the periods follow k and n_total instead of being written down.
# Drop the `#` here AND on the two `events:` lines below to switch it on.
# (`at: [14, 47]` names exact periods instead; `rate: 0.05` fires randomly.)
# events: { storm: { per_fold: 1 } }

# Everything the scenario builds lives in this one list — climate and disease
# alike. A series becomes a disease signal by carrying a `counts:` block.
series:
  - name: rainfall            # becomes a column in the output CSV
    generate: seasonal_spike  # a sharp yearly rainy season
    params:                   # every generator takes `params:`
      spike_center: 7         # peak month of the wet season (1-12)
      spike_height: 25        # how far the peak rises above baseline
      clamp_min: 0            # rainfall can't go negative
    # events: { storm: { multiplier: 2.5 } }   # 2.5x the rain that month
    # missing_rate: 0.02      # blank ~2% of periods, as a broken gauge would

  - name: mean_temperature    # the conventional name (not "temperature")
    generate: seasonal_smooth # a yearly sine wave, good for temperature
    params:
      mean: 25                # average temperature
      amplitude: 6            # how far it swings across the year
    # events: { storm: { add: -6.0 } }         # and six degrees cooler, the
                                               # same storm, the same month

  # `counts:` turns this series' values into whole case counts, drawn against
  # the location's population. Empty uses the defaults (REFERENCE lists them).
  - name: disease_cases
    counts:
      # Empty would work too — these are the defaults, shown so you can tune
      # them. REFERENCE lists the rest.
      median_rate: 0.1                  # where a typical period sits
      max_rate: 0.3                     # ceiling, as a fraction of population
      # distribution: negative_binomial # spikier than the default poisson
    depends_on:
      # One entry per driver. The lag is the ground truth you are planting:
      # change it, re-run, and see the disease peak move.
      - series: rainfall
        lag: 2                # disease reacts 2 months after the rain
        weight: 1.5           # strength relative to the other drivers
        # Reshape a driver before it is weighted in — a threshold makes the
        # effect kick in only above 5 mm. REFERENCE lists every transform.
        # transforms: [{ name: threshold, params: { mode: hinge, threshold: 5 } }]
      - series: mean_temperature
        lag: 1
        weight: 1.0

  # Series can also drive each other. Drop the `#` to add a moisture store that
  # rain fills and heat drains, then point disease_cases at it instead — the
  # chain rain -> soil_moisture -> disease becomes its own column.
  # - { name: soil_moisture, autoregressive: { phi: 0.85 }, depends_on: [{ series: rainfall, lag: 1, weight: 1.0 }, { series: mean_temperature, lag: 1, weight: -0.4 }] }
"""


def _resolve_out_dir(input_path: str, out_arg: str | None) -> Path:
    """Decide where to write, never overwriting a previous run by default.

    If ``out_arg`` is given, use it directly (the user's explicit choice,
    overwrite allowed). Otherwise write into ``out/<name>/`` where ``name``
    is the input's stem — or, for a ``metadata.json`` sidecar, its parent
    folder name, so reproducing ``out/foo/metadata.json`` yields ``out/foo``-
    style names rather than ``out/metadata``. The first run is unnumbered;
    if it already exists, the lowest free ``out/<name>_<n>`` (n from 1) wins.

    Args:
        input_path: Path to the scenario YAML or metadata.json being run.
        out_arg: The user's explicit -o/--out-dir value, or None.

    Returns:
        The output directory path to write into.
        Example: Path("out/scenario") or Path("out/scenario_2").
    """
    if out_arg is not None:
        return Path(out_arg)

    path = Path(input_path)
    name = path.parent.name if path.name == "metadata.json" else path.stem

    base = Path("out") / name
    if not base.exists():
        return base
    n = 1
    while (Path("out") / f"{name}_{n}").exists():
        n += 1
    return Path("out") / f"{name}_{n}"


def _numbered_dir_suffix(entry: Path, prefix: str) -> int | None:
    """The integer suffix of ``entry`` if its name is ``<prefix><digits>``.

    Shared by ``_run._scenario_runs`` (``out/<name>_<N>`` siblings) and the
    replicate cleanup (``rep_NN`` dirs) in ``_run._run_replicates``.

    Args:
        entry: A candidate directory entry.
        prefix: The expected name prefix (e.g. "scenario_" or "rep_").

    Returns:
        The trailing integer, or None if ``entry`` isn't a directory or its
        name doesn't match (a non-numeric or missing suffix).
    """
    if not entry.is_dir() or not entry.name.startswith(prefix):
        return None
    suffix = entry.name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _write_starter(path: Path, force: bool) -> int:
    """Write the starter scenario to ``path``. Refuse to clobber unless forced.

    Args:
        path: Where to write the starter scenario.
        force: If True, overwrite an existing file at path.

    Returns:
        Exit code: 0 on success, 1 if path exists and force is False.
    """
    if path.exists() and not force:
        print(
            f"error: {path} already exists (use --force to overwrite)",
            file=sys.stderr,
        )
        return 1
    path.write_text(STARTER_TEMPLATE)
    print(
        f"Wrote a starter scenario to {path}. Run it live with:\n"
        f"  dsl run {path} --plot --watch\n"
        f"then edit the file and save to see the data update. "
        f"(Prefix with 'uv run' if you use uv.)"
    )
    return 0
