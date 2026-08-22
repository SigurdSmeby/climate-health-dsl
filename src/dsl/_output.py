"""Output directory management and starter scenario template.

This module handles deciding where a run writes its files (with
auto-numbering so a previous run is never silently overwritten) and
writing the commented starter scenario `dsl new` produces.
"""
import sys
from pathlib import Path

# A minimal, valid starter scenario `dsl new` writes for the user to edit.
# Kept simpler than examples/basic_scenario.yaml (monthly, one driver) and
# commented so the file itself teaches; it must parse and run with no warnings.
STARTER_TEMPLATE = """\
# A starter scenario. Run it, open the plot, then change a value and re-run:
#   dsl run scenario.yaml --plot --watch   (re-runs every time you save)
# Lines starting with '#' are comments. Uncomment the blocks below to try more.

period: monthly       # daily | weekly | monthly | yearly
n_total: 36           # how many periods to generate (here: 3 years)
seed: 42              # same seed -> identical data every run

variables:
  - name: rainfall            # becomes a column; CHAP uses 'rainfall'
    generate: seasonal_spike  # a yearly rainy-season bump
    params:                   # every generator takes `params:` -- tune its shape
      spike_center: 7         # peak month of the rainy season (1-12)
      spike_height: 25        # how tall the wet-season peak is above baseline
      clamp_min: 0            # rainfall can't go negative
    # shared: 0.8            # multi-location only: fraction of this signal shared
                             #   across locations (a latent regional driver). Try
                             #   it with `locations: [north, south]` at the top.

  # A second climate variable -- uncomment to add it (no code needed, just YAML).
  # 'seasonal_smooth' is a yearly sine wave, good for temperature.
  # - name: mean_temperature      # CHAP's column name (not "temperature")
  #   generate: seasonal_smooth
  #   params:
  #     mean: 25                  # average temperature
  #     amplitude: 6              # how far it swings above/below across the year

  # A non-seasonal "decoy" the disease does NOT depend on -- a control to check
  # a model doesn't latch onto an irrelevant variable. 'flat' is constant+noise.
  # - name: humidity
  #   generate: flat
  #   params:
  #     level: 80
  #     noise: 5

disease_cases:
  population: 100000
  depends_on:
    - variable: rainfall
      lag: 2          # disease reacts 2 months after rainfall -- change and re-run
      weight: 1.0     # strength of this driver relative to the others
      # transforms:   # reshape this driver (nonlinear / distributed-lag effects):
      #   - { name: threshold, params: { mode: hinge, threshold: 5 } }
      #   - { name: distributed_lag, params: { weights: [0.5, 0.3, 0.2] } }
    # Add a driver for each extra variable you enable above:
    # - variable: mean_temperature
    #   lag: 1
    #   weight: 0.5
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
