"""The `dsl` command: wires loader → schema → engine → output together.

Usage:

    dsl run scenario.yaml -o out/
    dsl run scenario.yaml              # auto-named folder under out/
    dsl run out/foo/metadata.json      # reproduce a previous run

Hard validation errors stop the run with a clear message and a non-zero
exit code BEFORE anything is generated; warnings are printed to stderr
(prefixed ``warning:``) and the run proceeds.
"""
import argparse
import math
import sys
from pathlib import Path

from pydantic import ValidationError

from dsl.core.config.loader import load_yaml
from dsl.core.config.schema import parse_config, validate_scenario


def _load_scenario_dict(path: str) -> dict:
    """Load a scenario dict from a YAML scenario OR a metadata.json sidecar.

    ``load_yaml`` parses both (JSON is valid YAML). A metadata file wraps the
    real scenario under a ``"scenario"`` key — unwrap it so a dataset can be
    reproduced from its own ``metadata.json``.
    """
    data = load_yaml(path)
    if "scenario" in data and isinstance(data["scenario"], dict):
        return data["scenario"]
    return data


def _resolve_out_dir(input_path: str, out_arg: str | None) -> Path:
    """Decide where to write, never overwriting a previous run by default.

    If ``out_arg`` is given, use it directly (the user's explicit choice,
    overwrite allowed). Otherwise write into ``out/<name>/`` where ``name``
    is the input's stem — or, for a ``metadata.json`` sidecar, its parent
    folder name, so reproducing ``out/foo/metadata.json`` yields ``out/foo``-
    style names rather than ``out/metadata``. The first run is unnumbered;
    if it already exists, the lowest free ``out/<name>_<n>`` (n from 1) wins.
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
from dsl.core.pipeline.chap_check import validate_chap
from dsl.core.pipeline.engine import run as run_engine
from dsl.core.pipeline.metadata import write_metadata
from dsl.core.pipeline.output import write_output
from dsl.core.pipeline.plot import plot_dataset


def main(argv: list[str] | None = None) -> int:
    """Entry point for the console script. Returns the process exit code.

    ``argv`` defaults to the real command line; tests pass a list instead
    so they can invoke the CLI in-process.
    """
    parser = argparse.ArgumentParser(
        prog="dsl",
        description="Generate a synthetic climate-health dataset from a YAML scenario.",
    )
    # Subcommands ("run") leave room for future verbs like "validate".
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run a scenario file")
    run_parser.add_argument(
        "scenario",
        help="path to a scenario YAML file, or a metadata.json to reproduce",
    )
    run_parser.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help=(
            "directory to write into. If omitted, an auto-named folder under "
            "out/ is created so previous runs are never overwritten."
        ),
    )
    run_parser.add_argument(
        "--plot",
        action="store_true",
        help="also write a plot of the dataset into the output directory",
    )
    run_parser.add_argument(
        "--plot-format",
        default="html",
        help="plot file format: html (interactive) or png/svg/pdf (default: html)",
    )
    args = parser.parse_args(argv)

    # --- Parse + validate. Any hard error exits here, before generating. ---
    try:
        config = parse_config(_load_scenario_dict(args.scenario))
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        # Print to stderr (the conventional stream for errors) and return a
        # non-zero code so scripts and CI notice the failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Where to write — an auto-named folder under out/ unless -o was given.
    out_dir = _resolve_out_dir(args.scenario, args.out_dir)

    # Non-fatal warnings: inform and proceed.
    for warning in validate_scenario(config):
        print(f"warning: {warning}", file=sys.stderr)

    # --- Generate, check CHAP compatibility, write. ---
    df = run_engine(config)

    # CHAP-compatibility findings are advisory: print them and proceed.
    # (In practice these only fire when real data enters via from_csv with
    # gaps; the synthetic generators always produce CHAP-valid output.)
    for finding in validate_chap(df):
        print(f"warning: {finding}", file=sys.stderr)

    write_output(df, config, out_dir)
    # The ground-truth sidecar: records the resolved scenario so the dataset
    # is self-describing and reproducible.
    write_metadata(config, out_dir)
    print(f"Wrote {len(df)} rows to {out_dir}/")

    if args.plot:
        # The train/test boundary, as a period index, so the plot can mark it
        # (same floor() rule the output split uses).
        split = (
            math.floor(config.n_total * config.train_fraction)
            if config.train_fraction is not None
            else None
        )
        plot_path = out_dir / f"plot.{args.plot_format}"
        plot_dataset(df, plot_path, train_split=split)
        print(f"Wrote plot to {plot_path}")
    return 0


# Allows `python -m dsl.cli` / `python src/dsl/cli.py` during development;
# the installed `dsl` command calls main() via the pyproject entry point.
if __name__ == "__main__":
    sys.exit(main())
