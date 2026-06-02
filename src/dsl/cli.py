"""The `dsl` command: wires loader → schema → engine → output together.

Usage:

    dsl run scenario.yaml -o out/

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
    run_parser.add_argument("scenario", help="path to the scenario YAML file")
    run_parser.add_argument(
        "-o",
        "--out-dir",
        default="out",
        help="directory to write the CSV files into (default: out/)",
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
        config = parse_config(load_yaml(args.scenario))
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        # Print to stderr (the conventional stream for errors) and return a
        # non-zero code so scripts and CI notice the failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1

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

    write_output(df, config, args.out_dir)
    # The ground-truth sidecar: records the resolved scenario so the dataset
    # is self-describing and reproducible.
    write_metadata(config, args.out_dir)
    print(f"Wrote {len(df)} rows to {args.out_dir}/")

    if args.plot:
        # The train/test boundary, as a period index, so the plot can mark it
        # (same floor() rule the output split uses).
        split = (
            math.floor(config.n_total * config.train_fraction)
            if config.train_fraction is not None
            else None
        )
        plot_path = Path(args.out_dir) / f"plot.{args.plot_format}"
        plot_dataset(df, plot_path, train_split=split)
        print(f"Wrote plot to {plot_path}")
    return 0


# Allows `python -m dsl.cli` / `python src/dsl/cli.py` during development;
# the installed `dsl` command calls main() via the pyproject entry point.
if __name__ == "__main__":
    sys.exit(main())
