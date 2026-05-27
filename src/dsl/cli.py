"""The `dsl` command: wires loader → schema → engine → output together.

Usage:

    dsl run scenario.yaml -o out/

Hard validation errors stop the run with a clear message and a non-zero
exit code BEFORE anything is generated; warnings are printed to stderr
(prefixed ``warning:``) and the run proceeds.
"""
import argparse
import sys

from pydantic import ValidationError

from dsl.core.config.loader import load_yaml
from dsl.core.config.schema import parse_config, validate_scenario
from dsl.core.pipeline.engine import run as run_engine
from dsl.core.pipeline.output import write_output


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

    # --- Generate and write. ---
    df = run_engine(config)
    write_output(df, config, args.out_dir)
    print(f"Wrote {len(df)} rows to {args.out_dir}/")
    return 0


# Allows `python -m dsl.cli` / `python src/dsl/cli.py` during development;
# the installed `dsl` command calls main() via the pyproject entry point.
if __name__ == "__main__":
    sys.exit(main())
