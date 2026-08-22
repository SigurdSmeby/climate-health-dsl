"""The `dsl` command: wires loader → schema → engine → output together.

Usage:

    dsl run scenario.yaml -o out/
    dsl run scenario.yaml              # auto-named folder under out/
    dsl run out/foo/metadata.json      # reproduce a previous run

Hard validation errors stop the run with a clear message and a non-zero
exit code BEFORE anything is generated; warnings are printed to stderr
(prefixed ``warning:``) and the run proceeds.

The implementation is split across a few internal modules:
``_scenario.py`` (load/parse/validate), ``_run.py`` (execute: once,
replicates, watch), and ``_output.py`` (output directory + starter
template). This file owns only argument parsing and dispatch.
"""
import argparse
import sys
from pathlib import Path

from dsl._output import _resolve_out_dir, _write_starter
from dsl._run import _prompt_continue, _run_once, _run_replicates, _watch_loop


def _list_blocks() -> int:
    """Print the registered generators and transforms (names) for discovery.

    Returns:
        Exit code: always 0.
    """
    import dsl.generators  # import triggers registration
    import dsl.transforms  # noqa: F401
    from dsl.core.extension.generator_base import generator_registry
    from dsl.core.extension.transform_base import transform_registry

    print("generators (variables -> generate:):")
    for name in generator_registry.names():
        print(f"  {name}")
    print("transforms (depends_on[].transforms / series modifiers):")
    for name in transform_registry.names():
        print(f"  {name}")
    print("\nParams for each: see docs/GUIDE.md.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the console script.

    ``argv`` defaults to the real command line; tests pass a list instead
    so they can invoke the CLI in-process.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="dsl",
        description="Generate a synthetic climate-health dataset from a YAML scenario.",
    )
    # Subcommands ("run", "new") leave room for future verbs like "validate".
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser(
        "new", help="write a commented starter scenario to edit"
    )
    new_parser.add_argument(
        "path",
        nargs="?",
        default="scenario.yaml",
        help="where to write the starter file (default: scenario.yaml)",
    )
    new_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite the file if it already exists",
    )

    subparsers.add_parser(
        "list", help="list the registered generators and transforms"
    )

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
    run_parser.add_argument(
        "--watch",
        action="store_true",
        help="re-run automatically whenever the scenario file is saved",
    )
    run_parser.add_argument(
        "--new",
        action="store_true",
        help="skip the continue prompt; always write a fresh auto-named folder",
    )
    run_parser.add_argument(
        "--replicates",
        "-n",
        type=int,
        default=1,
        help=(
            "generate N replicates with seeds base, base+1, … into rep_00/, "
            "rep_01/, … (each independently reproducible). Default 1 = a single "
            "run written directly to the output folder."
        ),
    )
    args = parser.parse_args(argv)

    if args.command == "new":
        return _write_starter(Path(args.path), args.force)

    if args.command == "list":
        return _list_blocks()

    # Cheap flag validation before anything interactive/side-effecting below.
    if args.replicates < 1:
        print("error: --replicates must be >= 1", file=sys.stderr)
        return 1

    # Where to write. With no -o and no --new, offer to continue an existing run
    # rather than spawning out/<name>_1, _2, … An explicit -o always wins; --new
    # forces fresh auto-numbering; a non-interactive shell falls back to it too.
    if args.out_dir is None and not args.new:
        out_dir = _prompt_continue(args.scenario) or _resolve_out_dir(
            args.scenario, None
        )
    else:
        out_dir = _resolve_out_dir(args.scenario, args.out_dir)

    if args.replicates > 1:
        return _run_replicates(args, out_dir)

    code = _run_once(args.scenario, out_dir, args.plot, args.plot_format)
    if code != 0 or not args.watch:
        return code
    # --watch keeps the same out_dir so each save overwrites in place (the
    # browser re-reads plot.html). A first-run failure already returned above.
    return _watch_loop(args.scenario, out_dir, args.plot, args.plot_format)


# Allows `python -m dsl.cli` / `python src/dsl/cli.py` during development;
# the installed `dsl` command calls main() via the pyproject entry point.
if __name__ == "__main__":
    sys.exit(main())
