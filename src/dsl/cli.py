"""The `dsl` command: wires loader → schema → engine → output together.

Usage:

    dsl run scenario.yaml -o out/
    dsl run scenario.yaml              # auto-named folder under out/
    dsl run out/foo/metadata.json      # reproduce a previous run

Hard validation errors stop the run with a non-zero exit code BEFORE
anything is generated; warnings go to stderr and the run proceeds.
"""
import argparse
import difflib
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
from dsl.watch import watch_loop


def _load_scenario_dict(path: str) -> dict:
    """Load a scenario dict from a YAML scenario OR a metadata.json sidecar.

    ``load_yaml`` parses both (JSON is valid YAML); a metadata file wraps the
    real scenario under a ``"scenario"`` key, so unwrap it.
    """
    data = load_yaml(path)
    if "scenario" in data and isinstance(data["scenario"], dict):
        data = data["scenario"]
    # Resolve relative from_csv paths against the scenario file's directory,
    # so a portable folder (scenario.yaml + data.csv) works from any cwd.
    _resolve_from_csv_paths(data, Path(path).resolve().parent)
    return data


def _resolve_from_csv_paths(scenario: dict, base_dir: Path) -> None:
    """Rewrite relative from_csv `file` params to be relative to base_dir.

    Edits the dict in place. Absolute paths and paths that already exist
    relative to the cwd are left alone.
    """

    def fix(params: dict) -> None:
        file = params.get("file")
        if not isinstance(file, str):
            return
        p = Path(file)
        if p.is_absolute() or p.exists():
            return
        candidate = base_dir / file
        if candidate.exists():
            params["file"] = str(candidate)

    for var in scenario.get("variables", []):
        if isinstance(var, dict) and var.get("generate") == "from_csv":
            fix(var.get("params", {}))
    # Population can also be from_csv (top-level and per-location).
    disease = scenario.get("disease_cases", {})
    pop = disease.get("population")
    if isinstance(pop, dict) and pop.get("generate") == "from_csv":
        fix(pop.get("params", {}))
    locations = scenario.get("locations")
    if isinstance(locations, dict):
        for loc in locations.values():
            lpop = loc.get("population") if isinstance(loc, dict) else None
            if isinstance(lpop, dict) and lpop.get("generate") == "from_csv":
                fix(lpop.get("params", {}))


def _resolve_out_dir(input_path: str, out_arg: str | None) -> Path:
    """Decide where to write, never overwriting a previous run by default.

    An explicit ``out_arg`` is used directly (overwrite allowed). Otherwise
    write into ``out/<name>/`` — for a metadata.json input, ``name`` is its
    parent folder. The first run is unnumbered; after that the lowest free
    ``out/<name>_<n>`` wins.
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


def _scenario_runs(scenario: str) -> list[Path]:
    """Existing out/ folders for ``scenario``: the base first, then _1, _2, …"""
    name = Path(scenario).stem
    out = Path("out")
    if not out.is_dir():
        return []
    runs = [out / name] if (out / name).is_dir() else []
    numbered = []
    for d in out.iterdir():
        if d.is_dir() and d.name.startswith(f"{name}_"):
            suffix = d.name[len(name) + 1 :]
            if suffix.isdigit():
                numbered.append((int(suffix), d))
    runs += [d for _, d in sorted(numbered)]
    return runs


def _prompt_continue(scenario: str) -> Path | None:
    """Ask whether to continue an existing run folder or start a new one.

    Returns the chosen folder, or ``None`` for a new auto-named run. Falls
    back to ``None`` with nothing to continue or no interactive TTY.
    """
    runs = _scenario_runs(scenario)
    if not runs or not sys.stdin.isatty():
        return None
    print(f"Found {len(runs)} existing run(s) for this scenario in out/:")
    for i, d in enumerate(runs, 1):
        print(f"  {i}) {d}")
    print("Continue one of these, or start a new run?")
    print("  [1-{n}] continue · [n] new · (tip: pass --new to skip this)".format(
        n=len(runs)
    ))
    try:
        choice = input("> ").strip().lower()
    except EOFError:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(runs):
        chosen = runs[int(choice) - 1]
        print(f"continuing in {chosen}/ (overwriting it).")
        return chosen
    return None  # "n", empty, or anything else → a new run


# The starter scenario `dsl new` writes. Commented so the file itself
# teaches; it must parse and run with no warnings.
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


def _write_starter(path: Path, force: bool) -> int:
    """Write the starter scenario to ``path``. Refuse to clobber unless forced."""
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


def _friendly_error(exc: ValidationError) -> str:
    """Rewrite a Pydantic ValidationError into a message a scenario author
    can act on: ``extra_forbidden`` (a typo'd key) becomes
    ``unknown field 'x' … did you mean 'y'?``; other error types keep
    Pydantic's own text."""
    from dsl.core.config import schema as _schema

    # Every field name the schema models accept — the suggestion pool.
    valid: set[str] = set()
    for obj in vars(_schema).values():
        fields = getattr(obj, "model_fields", None)
        if isinstance(fields, dict):
            valid.update(fields)

    lines = []
    for err in exc.errors():
        if err["type"] == "extra_forbidden" and err["loc"]:
            key = str(err["loc"][-1])
            where = ".".join(str(p) for p in err["loc"][:-1])
            at = f" in {where}" if where else ""
            near = difflib.get_close_matches(key, valid, n=1)
            hint = f" — did you mean '{near[0]}'?" if near else ""
            lines.append(f"unknown field '{key}'{at}{hint}")
        else:
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(lines)


def _list_blocks() -> int:
    """Print the registered generators and transforms for discovery."""
    import dsl.generators  # noqa: F401  (import triggers registration)
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


def _run_once(
    scenario: str,
    out_dir: Path,
    plot: bool,
    plot_format: str,
    seed_override: int | None = None,
) -> int:
    """Parse → generate → write (and optionally plot) one scenario.

    Returns a process exit code. Shared by the one-shot path and ``--watch``.
    ``seed_override`` replaces the scenario's seed (used for replicates), so
    the written metadata records the actual seed.
    """
    # Parse + validate. Any hard error exits here, before generating.
    try:
        config = parse_config(_load_scenario_dict(scenario))
        if seed_override is not None:
            config = config.model_copy(update={"seed": seed_override})
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        msg = _friendly_error(exc) if isinstance(exc, ValidationError) else exc
        print(f"error: {msg}", file=sys.stderr)
        return 1

    for warning in validate_scenario(config):
        print(f"warning: {warning}", file=sys.stderr)

    # Generation can fail on input the schema can't catch (a bad generator
    # param, a malformed from_csv source). Surface those as clean CLI errors;
    # let genuine programming bugs propagate.
    try:
        df = run_engine(config)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # CHAP-compatibility findings are advisory: print and proceed.
    for finding in validate_chap(df):
        print(f"warning: {finding}", file=sys.stderr)

    write_output(df, config, out_dir)
    write_metadata(config, out_dir)  # the ground-truth sidecar
    print(f"Wrote {len(df)} rows to {out_dir}/")

    if plot:
        # Train/test boundary as a period index (same floor() rule the
        # output split uses), so the plot can mark it.
        split = (
            math.floor(config.n_total * config.train_fraction)
            if config.train_fraction is not None
            else None
        )
        plot_path = out_dir / f"plot.{plot_format}"
        plot_dataset(df, plot_path, train_split=split)
        print(f"Wrote plot to {plot_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsl",
        description="Generate a synthetic climate-health dataset from a YAML scenario.",
    )
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
    return parser


def _run_replicates(args, out_dir: Path) -> int:
    """Run the scenario N times with seeds base, base+1, … into rep_NN/ dirs."""
    if args.watch:
        print("error: --watch cannot be combined with --replicates", file=sys.stderr)
        return 1
    # Read the base seed once so replicate i can use base + i.
    try:
        base_seed = parse_config(_load_scenario_dict(args.scenario)).seed
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        msg = _friendly_error(exc) if isinstance(exc, ValidationError) else exc
        print(f"error: {msg}", file=sys.stderr)
        return 1
    for i in range(args.replicates):
        code = _run_once(
            args.scenario, out_dir / f"rep_{i:02d}", args.plot, args.plot_format,
            seed_override=base_seed + i,
        )
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the console script. Returns the process exit code.

    ``argv`` defaults to the real command line; tests pass a list to invoke
    the CLI in-process.
    """
    args = _build_parser().parse_args(argv)

    if args.command == "new":
        return _write_starter(Path(args.path), args.force)

    if args.command == "list":
        return _list_blocks()

    # Where to write. With no -o and no --new, offer to continue an existing
    # run rather than spawning out/<name>_1, _2, … An explicit -o always
    # wins; a non-interactive shell falls back to auto-numbering too.
    if args.out_dir is None and not args.new:
        out_dir = _prompt_continue(args.scenario) or _resolve_out_dir(
            args.scenario, None
        )
    else:
        out_dir = _resolve_out_dir(args.scenario, args.out_dir)

    if args.replicates < 1:
        print("error: --replicates must be >= 1", file=sys.stderr)
        return 1
    if args.replicates > 1:
        return _run_replicates(args, out_dir)

    code = _run_once(args.scenario, out_dir, args.plot, args.plot_format)
    if code != 0 or not args.watch:
        return code
    # --watch keeps the same out_dir so each save overwrites in place.
    return watch_loop(args.scenario, out_dir, args.plot, args.plot_format, _run_once)


if __name__ == "__main__":
    sys.exit(main())
