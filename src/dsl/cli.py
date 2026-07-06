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
import functools
import http.server
import math
import sys
import threading
import time
import webbrowser
from pathlib import Path

from pydantic import ValidationError

from dsl.core.config.loader import load_yaml
from dsl.core.config.schema import parse_config, validate_scenario
from dsl.core.pipeline.chap_check import validate_chap
from dsl.core.pipeline.engine import run as run_engine
from dsl.core.pipeline.metadata import write_metadata
from dsl.core.pipeline.output import write_output
from dsl.core.pipeline.plot import plot_dataset


def _load_scenario_dict(path: str) -> dict:
    """Load a scenario dict from a YAML scenario OR a metadata.json sidecar.

    ``load_yaml`` parses both (JSON is valid YAML). A metadata file wraps the
    real scenario under a ``"scenario"`` key — unwrap it so a dataset can be
    reproduced from its own ``metadata.json``.
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
    relative to the cwd are left alone (the latter keeps old behavior).
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
    # Population can also be a from_csv generator (top-level and per-location).
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


def _scenario_runs(scenario: str) -> list[Path]:
    """Existing out/ folders belonging to ``scenario`` (its base + _N siblings).

    Sorted: the base ``out/<name>`` first, then ``_1``, ``_2``, … numerically.
    """
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

    Returns the chosen folder, or ``None`` to start a new (auto-named) run.
    Used by ``dsl run`` when neither ``-o`` nor ``--new`` is given; falls back
    to ``None`` when there's nothing to continue or there's no interactive TTY.
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
        f"  uv run dsl run {path} --plot --watch\n"
        f"then edit the file and save to see the data update."
    )
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
    ``seed_override`` replaces the scenario's seed (used for replicates), so the
    written metadata records the actual seed and stays reproducible.
    """
    # --- Parse + validate. Any hard error exits here, before generating. ---
    try:
        config = parse_config(_load_scenario_dict(scenario))
        if seed_override is not None:
            config = config.model_copy(update={"seed": seed_override})
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        # Print to stderr (the conventional stream for errors) and return a
        # non-zero code so scripts and CI notice the failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Non-fatal warnings: inform and proceed.
    for warning in validate_scenario(config):
        print(f"warning: {warning}", file=sys.stderr)

    # --- Generate, check CHAP compatibility, write. ---
    # Generation/output can fail on input the schema can't catch (a bad
    # generator param, a malformed from_csv source). Surface those as clean
    # CLI errors; let genuine programming bugs propagate.
    try:
        df = run_engine(config)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # CHAP-compatibility findings are advisory: print them and proceed.
    for finding in validate_chap(df):
        print(f"warning: {finding}", file=sys.stderr)

    write_output(df, config, out_dir)
    # The ground-truth sidecar: records the resolved scenario so the dataset
    # is self-describing and reproducible.
    write_metadata(config, out_dir)
    print(f"Wrote {len(df)} rows to {out_dir}/")

    if plot:
        # The train/test boundary, as a period index, so the plot can mark it
        # (same floor() rule the output split uses).
        split = (
            math.floor(config.n_total * config.train_fraction)
            if config.train_fraction is not None
            else None
        )
        plot_path = out_dir / f"plot.{plot_format}"
        plot_dataset(df, plot_path, train_split=split)
        print(f"Wrote plot to {plot_path}")
    return 0


def _changed(path: str, last_mtime: float) -> tuple[bool, float]:
    """Has ``path``'s mtime advanced past ``last_mtime``? Returns (changed, mtime).

    A missing file (mid-save by some editors) counts as unchanged.
    """
    try:
        mtime = Path(path).stat().st_mtime
    except FileNotFoundError:
        return False, last_mtime
    return mtime > last_mtime, mtime


# Injected into plot.html under --watch: poll a version the server bumps on each
# re-run and reload only when it actually changes (so the page doesn't flicker
# or reset zoom/pan between edits).
_RELOAD_SCRIPT = """
<script>
(function () {
  let last = null;
  setInterval(async function () {
    try {
      const v = await (await fetch("/__plot_version__", {cache: "no-store"})).text();
      if (last !== null && v !== last) location.reload();
      last = v;
    } catch (e) { /* server gone (watch stopped): stop trying */ }
  }, 700);
})();
</script>
"""


def _inject_reload(html_path: Path) -> None:
    """Append the live-reload script to a written plot.html (idempotent)."""
    html = html_path.read_text()
    if "__plot_version__" not in html:
        html = html.replace("</body>", _RELOAD_SCRIPT + "</body>", 1)
        html_path.write_text(html)


def _serve(out_dir: Path, version: list[int]) -> "http.server.HTTPServer":
    """Serve ``out_dir`` on a free localhost port; expose the reload version.

    ``version`` is a one-element list shared with the watch loop — the handler
    reads ``version[0]`` so a re-run can bump it without restarting the server.
    """
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (stdlib's required name)
            if self.path == "/__plot_version__":
                body = str(version[0]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *args):  # silence per-request logging
            pass

    handler = functools.partial(Handler, directory=str(out_dir))
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _watch_loop(
    scenario: str, out_dir: Path, plot: bool, plot_format: str
) -> int:
    """Re-run ``scenario`` whenever its file is saved, until interrupted.

    With an HTML plot, also serve it on localhost and live-reload the browser
    on each successful re-run; otherwise just regenerate the files in place.
    """
    last_mtime = Path(scenario).stat().st_mtime
    serve = plot and plot_format == "html"
    server = None
    version = [0]
    if serve:
        plot_path = out_dir / "plot.html"
        _inject_reload(plot_path)
        server = _serve(out_dir, version)
        url = f"http://127.0.0.1:{server.server_address[1]}/plot.html"
        webbrowser.open(url)
        print(f"serving plot at {url} — it reloads on every save.")
    print("watching for changes — edit the scenario and save, or Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.5)
            changed, last_mtime = _changed(scenario, last_mtime)
            if changed and _run_once(scenario, out_dir, plot, plot_format) == 0:
                if serve:
                    _inject_reload(out_dir / "plot.html")
                    version[0] += 1  # tells the open page to reload
    except KeyboardInterrupt:
        print("\nstopped watching.")
        return 0
    finally:
        if server is not None:
            server.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the console script. Returns the process exit code.

    ``argv`` defaults to the real command line; tests pass a list instead
    so they can invoke the CLI in-process.
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

    # Where to write. With no -o and no --new, offer to continue an existing run
    # rather than spawning out/<name>_1, _2, … An explicit -o always wins; --new
    # forces fresh auto-numbering; a non-interactive shell falls back to it too.
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
        if args.watch:
            print("error: --watch cannot be combined with --replicates", file=sys.stderr)
            return 1
        # Read the base seed once so replicate i can use base + i.
        try:
            base_seed = parse_config(_load_scenario_dict(args.scenario)).seed
        except (FileNotFoundError, ValueError, ValidationError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for i in range(args.replicates):
            rep_dir = out_dir / f"rep_{i:02d}"
            code = _run_once(
                args.scenario, rep_dir, args.plot, args.plot_format,
                seed_override=base_seed + i,
            )
            if code != 0:
                return code
        return 0

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
