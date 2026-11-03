"""Execute a scenario: run once, run a sweep, or watch for changes.

This module owns everything that happens after the output directory is
decided: parse-generate-write one scenario, run every combination a swept
scenario stands for into one directory each, or re-run it automatically on
every save and (for HTML plots) live-reload a browser tab.
"""
import argparse
import functools
import http.server
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

from pydantic import ValidationError

from dsl._output import _PLOT_FORMATS, _numbered_dir_suffix, _run_name
from dsl._scenario import (
    _friendly_error,
    _load_and_parse,
    _load_scenario_dict,
    _parse_dict,
)
from dsl._sweep import expand_sweeps
from dsl.core.config.schema import ScenarioConfig, validate_scenario
from dsl.core.pipeline.dataset_rules import check_dataset
from dsl.core.pipeline.engine import run as run_engine
from dsl.core.pipeline.folds import evaluation_boundary, write_report
from dsl.core.pipeline.metadata import write_metadata
from dsl.core.pipeline.output import write_output
from dsl.core.pipeline.plot import plot_dataset


def _run_once(
    scenario: str,
    out_dir: Path,
    plot: bool,
    plot_format: str,
    config: ScenarioConfig | None = None,
    tick_every: int | None = None,
    report_dir: Path | None = None,
    quiet_warnings: bool = False,
) -> int:
    """Parse → generate → write (and optionally plot) one scenario.

    Shared by the one-shot path and ``--watch``.

    Args:
        scenario: Path to the scenario YAML file or metadata.json.
        out_dir: Directory to write simulated_data.csv, metadata.json, and
            (if plot) the plot file into.
        plot: Whether to also write a plot of the dataset.
        plot_format: Plot file format (e.g. "html", "png").
        config: An already-parsed ScenarioConfig (e.g. from a sweep that
            parsed the file once per combination), to skip re-parsing
            ``scenario`` from disk.
        tick_every: Plot gridline spacing in periods. Defaults to one year.

    Returns:
        Exit code: 0 on success, 1 if parsing/validation or generation fails.

    Errors Caught (logged to stderr):
        FileNotFoundError: If the scenario file doesn't exist.
        ValueError: If the scenario is invalid, or generation hits a bad
            generator param or malformed from_csv source.
        ValidationError: If pydantic validation fails.
        KeyError: If a generator name is not registered.
    """
    # --- Parse + validate. Any hard error exits here, before generating. ---
    try:
        if config is None:
            config = _load_and_parse(scenario)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        # Print to stderr (the conventional stream for errors) and return a
        # non-zero code so scripts and CI notice the failure. A ValidationError
        # is rewritten into scenario-author-friendly text.
        msg = _friendly_error(exc) if isinstance(exc, ValidationError) else exc
        print(f"error: {msg}", file=sys.stderr)
        return 1

    # Non-fatal warnings: inform and proceed.
    if not quiet_warnings:
        # A sweep varies scalar values, not the scenario's shape, so these
        # would otherwise repeat identically N times and bury the per-run
        # output.
        for warning in validate_scenario(config):
            print(f"warning: {warning}", file=sys.stderr)

    # Generate, check the dataset rules, write. Generation can fail on input
    # the schema can't catch (a bad generator param, a malformed from_csv
    # source) — surface those as clean CLI errors, not a raw traceback.
    try:
        df = run_engine(config)

        # Dataset-rule findings are advisory: print them and proceed.
        # The config knows which columns are counts; the frame alone
        # cannot tell a second disease from a driver.
        for finding in check_dataset(
            df, [spec.name for spec in config.series if spec.counts]
        ):
            print(f"warning: {finding}", file=sys.stderr)

        # Where to SAY the files went: a multi-run set writes into a
        # staging folder and swaps it in, so naming the real path would
        # point the reader at one that is gone by the time they read it.
        shown = report_dir if report_dir is not None else out_dir

        # The plot names this tool writes, by name rather than by glob:
        # `plot.*` also matched a user's own plot.py or plot.csv under
        # `-o .`, and deleted them. Before the data lands, because a step
        # after this one can still fail and leave the old picture beside
        # new data.
        for name in _PLOT_NAMES:
            (out_dir / name).unlink(missing_ok=True)

        write_output(df, config, out_dir)
        # The ground-truth sidecar goes down with the data, before the
        # report: the report re-runs the generators and so can fail on its
        # own, and a folder holding a CSV with no metadata.json is a
        # dataset that cannot say what made it or reproduce itself.
        write_metadata(config, out_dir)
        # What each fold actually contains, so a fold that cannot measure
        # what the scenario planted is visible rather than silently scored
        # well. This re-runs the generators to read what they planted, so it
        # can hit the same input errors generation can — and under --watch an
        # escaping exception would kill the watch loop.
        write_report(config, df, out_dir)
        print(f"Wrote {len(df)} rows to {shown}/")

        if plot:
            # Mark where evaluation begins, so the part a model is scored on
            # is visible in the chart.
            plot_path = out_dir / f"plot.{plot_format}"
            try:
                plot_dataset(
                    df,
                    plot_path,
                    train_split=evaluation_boundary(config),
                    period=config.period,
                    tick_every=tick_every,
                    units={
                        spec.name: spec.unit
                        for spec in config.series
                        if spec.unit
                    },
                )
            except Exception as exc:  # noqa: BLE001 — see below
                # The data is already written and valid; only the picture
                # failed. Static formats go through kaleido, which needs a
                # browser and raises whatever that failure happens to be —
                # RuntimeError, OSError, something from the backend — so
                # the type cannot be enumerated. Letting any of them through
                # would throw away a good run, and end a --watch session
                # with a traceback on a machine that simply has no chrome.
                print(f"warning: could not write the plot ({exc})", file=sys.stderr)
            else:
                print(f"Wrote plot to {shown / plot_path.name}")
    except Exception as exc:  # noqa: BLE001 — see below
        # Deliberately everything. This function's contract is "report the
        # problem and return 1", and the callers depend on it: --watch
        # keeps going so the author can fix the scenario and save again,
        # and a multi-run set reports which one failed. Enumerating the
        # types meant each new one — OSError from a full disk, a pandas
        # error from a malformed CSV, an IndexError from a generator —
        # ended a watch session with a traceback until someone hit it.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


# Where a multi-run set is built before being swapped in. Left behind
# only by a run killed mid-write, and cleared by the next one.
_STAGING_NAME = ".partial"

# Every plot file a run can write, derived from the formats the CLI
# accepts so the two cannot drift — the hand-written list had fallen
# behind on jpg and jpeg.
_PLOT_NAMES = frozenset(f"plot.{ext}" for ext in (*_PLOT_FORMATS, "json"))

# Everything a run writes into its output directory. Anything else found
# there belongs to the user, so the set cleanup must not touch it.
_RUN_OUTPUT_NAMES = frozenset(
    {"simulated_data.csv", "metadata.json", "folds"}
) | _PLOT_NAMES

# Written at the root of a multi-run set rather than inside a run, so the
# foreign check must recognise them as this tool's own output.
_SET_INDEX_NAMES = frozenset({"sweep.json"})


def _foreign_entries(out_dir: Path, keep: set[str]) -> list[str]:
    """Names in ``out_dir`` that this tool never writes, sorted.

    A multi-run set deletes whatever it will not overwrite, so it may only
    run in a folder that holds output and nothing else. ``-o .`` on a
    working directory is the case this guards: the scenario file itself, a
    thesis, a src/ tree.

    A leftover run directory from a previous set counts as output: it holds
    a ``metadata.json``, which only this tool writes. Run directories are
    named after the values they swept, so there is no name shape to go on.

    Args:
        out_dir: The directory about to be cleaned.
        keep: The run-directory names this run will write.

    Returns:
        Up to five foreign names (with a trailing "…" entry if there are
        more), or an empty list when the directory is safe to clean.
    """
    foreign = sorted(
        entry.name
        for entry in out_dir.iterdir()
        if not (entry.name in keep and entry.is_dir())
        and entry.name not in _RUN_OUTPUT_NAMES
        and entry.name not in _SET_INDEX_NAMES
        and entry.name != _STAGING_NAME
        and not (entry / "metadata.json").is_file()
    )
    return foreign[:5] + ["…"] if len(foreign) > 5 else foreign


def _run_set(
    args: argparse.Namespace,
    out_dir: Path,
    runs: list[tuple[str, ScenarioConfig]],
) -> int:
    """Write a named set of runs into ``out_dir``, one subdirectory each.

    Each set is staged in ``.partial`` and swapped in, so a failure partway
    never replaces a good set with half of one.

    Args:
        args: Parsed CLI args (plot, plot_format, scenario).
        out_dir: Parent directory; each run writes into out_dir/<label>/.
        runs: (directory name, already-parsed config) in the order to run.

    Returns:
        Exit code: 0 if every run succeeds, 1 on the first failure.
    """
    # Checked before anything is written: refusing a folder the tool does
    # not own has to happen while it is still untouched.
    keep = {label for label, _ in runs}
    if out_dir.is_dir():
        foreign = _foreign_entries(out_dir, keep)
        if foreign:
            print(
                f"error: {out_dir}/ holds files this tool did not write "
                f"({', '.join(foreign)}), so it will not be cleaned.\n"
                f"       a multi-run set clears the output folder of stale "
                f"results before it writes.\n"
                f"       Point -o at a new or empty folder, or remove those "
                f"entries first.",
                file=sys.stderr,
            )
            return 1
    # Staged, then swapped. Clearing up front destroyed the previous
    # results before anything replaced them; writing in place instead left
    # a failed run's first new directory sitting among an older set's
    # remaining ones, which still looks like one valid set from the
    # outside. Neither is a thing this tool should hand to someone.
    staging = out_dir / _STAGING_NAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for i, (label, config) in enumerate(runs):
            code = _run_once(
                args.scenario, staging / label, args.plot,
                args.plot_format,
                config=config,
                tick_every=getattr(args, "plot_ticks", None),
                report_dir=out_dir / label,
                quiet_warnings=i > 0,
            )
            if code != 0:
                return code

        # A complete set exists, so the old one can go. Everything here is
        # output this tool wrote — the foreign check above already refused
        # the folder otherwise.
        #
        # Each new directory is moved into place over the old one FIRST, so
        # a kill partway leaves a mix of complete runs rather than a gap.
        # Only then is whatever the new set does not cover removed: extra
        # directories from a wider previous sweep, and a single run's CSVs
        # from a plain run into the same folder. A true atomic swap would need
        # to exchange two directories, which no portable call offers.
        staged = sorted(staging.iterdir())
        for fresh in staged:
            landing = out_dir / fresh.name
            if landing.is_dir():
                shutil.rmtree(landing)
            fresh.rename(landing)
        written = {run_dir.name for run_dir in staged}
        for stale in out_dir.iterdir():
            if stale == staging or stale.name in written:
                continue
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


def _changed(path: str, last_mtime: float) -> tuple[bool, float]:
    """Has ``path``'s mtime moved since ``last_mtime``?

    A missing file (mid-save by some editors) counts as unchanged.

    Args:
        path: File to check.
        last_mtime: The mtime last observed.

    Returns:
        (changed, mtime): whether it changed, and the current mtime to
        remember for the next check (unchanged from last_mtime if missing).
    """
    try:
        # Nanoseconds, and != rather than >: an editor that writes twice
        # within one clock tick, or restores an older copy, still counts as
        # a change — and a save the watcher misses is a save the author
        # waits for forever.
        mtime = Path(path).stat().st_mtime_ns
    except FileNotFoundError:
        return False, last_mtime
    return mtime != last_mtime, mtime


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
    """Append the live-reload script to a written plot.html (idempotent).

    Does nothing when the file is absent: a failed run never wrote one, and
    it has already reported why — crashing here would kill the watch loop
    before the author could fix the scenario.

    Args:
        html_path: Path to the plot.html file to patch in place.
    """
    if not html_path.is_file():
        return
    html = html_path.read_text(encoding="utf-8")
    if "__plot_version__" not in html:
        # The LAST </body>: plotly's own output can embed one earlier, and
        # a script injected before the real end of the document may not
        # run. rfind returns -1 when there is none, which would splice the
        # script in before the final character, so append instead.
        cut = html.rfind("</body>")
        if cut == -1:
            html += _RELOAD_SCRIPT
        else:
            html = html[:cut] + _RELOAD_SCRIPT + html[cut:]
        html_path.write_text(html, encoding="utf-8")


def _serve(out_dir: Path, version: list[int]) -> "http.server.ThreadingHTTPServer":
    """Serve ``out_dir`` on a free localhost port; expose the reload version.

    ``version`` is a one-element list shared with the watch loop — the handler
    reads ``version[0]`` so a re-run can bump it without restarting the server.

    Args:
        out_dir: Directory to serve as static files (contains plot.html).
        version: One-element list; version[0] is polled by the injected
            reload script and bumped by the watch loop after each re-run.

    Returns:
        The running server (already serving in a daemon thread).
    """
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
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
    # Threading: a single-threaded server blocks on one slow request,
    # and the page polls for a version bump while also fetching the
    # plot — so one stalled fetch could freeze the reload.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _watch_loop(
    scenario: str,
    out_dir: Path,
    plot: bool,
    plot_format: str,
    tick_every: int | None = None,
) -> int:
    """Re-run ``scenario`` whenever its file is saved, until interrupted.

    With an HTML plot, also serve it on localhost and live-reload the browser
    on each successful re-run; otherwise just regenerate the files in place.

    Args:
        scenario: Path to the scenario YAML file being watched.
        out_dir: Directory each re-run writes into (overwritten in place).
        plot: Whether to also write (and, if html, serve) a plot.
        plot_format: Plot file format (e.g. "html", "png").
        tick_every: Plot gridline spacing in periods. Defaults to one year.

    Returns:
        Exit code: 0 (Ctrl-C is the normal way to stop watching).
    """
    # st_mtime_ns, matching what _changed returns and compares — seeding
    # this with float seconds made the two never equal, so the loop
    # re-ran once at startup without anyone saving anything.
    try:
        last_mtime = Path(scenario).stat().st_mtime_ns
    except FileNotFoundError:
        # An editor that saves by rename-and-replace can make the file
        # vanish for a moment. _changed already tolerates that; this stat
        # sits outside the loop's handlers, so it has to as well.
        last_mtime = 0
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
            if (
                changed
                and _run_once(
                    scenario, out_dir, plot, plot_format, tick_every=tick_every
                ) == 0
                and serve
            ):
                _inject_reload(out_dir / "plot.html")
                version[0] += 1  # tells the open page to reload
    except KeyboardInterrupt:
        print("\nstopped watching.")
        return 0
    finally:
        if server is not None:
            server.shutdown()
            # Releases the listening socket; without it the port stays
            # bound for the rest of the process.
            server.server_close()


def _prompt_continue(scenario: str) -> Path | None:
    """Ask whether to continue an existing run folder or start a new one.

    Used by ``dsl run`` when neither ``-o`` nor ``--new`` is given.

    Args:
        scenario: Path to the scenario file (used to find existing runs).

    Returns:
        The chosen folder to continue (and overwrite), or None to start a
        new auto-named run. None when there's nothing to continue, there's
        no interactive TTY, or the user picks "new"/anything but a number.
    """
    runs = _scenario_runs(scenario)
    if not runs or not sys.stdin.isatty():
        return None
    print(f"Found {len(runs)} existing run(s) for this scenario in out/:")
    for i, d in enumerate(runs, 1):
        print(f"  {i}) {d}")
    print("Continue one of these, or start a new run?")
    print(f"  [1-{len(runs)}] continue · [n] new · (tip: pass --new to skip this)")
    try:
        choice = input("> ").strip().lower()
    except EOFError:
        return None
    if choice.isdecimal() and 1 <= int(choice) <= len(runs):
        chosen = runs[int(choice) - 1]
        print(f"continuing in {chosen}/ (overwriting it).")
        return chosen
    return None  # "n", empty, or anything else → a new run


def _scenario_runs(scenario: str) -> list[Path]:
    """Existing out/ folders belonging to ``scenario`` (its base + _N siblings).

    Args:
        scenario: Path to the scenario file, or a metadata.json to reproduce.

    Returns:
        Existing run directories, sorted: base ``out/<name>`` first, then
        ``_1``, ``_2``, … numerically. Empty list if out/ doesn't exist.
    """
    # Must match _resolve_out_dir: a metadata.json names its parent folder,
    # not the file, or the prompt searches for out/metadata and finds nothing.
    path = Path(scenario)
    name = _run_name(path)
    out = Path("out")
    if not out.is_dir():
        return []
    runs = [out / name] if (out / name).is_dir() else []
    numbered = []
    for d in out.iterdir():
        n = _numbered_dir_suffix(d, f"{name}_")
        if n is not None:
            numbered.append((n, d))
    runs += [d for _, d in sorted(numbered)]
    return runs


def _run_sweep(args: argparse.Namespace, out_dir: Path) -> int:
    """Run every combination a swept scenario stands for.

    ``main`` calls this when the loaded scenario holds a ``sweep:``. Every
    variant is parsed up front, so a scenario that is only invalid at the
    third combination fails before the first one is written.

    Args:
        args: Parsed CLI args (scenario, plot, plot_format, watch).
        out_dir: Parent directory; each combination writes into out_dir/<label>/.

    Returns:
        Exit code: 0 if every combination succeeds, 1 otherwise.

    Errors Caught (logged to stderr):
        FileNotFoundError: If the scenario file doesn't exist.
        ValueError: If a sweep spec or a resulting scenario is invalid.
        ValidationError: If pydantic validation fails on any combination.
    """
    if args.watch:
        print("error: --watch cannot be combined with a sweep", file=sys.stderr)
        return 1
    try:
        runs = [
            (label, _parse_dict(variant))
            for label, variant in expand_sweeps(_load_scenario_dict(args.scenario))
        ]
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        msg = _friendly_error(exc) if isinstance(exc, ValidationError) else exc
        print(f"error: {msg}", file=sys.stderr)
        return 1
    print(f"Sweeping {len(runs)} runs into {out_dir}/")
    return _run_set(args, out_dir, runs)
