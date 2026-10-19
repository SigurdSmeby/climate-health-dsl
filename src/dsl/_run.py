"""Execute a scenario: run once, run replicates, or watch for changes.

This module owns everything that happens after the output directory is
decided: parse-generate-write one scenario, run it N times into rep_NN/
dirs with varying seeds, or re-run it automatically on every save and
(for HTML plots) live-reload a browser tab.
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

from dsl._output import _numbered_dir_suffix
from dsl._scenario import _friendly_error, _load_and_parse
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
    seed_override: int | None = None,
    config: ScenarioConfig | None = None,
    tick_every: int | None = None,
) -> int:
    """Parse → generate → write (and optionally plot) one scenario.

    Shared by the one-shot path and ``--watch``.

    Args:
        scenario: Path to the scenario YAML file or metadata.json.
        out_dir: Directory to write simulated_data.csv, metadata.json, and
            (if plot) the plot file into.
        plot: Whether to also write a plot of the dataset.
        plot_format: Plot file format (e.g. "html", "png").
        seed_override: Passed to _load_and_parse when config is None.
        config: An already-parsed ScenarioConfig (e.g. from a replicate loop
            that parsed once and only varies the seed), to skip re-parsing
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
            config = _load_and_parse(scenario, seed_override)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        # Print to stderr (the conventional stream for errors) and return a
        # non-zero code so scripts and CI notice the failure. A ValidationError
        # is rewritten into scenario-author-friendly text.
        msg = _friendly_error(exc) if isinstance(exc, ValidationError) else exc
        print(f"error: {msg}", file=sys.stderr)
        return 1

    # Non-fatal warnings: inform and proceed.
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
        print(f"Wrote {len(df)} rows to {out_dir}/")

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
                # A plot left from an earlier run would otherwise sit beside
                # data it does not describe — and under --watch the browser
                # would reload it as if it were fresh.
                plot_path.unlink(missing_ok=True)
            else:
                print(f"Wrote plot to {plot_path}")
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        # OSError too: a full disk or an unwritable output folder is an
        # ordinary thing to hit, and under --watch a traceback ends the
        # session instead of letting the author free some space and save
        # again. (FileNotFoundError is an OSError, kept for clarity.)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


# Everything a run writes into its output directory. Anything else found
# there belongs to the user, so the replicate cleanup must not touch it.
_RUN_OUTPUT_NAMES = frozenset(
    {"simulated_data.csv", "metadata.json", "folds"}
) | frozenset(f"plot.{ext}" for ext in ("html", "png", "svg", "pdf", "json"))


def _foreign_entries(out_dir: Path, keep: set[str]) -> list[str]:
    """Names in ``out_dir`` that this tool never writes, sorted.

    The replicate cleanup deletes whatever it will not overwrite, so it may
    only run in a folder that holds output and nothing else. ``-o .`` on a
    working directory is the case this guards: the scenario file itself, a
    thesis, a src/ tree.

    Args:
        out_dir: The directory about to be cleaned.
        keep: The ``rep_NN`` names this run will write.

    Returns:
        Up to five foreign names (with a trailing "…" entry if there are
        more), or an empty list when the directory is safe to clean.
    """
    foreign = sorted(
        entry.name
        for entry in out_dir.iterdir()
        if entry.name not in keep
        and entry.name not in _RUN_OUTPUT_NAMES
        and _numbered_dir_suffix(entry, "rep_") is None
    )
    return foreign[:5] + ["…"] if len(foreign) > 5 else foreign


def _run_replicates(args: argparse.Namespace, out_dir: Path) -> int:
    """Run the scenario N times with seeds base, base+1, … into rep_NN/ dirs.

    Args:
        args: Parsed CLI args (scenario, replicates, plot, plot_format, watch).
        out_dir: Parent directory; each replicate writes into out_dir/rep_NN/.

    Returns:
        Exit code: 0 if all replicates succeed, 1 on the first failure
        (--watch combined with --replicates is also a failure).

    Errors Caught (logged to stderr):
        FileNotFoundError: If the scenario file doesn't exist.
        ValueError: If the scenario is invalid.
        ValidationError: If pydantic validation fails.
    """
    if args.watch:
        print("error: --watch cannot be combined with --replicates", file=sys.stderr)
        return 1
    # Parse once — every replicate is the same scenario, differing only by
    # seed, so there's no need to re-read and re-validate the file per
    # replicate (or, for a from_csv-backed scenario, re-read its CSV source).
    try:
        base_config = _load_and_parse(args.scenario)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        msg = _friendly_error(exc) if isinstance(exc, ValidationError) else exc
        print(f"error: {msg}", file=sys.stderr)
        return 1
    base_seed = base_config.seed
    # Checked before anything is written: refusing a folder the tool does
    # not own has to happen while it is still untouched.
    keep = {f"rep_{i:02d}" for i in range(args.replicates)}
    if out_dir.is_dir():
        foreign = _foreign_entries(out_dir, keep)
        if foreign:
            print(
                f"error: {out_dir}/ holds files this tool did not write "
                f"({', '.join(foreign)}), so it will not be cleaned.\n"
                f"       --replicates clears the output folder of stale "
                f"results before it writes.\n"
                f"       Point -o at a new or empty folder, or remove those "
                f"entries first.",
                file=sys.stderr,
            )
            return 1
    # Write FIRST, clean after. Clearing up front meant a run that failed
    # part-way had already deleted the previous results — three good
    # replicates plus a failing run asking for one left rep_00 half
    # written and rep_01, rep_02 gone, with nothing to show for it.
    for i in range(args.replicates):
        rep_dir = out_dir / f"rep_{i:02d}"
        # A reused rep_NN may hold an earlier run's plot. write_output
        # rewrites the data and clears folds/, but a plot.* left from a
        # run made with different --plot flags or a different seed would
        # otherwise ship beside data it does not describe.
        if rep_dir.is_dir():
            for stale in rep_dir.glob("plot.*"):
                stale.unlink()
        code = _run_once(
            args.scenario, rep_dir, args.plot, args.plot_format,
            config=base_config.model_copy(update={"seed": base_seed + i}),
            tick_every=getattr(args, "plot_ticks", None),
        )
        if code != 0:
            return code

    # Every replicate succeeded, so anything still here belongs to an
    # earlier run — more rep_NN/ dirs than this one wrote, or a single
    # run's CSVs and folds/ from before --replicates was used. Either
    # would be read as part of this run.
    if out_dir.is_dir():
        for stale in out_dir.iterdir():
            if stale.name in keep:
                continue
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
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
    # resolve() first: `dsl run metadata.json` from inside the folder
    # gives a bare filename, whose .parent.name is "" — which would
    # name the output directory out/ rather than out/<run>.
    name = (
        path.resolve().parent.name
        if path.name == "metadata.json"
        else path.stem
    )
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
