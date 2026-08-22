"""Execute a scenario: run once, run replicates, or watch for changes.

This module owns everything that happens after the output directory is
decided: parse-generate-write one scenario, run it N times into rep_NN/
dirs with varying seeds, or re-run it automatically on every save and
(for HTML plots) live-reload a browser tab.
"""
import argparse
import functools
import http.server
import math
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
from dsl.core.pipeline.chap_check import validate_chap
from dsl.core.pipeline.engine import run as run_engine
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

    # Generate, check CHAP compatibility, write. Generation can fail on input
    # the schema can't catch (a bad generator param, a malformed from_csv
    # source) — surface those as clean CLI errors, not a raw traceback.
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
    # A reused folder may hold MORE rep_NN/ dirs than this run writes (e.g. a
    # previous --replicates 5, now --replicates 2) — clear the ones this run
    # won't overwrite so no stale replicate is left mixed in with fresh ones.
    if out_dir.is_dir():
        for stale_rep in out_dir.iterdir():
            n = _numbered_dir_suffix(stale_rep, "rep_")
            if n is not None and n >= args.replicates:
                shutil.rmtree(stale_rep)
    for i in range(args.replicates):
        code = _run_once(
            args.scenario, out_dir / f"rep_{i:02d}", args.plot, args.plot_format,
            config=base_config.model_copy(update={"seed": base_seed + i}),
        )
        if code != 0:
            return code
    return 0


def _changed(path: str, last_mtime: float) -> tuple[bool, float]:
    """Has ``path``'s mtime advanced past ``last_mtime``?

    A missing file (mid-save by some editors) counts as unchanged.

    Args:
        path: File to check.
        last_mtime: The mtime last observed.

    Returns:
        (changed, mtime): whether it changed, and the current mtime to
        remember for the next check (unchanged from last_mtime if missing).
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
    """Append the live-reload script to a written plot.html (idempotent).

    Args:
        html_path: Path to the plot.html file to patch in place.
    """
    html = html_path.read_text()
    if "__plot_version__" not in html:
        html = html.replace("</body>", _RELOAD_SCRIPT + "</body>", 1)
        html_path.write_text(html)


def _serve(out_dir: Path, version: list[int]) -> "http.server.HTTPServer":
    """Serve ``out_dir`` on a free localhost port; expose the reload version.

    ``version`` is a one-element list shared with the watch loop — the handler
    reads ``version[0]`` so a re-run can bump it without restarting the server.

    Args:
        out_dir: Directory to serve as static files (contains plot.html).
        version: One-element list; version[0] is polled by the injected
            reload script and bumped by the watch loop after each re-run.

    Returns:
        The running HTTPServer (already serving in a daemon thread).
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
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _watch_loop(
    scenario: str, out_dir: Path, plot: bool, plot_format: str
) -> int:
    """Re-run ``scenario`` whenever its file is saved, until interrupted.

    With an HTML plot, also serve it on localhost and live-reload the browser
    on each successful re-run; otherwise just regenerate the files in place.

    Args:
        scenario: Path to the scenario YAML file being watched.
        out_dir: Directory each re-run writes into (overwritten in place).
        plot: Whether to also write (and, if html, serve) a plot.
        plot_format: Plot file format (e.g. "html", "png").

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
                and _run_once(scenario, out_dir, plot, plot_format) == 0
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
    if choice.isdigit() and 1 <= int(choice) <= len(runs):
        chosen = runs[int(choice) - 1]
        print(f"continuing in {chosen}/ (overwriting it).")
        return chosen
    return None  # "n", empty, or anything else → a new run


def _scenario_runs(scenario: str) -> list[Path]:
    """Existing out/ folders belonging to ``scenario`` (its base + _N siblings).

    Args:
        scenario: Path to the scenario file (only its stem is used).

    Returns:
        Existing run directories, sorted: base ``out/<name>`` first, then
        ``_1``, ``_2``, … numerically. Empty list if out/ doesn't exist.
    """
    name = Path(scenario).stem
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
