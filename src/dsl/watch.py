"""``dsl run --watch``: re-run a scenario on every save, with live reload.

With an HTML plot, the output folder is served on localhost and the open
browser page reloads on each successful re-run; otherwise the files are
just regenerated in place.
"""
import functools
import http.server
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable

# Injected into plot.html: poll a version the server bumps on each re-run and
# reload only when it changes (no flicker / zoom reset between edits).
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


def _changed(path: str, last_mtime: float) -> tuple[bool, float]:
    """Has ``path``'s mtime advanced past ``last_mtime``? Returns (changed, mtime).

    A missing file (mid-save by some editors) counts as unchanged.
    """
    try:
        mtime = Path(path).stat().st_mtime
    except FileNotFoundError:
        return False, last_mtime
    return mtime > last_mtime, mtime


def _inject_reload(html_path: Path) -> None:
    """Append the live-reload script to a written plot.html (idempotent)."""
    html = html_path.read_text()
    if "__plot_version__" not in html:
        html = html.replace("</body>", _RELOAD_SCRIPT + "</body>", 1)
        html_path.write_text(html)


def _serve(out_dir: Path, version: list[int]) -> "http.server.HTTPServer":
    """Serve ``out_dir`` on a free localhost port; expose the reload version.

    ``version`` is a one-element list shared with the watch loop — the
    handler reads ``version[0]`` so a re-run can bump it without restarting
    the server.
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


def watch_loop(
    scenario: str,
    out_dir: Path,
    plot: bool,
    plot_format: str,
    run_once: Callable[[str, Path, bool, str], int],
) -> int:
    """Re-run ``scenario`` (via ``run_once``) whenever it is saved, until Ctrl-C."""
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
            if changed and run_once(scenario, out_dir, plot, plot_format) == 0:
                if serve:
                    _inject_reload(out_dir / "plot.html")
                    version[0] += 1  # tells the open page to reload
    except KeyboardInterrupt:
        print("\nstopped watching.")
        return 0
    finally:
        if server is not None:
            server.shutdown()
