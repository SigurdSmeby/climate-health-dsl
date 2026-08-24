"""Write a ground-truth metadata sidecar next to a generated dataset.

The ground truth (lags, weights, rates, seed) lives in the scenario YAML,
which can drift away from an output folder. ``metadata.json`` makes every
dataset self-describing: the resolved scenario, which is enough to
regenerate the data bit-for-bit.
"""
import json
from pathlib import Path

from dsl import __version__
from dsl.core.config.schema import ScenarioConfig

METADATA_FILENAME = "metadata.json"


def build_metadata(config: ScenarioConfig) -> dict:
    """Build a JSON-serializable metadata record from a validated scenario.

    The ``scenario`` key holds the complete resolved config — feeding it
    back to parse_config() reproduces the run exactly. Every scenario field
    is readable straight from there, so nothing is copied out alongside it.

    Args:
        config: A validated ScenarioConfig object.

    Returns:
        A dict with keys: dsl_version, scenario.
        Example: {'dsl_version': '0.1.0', 'scenario': {'seed': 42, ...}}
    """
    # model_dump fills in defaults, so the sidecar shows the RESOLVED
    # scenario, not just what the user typed.
    scenario = config.model_dump(mode="json")

    # location_overrides is excluded from dumps; rebuild the mapping form so
    # per-location populations survive the round-trip.
    if config.location_overrides:
        scenario["locations"] = {
            name: config.location_overrides[name].model_dump(mode="json")
            if name in config.location_overrides
            else {}
            for name in config.locations
        }

    return {"dsl_version": __version__, "scenario": scenario}


def write_metadata(config: ScenarioConfig, out_dir: str | Path) -> None:
    """Write metadata.json (human-readable, indented) into out_dir.

    Args:
        config: A validated ScenarioConfig object.
        out_dir: Output directory path (created if it doesn't exist).

    Errors Caught (raised to caller):
        OSError: If the output directory cannot be created or written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(config)
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
