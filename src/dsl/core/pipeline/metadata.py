"""Write a ground-truth metadata sidecar next to a generated dataset.

The ground truth (lags, weights, rates, seed) lives in the scenario YAML,
which can drift away from an output folder. ``metadata.json`` makes every
dataset self-describing: the resolved scenario (enough to regenerate the
data bit-for-bit), plus flattened fields for quick inspection.
"""
import json
from pathlib import Path

from dsl import __version__
from pydantic import BaseModel

from dsl.core.config.schema import ScenarioConfig

METADATA_FILENAME = "metadata.json"


def _population_json(population: "int | BaseModel") -> object:
    """A JSON-safe view of a population: an int as-is, a spec as a dict."""
    if isinstance(population, BaseModel):
        return population.model_dump(mode="json")
    return population


def build_metadata(config: ScenarioConfig) -> dict:
    """Return a JSON-serializable record of the ground truth behind a run.

    The ``scenario`` key holds the full resolved config — feeding it back to
    ``parse_config`` reproduces the run exactly. The other keys flatten the
    interesting fields for quick inspection.
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
    return {
        "dsl_version": __version__,
        "seed": config.seed,
        "period": config.period,
        "n_total": config.n_total,
        "start_period": config.start_period,
        "locations": list(config.locations),
        "variables": [
            {"name": v.name, "generate": v.generate, "params": v.params}
            for v in config.variables
        ],
        "disease_cases": {
            "population": _population_json(config.disease_cases.population),
            "autoregressive": config.disease_cases.autoregressive,
            "missing_rate": config.disease_cases.missing_rate,
            "max_rate": config.disease_cases.max_rate,
            "median_rate": config.disease_cases.median_rate,
            "count_distribution": config.disease_cases.count_distribution,
            "overdispersion": config.disease_cases.overdispersion,
            "depends_on": [
                {
                    "variable": d.variable, "lag": d.lag, "weight": d.weight,
                    "transforms": [
                        {"name": t.name, "params": t.params} for t in d.transforms
                    ],
                }
                for d in config.disease_cases.depends_on
            ],
        },
        "scenario": scenario,
    }


def write_metadata(config: ScenarioConfig, out_dir: str | Path) -> None:
    """Write ``metadata.json`` (human-readable, indented) in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(config)
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
