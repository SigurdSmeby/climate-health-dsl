"""Write a ground-truth metadata sidecar next to a generated dataset.

The ground truth (lags, weights, rates, seed) lives in the scenario YAML,
which can drift away from an output folder. ``metadata.json`` makes every
dataset self-describing: the resolved scenario (enough to regenerate the
data bit-for-bit), plus flattened fields for quick inspection.
"""
import json
from pathlib import Path

from pydantic import BaseModel

from dsl import __version__
from dsl.core.config.schema import ScenarioConfig

METADATA_FILENAME = "metadata.json"


def build_metadata(config: ScenarioConfig) -> dict:
    """Build a JSON-serializable metadata record from a validated scenario.

    The returned dict contains two parts:
    1. Flattened top-level fields (seed, period, locations, variables) for
       quick inspection.
    2. A full "scenario" key with the complete resolved config — feeding it
       back to parse_config() reproduces the run exactly.

    Args:
        config: A validated ScenarioConfig object.

    Returns:
        A dict with keys: dsl_version, seed, period, n_total, start_period,
        locations, variables, disease_cases, scenario.
        Example: {'dsl_version': '0.1.0', 'seed': 42, 'period': 'monthly', ...}
    """
    # Step 1: Resolve defaults. model_dump fills them in, so the sidecar
    # shows the RESOLVED scenario, not just what the user typed.
    scenario = config.model_dump(mode="json")
    # Now scenario = {'period': 'monthly', 'n_total': 36, ..., <defaults filled>}

    # location_overrides is excluded from dumps; rebuild the mapping form so
    # per-location populations survive the round-trip.
    if config.location_overrides:
        scenario["locations"] = {
            name: config.location_overrides[name].model_dump(mode="json")
            if name in config.location_overrides
            else {}
            for name in config.locations
        }

    # Step 2: Flatten the interesting fields for quick inspection, alongside
    # the full scenario for reproduction.
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
    # Now out_dir/metadata.json contains the resolved scenario + flattened fields


def _population_json(population: "int | BaseModel") -> object:
    """A JSON-safe view of a population: an int as-is, a spec as a dict.

    Args:
        population: Either a fixed population (int) or a PopulationSpec.

    Returns:
        The int unchanged, or population.model_dump(mode="json") for a spec.
    """
    if isinstance(population, BaseModel):
        return population.model_dump(mode="json")
    return population
