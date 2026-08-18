"""Write a ground-truth metadata sidecar next to a generated dataset.

The whole value of the tool is *known* ground truth — the lags, weights,
rates and seed that the data was built from. But that truth lives in the
scenario YAML, which can drift away from an output folder (edit the YAML,
forget which ``out/`` it produced). This module writes a ``metadata.json``
beside the CSVs so every dataset is self-describing: it records the resolved
scenario (enough to regenerate the data bit-for-bit), the generators used,
and the tool version.
"""
import json
from pathlib import Path

from pydantic import BaseModel

from dsl import __version__
from dsl.core.config.schema import ScenarioConfig

METADATA_FILENAME = "metadata.json"


def _population_json(population: "int | BaseModel") -> object:
    """A JSON-safe view of a population: an int as-is, a spec as a dict."""
    if isinstance(population, BaseModel):
        return population.model_dump(mode="json")
    return population


def build_metadata(config: ScenarioConfig) -> dict:
    """Return a JSON-serializable record of the ground truth behind a run.

    Includes a ``scenario`` key holding the full resolved config — feeding
    it back to ``parse_config`` reproduces the run exactly — plus flattened
    top-level fields (seed, period, the disease dependencies) for quick
    inspection without digging into the nested scenario.
    """
    # model_dump gives a plain dict of the validated config, with defaults
    # filled in (so the sidecar shows the *resolved* scenario, not just what
    # the user typed). mode="json" makes every value JSON-safe.
    scenario = config.model_dump(mode="json")
    # location_overrides is excluded from the dump (it's internal), so when a
    # scenario used the mapping form we rebuild it here — otherwise the
    # per-location populations would be lost and the round-trip wouldn't
    # reproduce the dataset.
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
            # population may be an int or a PopulationSpec (a generator);
            # dump the model form so the whole metadata dict is JSON-safe.
            "population": _population_json(config.disease_cases.population),
            "autoregressive": config.disease_cases.autoregressive,
            "missing_rate": config.disease_cases.missing_rate,
            "max_rate": config.disease_cases.max_rate,
            "median_rate": config.disease_cases.median_rate,
            "count_distribution": config.disease_cases.count_distribution,
            "overdispersion": config.disease_cases.overdispersion,
            "depends_on": [
                {"variable": d.variable, "lag": d.lag, "weight": d.weight}
                for d in config.disease_cases.depends_on
            ],
        },
        # The full resolved scenario, sufficient to regenerate the dataset.
        "scenario": scenario,
    }


def write_metadata(config: ScenarioConfig, out_dir: str | Path) -> None:
    """Write the metadata sidecar as ``metadata.json`` in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(config)
    # indent=2 keeps the file human-readable; it's meant to be opened.
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
