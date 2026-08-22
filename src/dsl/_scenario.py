"""Load, parse, and validate scenario files.

This module handles reading scenario YAML/JSON files, resolving relative
paths (especially for from_csv generators), validating against the schema,
and formatting validation errors for CLI output.
"""
import difflib
import typing
from pathlib import Path

from pydantic import BaseModel, ValidationError

from dsl.core.config.loader import load_yaml
from dsl.core.config.schema import ScenarioConfig, parse_config


def _load_scenario_dict(path: str) -> dict:
    """Load a scenario dict from a YAML scenario OR a metadata.json sidecar.

    ``load_yaml`` parses both (JSON is valid YAML). A metadata file wraps the
    real scenario under a ``"scenario"`` key — unwrap it so a dataset can be
    reproduced from its own ``metadata.json``.

    Args:
        path: Path to a scenario YAML file or a metadata.json sidecar.

    Returns:
        A dict ready for parse_config(), with any relative from_csv paths
        resolved against the scenario file's directory.

    Errors Caught (raised to caller):
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not valid YAML/JSON.
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

    Args:
        scenario: The scenario dict, as loaded from YAML (mutated in place).
        base_dir: Directory to resolve relative from_csv `file` params against.
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


def _load_and_parse(scenario: str, seed_override: int | None = None) -> ScenarioConfig:
    """Load + validate a scenario file. Raises on any hard error.

    Shared by ``_run_once`` and ``_run_replicates`` so both report
    parse/validation errors identically.

    Args:
        scenario: Path to the scenario YAML file or metadata.json.
        seed_override: If given, replaces the parsed seed (used for
            replicates), so the written metadata records the actual seed.

    Returns:
        A validated ScenarioConfig.

    Errors Caught (raised to caller):
        FileNotFoundError: If the scenario file doesn't exist.
        ValueError: If seed_override is negative, or the scenario is invalid
            in a way pydantic doesn't catch.
        ValidationError: If pydantic validation fails.
    """
    config = parse_config(_load_scenario_dict(scenario))
    if seed_override is not None:
        # model_copy skips validation and (unlike model_dump()+model_validate())
        # preserves location_overrides, so check the one constraint that
        # matters — ScenarioConfig.seed is `ge=0` — by hand instead.
        if seed_override < 0:
            raise ValueError(f"seed_override must be >= 0, got {seed_override}.")
        config = config.model_copy(update={"seed": seed_override})
    return config


def _friendly_error(exc: ValidationError) -> str:
    """Rewrite a Pydantic ValidationError into a message a scenario author can
    act on. Turns the noisy ``extra_forbidden`` (a typo'd/unknown key) into
    ``unknown field 'x' … did you mean 'y'?``, scoped to the near-miss pool of
    the model the typo is actually in (mirroring the generator-typo message).
    Other error types keep Pydantic's own (already clear) text.

    Args:
        exc: The ValidationError raised by pydantic during parsing.

    Returns:
        A single string with one "; "-joined line per error.
        Example: "unknown field 'periode' — did you mean 'period'?"
    """
    from dsl.core.config import schema as _schema

    def _nested_model_fields(annotation) -> set[str] | None:
        """If ``annotation`` is (possibly via list[]/dict[str, ]/X|None) a
        BaseModel subclass, that model's field names; else None."""
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return set(annotation.model_fields)
        for arg in typing.get_args(annotation):
            found = _nested_model_fields(arg)
            if found is not None:
                return found
        return None

    # field name -> its NESTED model's field pool, e.g. "location_overrides"
    # -> LocationSpec's fields, so a typo inside a location override only
    # gets suggestions from what LocationSpec accepts.
    model_for: dict[str, set[str]] = {}
    all_fields: set[str] = set()
    for obj in vars(_schema).values():
        fields = getattr(obj, "model_fields", None)
        if not isinstance(fields, dict):
            continue
        all_fields.update(fields)
        for field_name, field_info in fields.items():
            nested = _nested_model_fields(field_info.annotation)
            if nested is not None:
                model_for[field_name] = nested

    lines = []
    for err in exc.errors():
        if err["type"] == "extra_forbidden" and err["loc"]:
            key = str(err["loc"][-1])
            where = ".".join(str(p) for p in err["loc"][:-1])
            at = f" in {where}" if where else ""
            # The nearest enclosing model: scan loc backwards for the first
            # segment that's an actual field name we know a model for — NOT
            # just the first string, which could be a location name.
            pool = all_fields
            for segment in reversed(err["loc"][:-1]):
                if isinstance(segment, str) and segment in model_for:
                    pool = model_for[segment]
                    break
            near = difflib.get_close_matches(key, pool, n=1)
            hint = f" — did you mean '{near[0]}'?" if near else ""
            lines.append(f"unknown field '{key}'{at}{hint}")
        else:
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(lines)
