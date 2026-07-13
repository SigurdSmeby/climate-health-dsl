"""Step 1 of the pipeline: read the scenario YAML file into a plain dict.

Pure parsing and I/O — interpreting the fields is the schema's job, so a
file-reading problem and an invalid-scenario problem give different errors.
"""
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file and return its top-level mapping as a dict.

    Raises FileNotFoundError for a missing file, ValueError for invalid
    YAML or a top level that isn't a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    try:
        # safe_load refuses YAML's arbitrary-Python-object feature.
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Scenario file {path} must contain a YAML mapping (key: value pairs), "
            f"got {type(data).__name__}"
        )
    return data
