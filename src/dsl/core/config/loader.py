"""Step 1 of the pipeline: read the scenario YAML file into a plain dict.

This module does pure parsing and I/O. It does not interpret what any field
means — that is the schema's job (see ``schema.py``). Keeping the two apart
means a file-reading problem and an invalid-scenario problem produce clearly
different errors.
"""
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file from disk and return its contents as a dict.

    Parameters
    ----------
    path:
        Path to the scenario file.

    Returns
    -------
    dict
        The parsed YAML mapping, ready to be validated by ``parse_config``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid YAML, or its top level is not a mapping
        (e.g. the file contains just a list or a bare string).
    """
    path = Path(path)  # accept both strings and Path objects
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    try:
        # safe_load parses plain data only — it refuses YAML's "build arbitrary
        # Python objects" feature, which untrusted input could abuse.
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        # Re-raise with the filename attached; `from exc` keeps the original
        # YAML error (with line/column info) chained underneath.
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Scenario file {path} must contain a YAML mapping (key: value pairs), "
            f"got {type(data).__name__}"
        )
    return data
