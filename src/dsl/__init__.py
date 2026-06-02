"""dsl — a YAML-based DSL for generating synthetic climate-health datasets.

The package is split into two zones:

- ``dsl.core``: the locked machinery (registry, base classes, config,
  pipeline). Written once, then left alone.
- ``dsl.generators`` and ``dsl.transforms``: the extension zones. Each file
  there is one self-contained feature that registers itself on import.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    # Read the version from the installed package metadata, so pyproject.toml
    # stays the single source of truth (no second place to bump).
    __version__ = version("dsl")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0"
