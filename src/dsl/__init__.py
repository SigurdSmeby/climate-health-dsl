"""dsl — a YAML-based DSL for generating synthetic climate-health datasets.

Two zones: ``dsl.core`` is the locked machinery (registry, config, pipeline);
``dsl.generators`` and ``dsl.transforms`` are the extension zones, where each
file is one self-contained feature that registers itself on import.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    # pyproject.toml stays the single source of truth for the version.
    __version__ = version("dsl")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0"
