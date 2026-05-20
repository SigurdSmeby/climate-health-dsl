"""dsl — a YAML-based DSL for generating synthetic climate-health datasets.

The package is split into two zones:

- ``dsl.core``: the locked machinery (registry, base classes, config,
  pipeline). Written once, then left alone.
- ``dsl.generators`` and ``dsl.transforms``: the extension zones. Each file
  there is one self-contained feature that registers itself on import.
"""
