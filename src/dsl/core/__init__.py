"""dsl.core — the locked machinery: extension system, config, and pipeline.

Nothing in here should need editing once built, with one exception:
``core/config/schema.py`` may grow when a genuinely new top-level concept
(a new global field, a new disease-model option) is added.
"""
