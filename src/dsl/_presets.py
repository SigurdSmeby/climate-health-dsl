"""Resolve ``preset: <name>`` into the scenario it stands for.

A preset is an ordinary scenario file bundled in ``presets/``. A scenario
naming one is merged over it, so the file holds only what differs:

    preset: climate
    n_total: 120

The merge is the whole design. A plain dict merge would replace the
``series:`` list wholesale, so a scenario mentioning one series to change
one parameter would silently delete the others — a scenario that still
looks right and no longer means what it did. Lists of named entries are
therefore merged per entry, matched on the key that identifies them.
"""
import copy
import difflib
from pathlib import Path
from typing import Any

from dsl.core.config.loader import load_yaml

PRESET_KEY = "preset"

PRESETS_DIR = Path(__file__).parent / "presets"

# What identifies an entry in a list of mappings, in priority order:
# `series:` entries carry a name, `depends_on:` entries point at a series.
# A list whose entries have none of these cannot be merged per entry.
_IDENTITY_KEYS = ("name", "series")


def available() -> list[str]:
    """The bundled preset names, sorted."""
    return sorted(path.stem for path in PRESETS_DIR.glob("*.yaml"))


def _identity(entry: Any) -> str | None:
    """What ``entry`` is called, if it is an identifiable mapping."""
    if not isinstance(entry, dict):
        return None
    for key in _IDENTITY_KEYS:
        value = entry.get(key)
        if isinstance(value, str):
            return f"{key}={value}"
    return None


def _merge_lists(base: list, over: list) -> list:
    """Merge two lists entry by entry when their entries are identifiable.

    Entries present in both are merged; entries only the preset has are
    kept in its order; entries only the scenario has are appended.

    A list whose entries cannot be identified (plain numbers, mappings with
    no name) is replaced by ``over`` — there is nothing to match on, and
    guessing by position would make a scenario's meaning depend on how the
    preset happened to order its entries.
    """
    base_ids = [_identity(entry) for entry in base]
    over_ids = [_identity(entry) for entry in over]
    if None in base_ids or None in over_ids:
        return copy.deepcopy(over)

    merged = []
    by_id = dict(zip(over_ids, over))
    for entry, entry_id in zip(base, base_ids):
        if entry_id in by_id:
            merged.append(merge(entry, by_id[entry_id]))
        else:
            merged.append(copy.deepcopy(entry))
    # Whatever the preset did not have, in the order the scenario wrote it.
    merged += [
        copy.deepcopy(entry)
        for entry, entry_id in zip(over, over_ids)
        if entry_id not in base_ids
    ]
    return merged


def merge(base: dict, over: dict) -> dict:
    """``base`` with ``over`` applied on top. Neither is mutated.

    Mappings merge key by key, lists of identifiable entries merge per
    entry (see ``_merge_lists``), and anything else in ``over`` replaces
    what ``base`` had — including an explicit ``None``, which is how a
    scenario turns a preset's block off.

    Args:
        base: The preset's scenario dict.
        over: The scenario's own dict.

    Returns:
        A new dict; the inputs are untouched.
    """
    merged = copy.deepcopy(base)
    for key, value in over.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            merged[key] = _merge_lists(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_preset(scenario: dict) -> dict:
    """Apply ``scenario``'s preset, if it names one, and drop the key.

    ``preset:`` is not a scenario field — the schema never sees it, so it
    is resolved away here, before validation.

    Args:
        scenario: The loaded scenario dict. Not mutated.

    Returns:
        The scenario merged over its preset, without the ``preset`` key.
        Returned unchanged when it names no preset.

    Errors Caught (raised to caller):
        ValueError: If the name is not a string, names no bundled preset,
            or the preset itself names one.
    """
    if PRESET_KEY not in scenario:
        return scenario

    name = scenario[PRESET_KEY]
    if isinstance(name, bool) or not isinstance(name, str):
        raise ValueError(  # noqa: TRY004
            f"{PRESET_KEY}: must be the name of a bundled preset "
            f"({', '.join(available())}), got {name!r}."
        )

    path = PRESETS_DIR / f"{name}.yaml"
    if not path.is_file():
        near = difflib.get_close_matches(name, available(), n=1)
        hint = f" — did you mean '{near[0]}'?" if near else ""
        raise ValueError(
            f"unknown preset '{name}'{hint}\n"
            f"       available: {', '.join(available()) or '(none)'}"
        )

    base = load_yaml(path)
    if PRESET_KEY in base:
        # One level only: no cycle to detect, no resolution order to
        # explain. Worth revisiting if a real scenario ever needs it.
        raise ValueError(
            f"preset '{name}' cannot use a preset itself (one level only)."
        )

    merged = merge(base, scenario)
    del merged[PRESET_KEY]
    return merged
