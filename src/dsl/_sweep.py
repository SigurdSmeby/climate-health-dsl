"""Expand a scenario dict holding ``{sweep: ...}`` values into one dict per
combination.

A sweep lives where the parameter lives::

    series:
      - name: rainfall
        params:
          spike_height: {sweep: [10, 20, 30]}

This runs on the loaded YAML dict, BEFORE the schema sees it, for two
reasons. A swept value is not a legal value of its own field (``lag:
{sweep: [1, 2]}`` is not an int), and ``params:`` is an untyped ``dict`` in
the schema — so an unexpanded sweep there would sail past validation and
fail much later, inside a generator, with a message about a dict where a
number was wanted.

A ``sweep:`` in the YAML is the trigger; there is no flag.
"""
import copy
import itertools
from typing import Any

SWEEP_KEY = "sweep"

_RANGE_KEYS = {"from", "to", "step"}


def _values(spec: Any, where: str) -> list:
    """The values one sweep spec stands for.

    Args:
        spec: What followed ``sweep:`` — a list, or a from/to/step mapping.
        where: The field name holding the sweep, for error messages.

    Returns:
        The list of values to run, in order.

    Errors Caught (raised to caller):
        ValueError: If the spec is neither a list nor a usable range.
    """
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"sweep on '{where}' needs at least one value, got [].")
        return spec
    if not isinstance(spec, dict):
        # ValueError, not TypeError: bad scenario content, which the CLI
        # catches to print a clean error instead of a traceback.
        raise ValueError(  # noqa: TRY004
            f"sweep on '{where}' must be a list of values or a "
            f"{{from, to, step}} range, got {type(spec).__name__}."
        )

    unknown = sorted(set(spec) - _RANGE_KEYS)
    if unknown:
        raise ValueError(
            f"sweep on '{where}' has unknown key(s) {', '.join(unknown)} — "
            f"a range takes 'from', 'to' and 'step'."
        )
    for required in ("from", "to"):
        if required not in spec:
            raise ValueError(
                f"sweep range on '{where}' needs '{required}'."
            )
    start, stop, step = spec["from"], spec["to"], spec.get("step", 1)
    if step == 0:
        raise ValueError(f"sweep range on '{where}' needs a non-zero step.")
    if (stop - start) * step < 0:
        raise ValueError(
            f"sweep range on '{where}' never reaches 'to': from {start} to "
            f"{stop} by {step}."
        )

    # Inclusive of 'to' when it lands exactly on a step. Built by
    # multiplying rather than accumulating, so a float step does not drift.
    count = int((stop - start) / step) + 1
    return [start + i * step for i in range(count)]


def _slug(value: Any) -> str:
    """A value as a filesystem-plain directory-name fragment.

    ``.`` and ``-`` make a directory name ambiguous to split on, so they
    become ``p`` and a leading ``m``. ``sweep.json`` carries the real
    values; this only has to be readable.

    ``.`` and ``-`` are the two characters that make a directory name
    ambiguous to split on, so they become ``p`` and a leading ``m``.
    ``sweep.json`` carries the real values; this only has to be readable.
    """
    text = str(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "".join(c if c.isalnum() else "_" for c in text)
    return text.replace("-", "m").replace(".", "p")


def _owner(node: dict) -> str:
    """What the dict holding a sweep calls itself, or "".

    A ``series:`` entry is identified by its ``name``, a ``depends_on``
    entry by the ``series`` it points at. That is the only fragment that
    tells two sweeps on the same field name apart.
    """
    for key in ("name", "series"):
        value = node.get(key)
        if isinstance(value, str):
            return value
    return ""


def _find(node: Any, path: tuple) -> list[tuple[tuple, str, list, str]]:
    """Every sweep in ``node``, depth-first in document order.

    Args:
        node: A dict, list, or scalar from the loaded scenario.
        path: The keys/indices walked to reach ``node``.

    Returns:
        A list of (path, field_name, values, owner) — path being where to
        write the value back, field_name the key the sweep sat under, owner
        what the enclosing dict calls itself (for disambiguation).
    """
    found: list[tuple[tuple, str, list, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            # A one-key {sweep: ...} dict is a sweep. The same key beside
            # others is a generator param that happens to be called sweep.
            if isinstance(value, dict) and set(value) == {SWEEP_KEY}:
                found.append((
                    (*path, key),
                    str(key),
                    _values(value[SWEEP_KEY], str(key)),
                    _owner(node),
                ))
            else:
                found += _find(value, (*path, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += _find(value, (*path, index))
    return found


def _label_parts(sweeps: list[tuple[tuple, str, list, str]]) -> list[str]:
    """The name fragment for each sweep: its field, qualified only if needed.

    Two sweeps on the same field name (``lag`` under two different parents)
    would give one directory two meanings, so those — and only those — are
    prefixed with what their enclosing entry calls itself.
    """
    names = [name for _, name, _, _ in sweeps]
    return [
        f"{owner}_{name}" if names.count(name) > 1 and owner else name
        for _, name, _, owner in sweeps
    ]


def has_sweep(scenario: dict) -> bool:
    """Does ``scenario`` hold at least one sweep anywhere?"""
    return bool(_find(scenario, ()))


def expand_sweeps(scenario: dict) -> list[tuple[str, dict]]:
    """One (label, scenario) pair per combination of swept values.

    Args:
        scenario: The loaded scenario dict. Not mutated.

    Returns:
        ``[("", scenario)]`` when nothing is swept — a plain run, so the
        caller needs no special case. Otherwise one pair per combination in
        document order, labelled ``spike_height_10__lag_1``.

    Errors Caught (raised to caller):
        ValueError: If a sweep spec is malformed (see ``_values``).
    """
    sweeps = _find(scenario, ())
    if not sweeps:
        return [("", scenario)]

    parts = _label_parts(sweeps)
    runs = []
    for combination in itertools.product(*(values for _, _, values, _ in sweeps)):
        variant = _replace(scenario, [
            (path, value) for (path, _, _, _), value in zip(sweeps, combination)
        ])
        label = "__".join(
            f"{part}_{_slug(value)}" for part, value in zip(parts, combination)
        )
        runs.append((label, variant))
    return runs


def _replace(scenario: dict, assignments: list[tuple[tuple, Any]]) -> dict:
    """A deep copy of ``scenario`` with each path set to its value."""
    variant = copy.deepcopy(scenario)
    for path, value in assignments:
        target = variant
        for segment in path[:-1]:
            target = target[segment]
        target[path[-1]] = value
    return variant
