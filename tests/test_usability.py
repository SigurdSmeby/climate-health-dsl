"""Usability polish: friendly unknown-field errors, teaching starter, dsl list.

TDD: written before the CLI changes.
"""
import yaml

from dsl.cli import main


def write_scenario(tmp_path, data):
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# --- 1. Friendly "did you mean" for a misspelled field (extra_forbidden) ---

def test_misspelled_field_suggests_correction(tmp_path, capsys):
    data = {
        "period": "weekly", "n_total": 52,
        "locations": {"loc": {"population": 100_000}}, "series": [{"name": "rain", "generate": "seasonal_spike"}, {"name": "disease_cases", "counts": {"maxx_rate": 0.3}, # typo of max_rate
            "depends_on": [{"series": "rain", "weight": 1}]}],
    }
    path = write_scenario(tmp_path, data)
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "maxx_rate" in err            # names the offending key
    assert "max_rate" in err           # suggests the near-miss
    assert "did you mean" in err.lower()
    # The raw pydantic noise should be gone.
    assert "extra_forbidden" not in err
    assert "errors.pydantic.dev" not in err


def test_unknown_field_with_no_close_match_still_clear(tmp_path, capsys):
    data = {
        "period": "weekly", "n_total": 52,
        "locations": {"loc": {"population": 100_000}}, "series": [{"name": "rain", "generate": "seasonal_spike"}, {"name": "disease_cases", "counts": {}, "zzzzz": 5, # not close to any real field
            "depends_on": [{"series": "rain", "weight": 1}]}],
    }
    path = write_scenario(tmp_path, data)
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "zzzzz" in err                # still names it
    assert "unknown field" in err.lower()


def test_misspelled_location_override_field_suggests_from_location_spec(
    tmp_path, capsys
):
    # Regression: the suggestion pool must come from LocationSpec's own
    # fields (just 'population'), not get derailed by the location's NAME
    # ('north') looking like a plausible parent-model key too.
    data = {
        "period": "weekly", "n_total": 52,
        "locations": {"north": {"populaton": 5}},  # typo, inside an override
        "series": [{"name": "rain", "generate": "seasonal_spike"}, {"name": "disease_cases", "counts": {}, "depends_on": [{"series": "rain", "weight": 1}]}],
    }
    path = write_scenario(tmp_path, data)
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "populaton" in err
    assert "did you mean 'population'?" in err


# --- 2. Starter template teaches the newer features ---

def test_starter_mentions_transforms_and_dependencies(tmp_path, capsys):
    path = tmp_path / "s.yaml"
    assert main(["new", str(path)]) == 0
    text = path.read_text()
    assert "transforms" in text
    assert "depends_on" in text


def test_new_message_does_not_assume_uv(tmp_path, capsys):
    # A no-uv user must not be told to run `uv run …`; the hint should be the
    # bare `dsl run` (with uv mentioned only as an option).
    main(["new", str(tmp_path / "s.yaml")])
    out = capsys.readouterr().out
    assert "uv run dsl run" not in out
    assert "dsl run" in out


def test_starter_still_runs_clean(tmp_path, capsys):
    # The teaching comments must not break the scaffold: it still parses/runs
    # with no warnings (the commented blocks are inert).
    path = tmp_path / "s.yaml"
    main(["new", str(path)])
    code = main(["run", str(path), "-o", str(tmp_path / "out")])
    assert code == 0
    err = capsys.readouterr().err
    assert "warning:" not in err


# --- 3. `dsl list` shows registered generators and transforms ---

def test_dsl_list_shows_generators_and_transforms(capsys):
    code = main(["list"])
    assert code == 0
    out = capsys.readouterr().out
    # A sample of each kind must appear.
    assert "seasonal_spike" in out
    assert "lag" in out
    assert "missing" in out


def test_cross_field_errors_drop_pydantic_boilerplate(tmp_path, capsys):
    """A validator's own message is already written for a scenario author;
    Pydantic's "Value error, " prefix in front of it is noise."""
    from tests.conftest import scenario_dict

    data = scenario_dict(locations={"loc": {}})
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data))
    assert main(["run", str(path), "-o", str(tmp_path / "out")]) != 0
    err = capsys.readouterr().err
    assert "Value error" not in err
    assert "need a population" in err


def _option_lines() -> list[int]:
    """Indices of the starter's switchable option lines.

    An option is a comment line that is valid YAML once the "#" is gone:
    a key or a list item, at the indent its parent expects. Prose comments
    are sentences and fail that test, so the two are told apart by what
    the line would become rather than by guessing at its text.

    Returns:
        Line indices into STARTER_TEMPLATE.splitlines(), in file order.
    """
    import re

    from dsl._output import STARTER_TEMPLATE

    # A YAML key or list item, lowercase — prose starts with a capital and
    # reads as a sentence. A bare "#" is a paragraph break, not a switch.
    body = re.compile(r"^# *(?:[a-z_][\w.]*:|- |\{)")
    return [
        i
        for i, line in enumerate(STARTER_TEMPLATE.splitlines())
        if body.match(line)
    ]


def _uncomment_starter(only: "list[int] | None" = None) -> str:
    """Enable the starter's options, as a reader would.

    Deletes the single "#" and nothing else — exactly what the tutorial
    tells the reader to do.

    Args:
        only: Enable just these line indices, leaving the rest commented.
            None (the default) enables all of them.

    Returns:
        The starter text with its option lines enabled.
    """
    from dsl._output import STARTER_TEMPLATE

    wanted = _option_lines() if only is None else list(only)
    lines = [
        line[1:] if i in wanted else line
        for i, line in enumerate(STARTER_TEMPLATE.splitlines())
    ]
    return "\n".join(lines)


def test_every_commented_option_in_the_starter_works_uncommented():
    """The starter invites the reader to drop a `#`. Every option it offers
    must then still parse and run — a comment that breaks when enabled is
    worse than no comment."""
    import yaml

    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run

    config = parse_config(yaml.safe_load(_uncomment_starter()))
    df = run(config)
    # The options add a second location and a dependent series.
    assert df["location"].nunique() == 2
    assert "soil_moisture" in df.columns


def test_the_starter_marks_its_options_unambiguously():
    """Every option is a "#" against the body; every "# " line is prose.

    The convention is what lets a reader delete one character anywhere in
    the file, so a line that breaks it would make the rule a half-truth.
    """
    import re

    from dsl._output import STARTER_TEMPLATE

    # Prose wraps to several lines and reads as a sentence; an option is a
    # YAML key or list item. A prose line that happens to start with a
    # lowercase key would be enabled by the rule and break the file.
    prose_as_option = [
        line
        for i, line in enumerate(STARTER_TEMPLATE.splitlines())
        if i in _option_lines() and not re.match(r"^# *([a-z_]+:|- |\{)", line)
    ]
    assert not prose_as_option, (
        "these are marked as options but are not YAML:\n"
        + "\n".join(prose_as_option)
    )
    assert _option_lines(), "the starter offers no options at all"


# Options that only make sense together: a series can react to `storm`
# only once `storm` itself is declared, so the starter's own comment tells
# the reader to enable all three. Each other option stands alone.
_GROUPED = ("events:",)


def test_each_standalone_option_works_by_itself():
    """Enabling one option must leave a scenario that actually runs.

    The check goes through the schema, not just the YAML parser: an option
    that parses but fails validation would hand the reader an error the
    tutorial never warned them about. Options the starter groups are
    enabled together, as its comment instructs.
    """
    import yaml
    from pydantic import ValidationError

    from dsl._output import STARTER_TEMPLATE
    from dsl.core.config.schema import parse_config

    lines = STARTER_TEMPLATE.splitlines()
    grouped = [i for i in _option_lines() if any(g in lines[i] for g in _GROUPED)]
    broken = []
    for i in _option_lines():
        # A grouped option brings its whole group; a standalone one is alone.
        enable = grouped if i in grouped else [i]
        try:
            parse_config(yaml.safe_load(_uncomment_starter(only=enable)))
        except (ValidationError, ValueError, yaml.YAMLError) as exc:
            broken.append(f"{lines[i]}\n    -> {type(exc).__name__}: {exc}")
    assert not broken, (
        "deleting the '#' left these options unusable:\n" + "\n".join(broken)
    )


def test_grouped_options_are_the_ones_the_starter_says_to_group():
    """The file must warn about every option that cannot stand alone.

    Otherwise "delete the #" is a half-truth exactly where a reader would
    be caught by it.
    """
    import yaml

    from dsl._output import STARTER_TEMPLATE
    from dsl.core.config.schema import parse_config

    lines = STARTER_TEMPLATE.splitlines()
    for i in _option_lines():
        if any(g in lines[i] for g in _GROUPED):
            continue
        # Every ungrouped option must genuinely work on its own.
        parse_config(yaml.safe_load(_uncomment_starter(only=[i])))

    # And the grouped ones must be announced in the prose above them.
    prose = "\n".join(lines)
    assert "Uncomment this AND" in prose, (
        "an option needs its group, but the starter does not say so"
    )


def test_the_starter_threshold_actually_bites():
    """An option that changes nothing teaches the reader the wrong lesson.

    `hinge` below the driver's minimum is a constant subtraction, and
    standardizing removes a constant exactly — so the transform would run,
    look enabled, and leave the output untouched.
    """
    import numpy as np
    import yaml

    from dsl._output import STARTER_TEMPLATE
    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run

    # Compare the two configs the reader actually has: every option on,
    # versus every option on except the transforms line. Identifying that
    # line by index rather than by its text keeps the comparison honest if
    # the line is reflowed or the word appears in a comment.
    lines = STARTER_TEMPLATE.splitlines()
    transform_line = [
        i for i in _option_lines() if lines[i].lstrip("# ").startswith("transforms:")
    ]
    assert len(transform_line) == 1, "expected exactly one transforms option"
    others = [i for i in _option_lines() if i not in transform_line]

    enabled = parse_config(yaml.safe_load(_uncomment_starter()))
    without = parse_config(yaml.safe_load(_uncomment_starter(only=others)))
    assert without.series != enabled.series, "the two configs are the same"

    a = run(without)["disease_cases"].to_numpy()
    b = run(enabled)["disease_cases"].to_numpy()
    assert not np.allclose(a, b, equal_nan=True), (
        "the starter's threshold sits outside the rainfall range, so "
        "enabling it changes nothing"
    )


def test_the_tutorial_shows_the_starter_s_own_transforms_line():
    """A reader compares the two side by side, so they must match.

    Showing a block-style list against the starter's flow-style line
    invites retyping it at the wrong indent.
    """
    import pathlib

    from dsl._output import STARTER_TEMPLATE

    lines = STARTER_TEMPLATE.splitlines()
    starter = next(
        lines[i].lstrip("# ")
        for i in _option_lines()
        if lines[i].lstrip("# ").startswith("transforms:")
    )
    tutorial = pathlib.Path("docs/TUTORIAL.md").read_text()
    assert starter in tutorial, (
        f"the tutorial does not show the starter's line:\n  {starter}"
    )
