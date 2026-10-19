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


def _is_option(line: str) -> bool:
    """Is this line one of the starter's switchable options?

    An option is a comment whose "#" sits against the body, so deleting
    that one character leaves valid YAML at the right indent. Prose puts a
    space after the "#" and reads as a sentence.

    Args:
        line: One line of the starter.

    Returns:
        True if deleting the leading "#" is meant to enable something.
    """
    import re

    return bool(re.match(r"^# *(?:[a-z_][\w.]*:|- |\{)", line))


def _option_lines() -> list[int]:
    """Indices of the starter's switchable option lines.

    An option is a comment line that is valid YAML once the "#" is gone:
    a key or a list item, at the indent its parent expects. Prose comments
    are sentences and fail that test, so the two are told apart by what
    the line would become rather than by guessing at its text.

    Returns:
        Line indices into STARTER_TEMPLATE.splitlines(), in file order.
    """
    from dsl._output import STARTER_TEMPLATE

    return [
        i
        for i, line in enumerate(STARTER_TEMPLATE.splitlines())
        if _is_option(line)
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
        "enabling the threshold changed nothing in the output"
    )

    # The reason it bites, stated as the docs state it: a threshold below
    # the driver's minimum is a constant subtraction, which standardizing
    # removes exactly. Asserting the output differs would not catch a
    # threshold that drifted out of range if something else moved too.
    rainfall = run(enabled)["rainfall"].to_numpy()
    threshold = next(
        t.params["threshold"]
        for spec in enabled.series
        for dep in spec.depends_on
        for t in dep.transforms
        if t.name == "threshold"
    )
    low, high = np.nanmin(rainfall), np.nanmax(rainfall)
    assert low < threshold < high, (
        f"threshold {threshold} is outside the rainfall range "
        f"{low:.2f}..{high:.2f}, so hinge is a constant shift"
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
    repo = pathlib.Path(__file__).resolve().parent.parent
    tutorial = (repo / "docs" / "TUTORIAL.md").read_text()
    # Inside the step that tells the reader to enable it, not merely
    # somewhere in the document.
    step = tutorial.split("## 10.")[1].split("## 11.")[0]
    assert starter in step, (
        f"the transforms step does not show the starter's line:\n  {starter}"
    )


def test_the_file_dsl_new_writes_follows_the_same_rule(tmp_path):
    """The rule is about the file on disk, not the template string.

    Every other test reads STARTER_TEMPLATE, so a change in how `dsl new`
    writes it out would go unnoticed. This enables the options in the
    written file and runs it, the way a reader does.
    """
    import yaml

    from dsl._output import _write_starter
    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run

    scenario = tmp_path / "s.yaml"
    _write_starter(scenario, force=False)
    written = scenario.read_text().splitlines()

    enabled = [
        line[1:] if _is_option(line) else line for line in written
    ]
    df = run(parse_config(yaml.safe_load("\n".join(enabled))))
    assert df["location"].nunique() == 2
    assert "soil_moisture" in df.columns


def test_the_starter_season_survives_its_own_storms():
    """A storm should disturb the season, not replace it.

    The starter is read as a picture before it is read as YAML, so the
    yearly rhythm has to stay the dominant feature of the rainfall panel
    with every option switched on. A `multiplier` storm scaled the whole
    value, landing well above a seasonal peak the default `spike_width`
    had already flattened — so the plot showed spikes and no season.
    """
    import numpy as np
    import yaml

    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run
    from dsl.generators.seasonal_spike import SeasonalSpikeGenerator

    config = parse_config(yaml.safe_load(_uncomment_starter()))
    frame = run(config)
    rainfall = frame[frame["location"] == "loc"]["rainfall"].to_numpy()

    # The season has to come back down to the baseline the generator
    # documents. A wide spike never decays inside the year, so the trough
    # stays partway up the peak and the series reads as a raised floor
    # with a bump rather than as a season.
    trough, peak = np.nanmin(rainfall), np.nanmax(rainfall)
    centre = config.series[0].params["spike_center"]
    seasonal = np.nanmean(rainfall[centre - 1 :: 12])
    baseline = SeasonalSpikeGenerator().baseline
    assert trough < baseline + 0.2 * (seasonal - baseline), (
        f"the dry season bottoms out at {trough:.1f}, far above the "
        f"{baseline} baseline, against a {seasonal:.1f} peak — the spike is "
        f"too wide to decay within the year"
    )

    # And the largest storm must stay inside the same visual range as the
    # season rather than rescaling the axis around itself.
    assert (peak - trough) < 2 * (seasonal - trough), (
        f"the biggest storm reaches {peak:.1f}, more than twice the "
        f"seasonal amplitude ({seasonal - trough:.1f}) — it drowns the season"
    )


def test_the_starter_labels_what_it_measures():
    """The field existing solves nothing on its own — a reader who never
    heard of it gets the same unlabelled plot. The starter has to use it,
    so a first run answers the question instead of raising it.

    Equally, the units must not spread to everything: a series built from
    standardized parents has no physical scale, and inventing one for it
    would teach the reader the opposite of what the field means.
    """
    import yaml

    from dsl._output import STARTER_TEMPLATE
    from dsl.core.config.schema import parse_config

    config = parse_config(yaml.safe_load(_uncomment_starter()))
    units = {spec.name: spec.unit for spec in config.series}

    assert units["rainfall"] == "mm"
    assert units["mean_temperature"] == "°C"
    # Counts are self-evident, and soil_moisture is an index.
    assert units["disease_cases"] is None
    assert units.get("soil_moisture") is None

    # The reason is stated in the file, not just here.
    assert "no `unit:`" in STARTER_TEMPLATE


def test_the_starter_s_units_reach_the_plot():
    """The whole chain, not just the config: a declared unit has to survive
    into the panel heading a reader actually looks at."""
    import yaml

    from dsl.core.config.schema import parse_config
    from dsl.core.pipeline.engine import run
    from dsl.core.pipeline.plot import _build_figure

    config = parse_config(yaml.safe_load(_uncomment_starter()))
    units = {spec.name: spec.unit for spec in config.series if spec.unit}
    fig = _build_figure(run(config), units=units)
    titles = [a.text for a in fig.layout.annotations]

    assert "rainfall (mm)" in titles
    assert "mean_temperature (°C)" in titles
