"""Describe what each cross-validation fold actually contains.

Splitting a dataset raises a question the split itself cannot answer: does
every fold still test what the scenario planted? A fold whose test half holds
no outbreak cannot measure outbreak recovery, however well the model scores.

This module counts the planted features — seasonal peaks, outbreak shocks,
declared events — on each side of every fold, and flags the folds that come up
empty. The counts come from the generators' own reports rather than from
detecting spikes in the output, so they are exact.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dsl.core.config.schema import ScenarioConfig
from dsl.core.pipeline.engine import series_events

REPORT_MD = "report.md"
REPORT_JSON = "report.json"


def evaluation_boundary(config: ScenarioConfig) -> int | None:
    """The period where the first fold starts being evaluated.

    What a chart should mark, so a reader sees which part of the series a
    model is scored on. Only a time split has such a period at all: holding a
    location out divides the data sideways, with every period on both sides.

    Args:
        config: The validated scenario configuration.

    Returns:
        The first test period of fold 0, or None when the scenario has no
        split or splits by location. Note this is NOT the size of fold 0's
        training set — those coincide only for an expanding split, since a
        blocked fold trains on the periods AFTER its test block.

    Errors Caught (raised to caller):
        ValueError: If the split is invalid (already rejected at parse time).
    """
    if config.split is None or config.split.kind != "time":
        return None
    folds = config.folds()
    return folds[0].test_periods[0] if folds else None


def build_report(config: ScenarioConfig, df: pd.DataFrame) -> dict | None:
    """Summarise every fold: size, coverage, missing values, planted features.

    Args:
        config: The validated scenario configuration.
        df: The generated dataset, as written to simulated_data.csv.

    Returns:
        A dict describing the split and one entry per fold, or None when the
        scenario declares no split. Machine-readable: the markdown report is
        rendered from exactly this.

    Errors Caught (raised to caller):
        KeyError: If a generator name is not registered.
    """
    folds = config.folds()
    if not folds:
        return None

    events = series_events(config)
    labels = _period_labels(df)
    count_columns = [spec.name for spec in config.series if spec.counts]

    entries = []
    for fold in folds:
        entries.append(
            {
                "index": fold.index,
                "train": _side(
                    df, labels, events, fold.train_periods, fold.train_locations,
                    count_columns,
                ),
                "test": _side(
                    df, labels, events, fold.test_periods, fold.test_locations,
                    count_columns,
                ),
            }
        )

    report = {
        "split": {
            "kind": config.split.kind,
            "k": len(folds),
            "scheme": config.split.scheme,
        },
        "total_events": len(events),
        "folds": entries,
        "warnings": _warnings(entries, events),
    }
    return report


def write_report(
    config: ScenarioConfig, df: pd.DataFrame, out_dir: str | Path
) -> None:
    """Write the fold report beside the folds it describes.

    Two files with the same content: report.json for analysis code, report.md
    for reading — the markdown holds CSV-style tables, so a table can be
    pasted straight into a spreadsheet or a thesis.

    Args:
        config: The validated scenario configuration.
        df: The generated dataset.
        out_dir: The run's output directory; the report goes in its folds/
            subdirectory.

    Errors Caught (raised to caller):
        OSError: If the directory cannot be written to.
    """
    report = build_report(config, df)
    if report is None:
        return
    folds_dir = Path(out_dir) / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    (folds_dir / REPORT_JSON).write_text(json.dumps(report, indent=2) + "\n")
    (folds_dir / REPORT_MD).write_text(render_markdown(report))


def render_markdown(report: dict) -> str:
    """Render a report as markdown with CSV-style tables.

    Args:
        report: The dict from build_report.

    Returns:
        The full markdown document, ending in a newline.
    """
    split = report["split"]
    scheme = f" ({split['scheme']})" if split["scheme"] else ""
    lines = [
        f"# Fold report — {split['kind']} split{scheme}, k={split['k']}",
        "",
        "## Size and coverage",
        "",
        "```",
        "fold,split,rows,periods,locations,first_period,last_period,missing",
    ]
    for fold in report["folds"]:
        for side in ("train", "test"):
            entry = fold[side]
            missing = sum(entry["missing"].values())
            lines.append(
                f"{fold['index']},{side},{entry['rows']},{entry['periods']},"
                f"{entry['locations']},{entry['first_period']},"
                f"{entry['last_period']},{missing}"
            )
    lines += ["```", ""]

    lines += ["## Planted features per fold", ""]
    if report["total_events"] == 0:
        lines += [
            "This scenario plants no countable features — no seasonal peaks,",
            "outbreak shocks or declared events.",
            "",
        ]
    else:
        lines += ["```", "fold,series,kind,train,test"]
        for fold in report["folds"]:
            for series, kinds in _merged_kinds(fold).items():
                for kind in sorted(kinds):
                    train = fold["train"]["events"].get(series, {}).get(kind, 0)
                    test = fold["test"]["events"].get(series, {}).get(kind, 0)
                    lines.append(
                        f"{fold['index']},{series},{kind},{train},{test}"
                    )
        lines += ["```", ""]

    lines += ["## Warnings", ""]
    if report["warnings"]:
        lines += [f"- {w}" for w in report["warnings"]]
    else:
        lines.append("None — every fold tests every kind of planted feature.")
    lines.append("")
    return "\n".join(lines)


def _side(
    df: pd.DataFrame,
    labels: list[str],
    events: list[dict],
    periods: list[int],
    locations: list[str],
    count_columns: list[str],
) -> dict:
    """Summarise one side (train or test) of one fold.

    Args:
        df: The generated dataset.
        labels: Period offset -> time_period label.
        events: Every planted feature, from series_events.
        periods: The period offsets on this side.
        locations: The location names on this side.
        count_columns: Names of the count series, whose missing values matter.

    Returns:
        Row/period/location counts, the calendar span, missing values per
        count series, and planted features per series per kind.
    """
    wanted = set(periods)
    position = df.groupby("location", sort=False).cumcount()
    mask = df["location"].isin(locations) & position.isin(wanted)
    rows = df[mask]

    missing = {
        column: int(rows[column].isna().sum())
        for column in count_columns
        if column in rows
    }

    counted: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in events:
        # A scenario event is regional (location None), so it counts once
        # wherever its periods fall; a generator's feature belongs to the
        # location that drew it.
        if event["location"] is not None and event["location"] not in locations:
            continue
        if event["index"] in wanted:
            counted[event["series"]][event["kind"]] += 1

    return {
        "rows": len(rows),
        "periods": len(periods),
        "locations": len(locations),
        "location_names": list(locations),
        "first_period": labels[min(periods)] if periods else None,
        "last_period": labels[max(periods)] if periods else None,
        "missing": missing,
        "events": {series: dict(kinds) for series, kinds in counted.items()},
    }


def _warnings(entries: list[dict], events: list[dict]) -> list[str]:
    """Flag folds whose test half cannot measure something that was planted.

    A fold with no outbreak in its test half will score well on outbreak
    recovery for the wrong reason — there was nothing to recover.

    Args:
        entries: The per-fold summaries.
        events: Every planted feature, used to know what kinds exist at all.

    Returns:
        One message per (fold, series, kind) that is planted in the scenario
        but absent from that fold's test half. Empty when every fold tests
        everything.
    """
    planted: dict[str, set[str]] = defaultdict(set)
    for event in events:
        planted[event["series"]].add(event["kind"])

    warnings = []
    for fold in entries:
        for series, kinds in sorted(planted.items()):
            for kind in sorted(kinds):
                test = fold["test"]["events"].get(series, {}).get(kind, 0)
                if test:
                    continue
                train = fold["train"]["events"].get(series, {}).get(kind, 0)
                warnings.append(
                    f"fold {fold['index']}: no '{kind}' in the test half of "
                    f"'{series}' ({train} in train); this fold cannot measure "
                    f"how well a model recovers it."
                )
    return warnings


def _merged_kinds(fold: dict) -> dict[str, set[str]]:
    """Every (series, kind) pair appearing on either side of a fold.

    Args:
        fold: One per-fold summary.

    Returns:
        Series name -> the kinds seen for it, so a kind present only in train
        still gets a row (showing 0 in test).
    """
    merged: dict[str, set[str]] = defaultdict(set)
    for side in ("train", "test"):
        for series, kinds in fold[side]["events"].items():
            merged[series].update(kinds)
    return merged


def _period_labels(df: pd.DataFrame) -> list[str]:
    """The time_period label for each period offset, in order.

    Args:
        df: The generated dataset, which repeats its periods per location.

    Returns:
        One label per offset, taken from the first location's rows.
    """
    first = df[df["location"] == df["location"].iloc[0]]
    return list(first["time_period"])
