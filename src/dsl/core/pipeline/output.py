"""The last pipeline step: write the finished DataFrame to disk, CHAP-style.

Every CHAP-specific naming and format decision lives here, so if CHAP's
conventions change there is one place to adjust.
"""
import shutil
from pathlib import Path

import pandas as pd

from dsl.core.config.schema import ScenarioConfig

FULL_FILENAME = "simulated_data.csv"
FOLDS_DIRNAME = "folds"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"


def write_output(df: pd.DataFrame, config: ScenarioConfig, out_dir: str | Path) -> None:
    """Write the dataset to disk in CHAP format.

    Always writes simulated_data.csv (the full dataset with all true values —
    CHAP does its own train/test hiding). If the scenario declares a
    ``split:``, also writes one directory per fold holding that fold's
    train.csv and test.csv, ready to hand to a model unsliced.

    Args:
        df: The output DataFrame.
        config: The validated scenario configuration (for the split).
        out_dir: Output directory path (created if it doesn't exist).

    Errors Caught (raised to caller):
        OSError: If the output directory cannot be created or written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # index=False: CHAP's reader does not expect pandas' row-index column.
    df.to_csv(out_dir / FULL_FILENAME, index=False)

    folds_dir = out_dir / FOLDS_DIRNAME
    # Clear first, so a rerun with fewer folds (or none) leaves nothing of the
    # previous run behind to be mistaken for part of this one.
    if folds_dir.exists():
        shutil.rmtree(folds_dir)

    folds = config.folds()
    if not folds:
        return

    # Zero-pad to the widest index, so the directories sort in fold order.
    width = len(str(len(folds) - 1))
    for fold in folds:
        directory = folds_dir / f"fold_{fold.index:0{width}d}"
        directory.mkdir(parents=True)
        _rows(df, fold.train_periods, fold.train_locations).to_csv(
            directory / TRAIN_FILENAME, index=False
        )
        _rows(df, fold.test_periods, fold.test_locations).to_csv(
            directory / TEST_FILENAME, index=False
        )


def _rows(
    df: pd.DataFrame, periods: list[int], locations: list[str]
) -> pd.DataFrame:
    """Select one side of a fold from the full dataset.

    A fold names both periods and locations, so a time split (every location,
    some periods) and a location split (every period, some locations) are the
    same selection with different arguments.

    Args:
        df: The full dataset, in the engine's row order.
        periods: Period offsets to keep, as indices into each location's own
            run of rows.
        locations: Location names to keep.

    Returns:
        The matching rows, in the original order and with the original
        columns. Row indices are reset, so the written CSV starts at 0.
    """
    wanted = set(periods)
    # Each location's rows are generated in period order, so the position
    # within a location IS the period offset.
    position = df.groupby("location", sort=False).cumcount()
    mask = df["location"].isin(locations) & position.isin(wanted)
    return df[mask].reset_index(drop=True)
