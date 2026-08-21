"""The last pipeline step: write the finished DataFrame to disk, CHAP-style.

Every CHAP-specific naming and format decision lives here, so if CHAP's
conventions change there is one place to adjust.
"""
import math
from pathlib import Path

import pandas as pd

from dsl.core.config.schema import ScenarioConfig

FULL_FILENAME = "simulated_data.csv"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"


def write_output(df: pd.DataFrame, config: ScenarioConfig, out_dir: str | Path) -> None:
    """Write the dataset to disk in CHAP format.

    Always writes simulated_data.csv (full dataset with all true values —
    CHAP does its own train/test hiding). If config.train_fraction is set,
    also writes train.csv and test.csv (time-based split per location, for
    evaluating models outside CHAP).

    Args:
        df: The output DataFrame.
        config: The validated scenario configuration (for train_fraction
            and n_total).
        out_dir: Output directory path (created if it doesn't exist).

    Errors Caught (raised to caller):
        OSError: If the output directory cannot be created or written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Write the full dataset.
    # index=False: CHAP's reader does not expect pandas' row-index column.
    df.to_csv(out_dir / FULL_FILENAME, index=False)
    # Now out_dir/simulated_data.csv contains all rows, all columns true

    # Step 2: Write a train/test split if requested.
    if config.train_fraction is not None:
        n_train = math.floor(config.n_total * config.train_fraction)
        # Split in TIME per location: each location's first n_train periods
        # go to train, the rest to test. (A plain row split would instead
        # put whole locations in train and others in test.)
        by_location = df.groupby("location", sort=False)
        by_location.head(n_train).to_csv(out_dir / TRAIN_FILENAME, index=False)
        by_location.tail(config.n_total - n_train).to_csv(
            out_dir / TEST_FILENAME, index=False
        )
        # Now out_dir/train.csv and test.csv hold the time-based split
    else:
        # Don't leave a previous run's split files beside the new dataset.
        (out_dir / TRAIN_FILENAME).unlink(missing_ok=True)
        (out_dir / TEST_FILENAME).unlink(missing_ok=True)
