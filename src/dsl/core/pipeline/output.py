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
    """Write the dataset to ``out_dir`` in CHAP's format.

    Always writes the full ``simulated_data.csv`` with true values everywhere
    (CHAP does its own train/test hiding). If ``train_fraction`` is set, also
    writes ``train.csv``/``test.csv`` — a per-location split in time, for
    evaluating models outside CHAP.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # index=False: CHAP's reader does not expect pandas' row-index column.
    df.to_csv(out_dir / FULL_FILENAME, index=False)

    if config.train_fraction is not None:
        n_train = math.floor(config.n_total * config.train_fraction)
        # Split in TIME per location — a plain row split would put whole
        # locations in train and others in test.
        by_location = df.groupby("location", sort=False)
        by_location.head(n_train).to_csv(out_dir / TRAIN_FILENAME, index=False)
        by_location.tail(config.n_total - n_train).to_csv(
            out_dir / TEST_FILENAME, index=False
        )
    else:
        # Don't leave a previous run's split files beside the new dataset.
        (out_dir / TRAIN_FILENAME).unlink(missing_ok=True)
        (out_dir / TEST_FILENAME).unlink(missing_ok=True)
