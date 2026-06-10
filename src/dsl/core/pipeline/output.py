"""The last pipeline step: write the finished DataFrame to disk, CHAP-style.

Every CHAP-specific naming and format decision lives in this one module, so
if CHAP's conventions change there is exactly one place to adjust. (Column
names and period formats were verified against ``chap_core``'s
``ClimateHealthTimeSeries``/``FullData`` and an example dataset.)
"""
import math
from pathlib import Path

import pandas as pd

from dsl.core.config.schema import ScenarioConfig

# CHAP's expected filename for a simulated dataset.
FULL_FILENAME = "simulated_data.csv"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"


def write_output(df: pd.DataFrame, config: ScenarioConfig, out_dir: str | Path) -> None:
    """Write the dataset to ``out_dir`` in CHAP's format.

    Always writes one ``simulated_data.csv`` — the full series with
    ``disease_cases`` filled in everywhere except the blanked lag warm-up.
    Do NOT drop or hide test-period values: CHAP needs the true values and
    does its own train/test hiding during evaluation.

    If ``config.train_fraction`` is set, ALSO writes ``train.csv`` (the
    first ``floor(n_total * train_fraction)`` periods of each location) and
    ``test.csv`` (the rest) — a split in time with all columns intact, for
    evaluating models *outside* CHAP.
    """
    out_dir = Path(out_dir)
    # parents=True creates intermediate folders; exist_ok makes rerunning
    # into the same folder fine (files are simply overwritten).
    out_dir.mkdir(parents=True, exist_ok=True)

    # index=False: pandas would otherwise prepend its row index as an
    # unnamed first column, which CHAP's reader does not expect.
    df.to_csv(out_dir / FULL_FILENAME, index=False)

    if config.train_fraction is not None:
        n_train = math.floor(config.n_total * config.train_fraction)
        # The split is in TIME, applied per location: a plain row split
        # would put whole locations in train and others in test. groupby
        # keeps each location's block together; head/tail take its first
        # n_train periods and the remainder respectively.
        by_location = df.groupby("location", sort=False)
        train = by_location.head(n_train)
        test = by_location.tail(config.n_total - n_train)
        train.to_csv(out_dir / TRAIN_FILENAME, index=False)
        test.to_csv(out_dir / TEST_FILENAME, index=False)
    else:
        # Reusing a directory must not leave a previous run's split files
        # beside the new full dataset (they would describe a different run).
        (out_dir / TRAIN_FILENAME).unlink(missing_ok=True)
        (out_dir / TEST_FILENAME).unlink(missing_ok=True)
