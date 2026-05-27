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
    first ``floor(n_total * train_fraction)`` rows) and ``test.csv`` (the
    rest) — a plain row split with all columns intact, for evaluating
    models *outside* CHAP.
    """
    out_dir = Path(out_dir)
    # parents=True creates intermediate folders; exist_ok makes rerunning
    # into the same folder fine (files are simply overwritten).
    out_dir.mkdir(parents=True, exist_ok=True)

    # index=False: pandas would otherwise prepend its row index as an
    # unnamed first column, which CHAP's reader does not expect.
    df.to_csv(out_dir / FULL_FILENAME, index=False)

    if config.train_fraction is not None:
        n_train = math.floor(len(df) * config.train_fraction)
        # iloc slices by row position; both halves keep every column.
        df.iloc[:n_train].to_csv(out_dir / TRAIN_FILENAME, index=False)
        df.iloc[n_train:].to_csv(out_dir / TEST_FILENAME, index=False)
