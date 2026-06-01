"""Real-data-backed covariates: read a variable's values from a CHAP CSV.

Instead of synthesizing a series, this generator takes it from a real
dataset (e.g. real Laos rainfall), while the disease signal is still
generated synthetically on top — a "semi-synthetic" experiment: realistic
weather, controlled cause→effect.

Hard rule: the generator never invents data. If the CSV holds fewer
periods than the scenario asks for, that is an error — real data is never
wrapped, repeated, or extrapolated beyond its available dates. (The old
reference code silently wrapped into the next region's rows; that was a
data-integrity bug, deliberately not reproduced.)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from dsl.core.extension.generator_base import VariableGenerator, register_generator

# What a period label looks like per resolution, used to catch a scenario
# whose resolution does not match the source data (e.g. weekly scenario,
# monthly CSV). Checked on the first label only — cheap and catches the
# realistic mistake.
_LABEL_SHAPE = {
    "monthly": r"^\d{4}-\d{2}$",
    "weekly": r"^\d{4}-W\d{2}$",
    "daily": r"^\d{8}$",
    "yearly": r"^\d{4}$",
}


@register_generator("from_csv")  # this string is what you write in YAML
class FromCsvGenerator(VariableGenerator):
    """Reads a column from a CHAP-format CSV instead of synthesizing it."""

    def __init__(
        self,
        file: str,
        column: str,
        source_location: str | None = None,
        start_period: str | None = None,
    ):
        """Store and validate the YAML ``params:`` for this variable.

        Parameters
        ----------
        file:
            Path to the CSV (CHAP format: a ``time_period`` column plus
            data columns; a ``location`` column if multi-location).
        column:
            Which column to use as this variable's values.
        source_location:
            Which location's rows to use. Required when the CSV contains
            more than one location.
        start_period:
            A ``time_period`` label to start reading from (e.g. "2011-01").
            Defaults to the first row.
        """
        path = Path(file)
        if not path.is_file():
            raise ValueError(f"from_csv: file not found: {file}")
        self.path = path
        self.column = column
        self.source_location = source_location
        self.start_period = start_period

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Return the first ``n_periods`` real values. ``rng`` is unused."""
        df = pd.read_csv(self.path)

        if self.column not in df.columns:
            raise ValueError(
                f"from_csv: column '{self.column}' not in {self.path.name}. "
                f"Available columns: {sorted(df.columns)}"
            )

        df = self._select_location(df)

        if "time_period" in df.columns:
            self._check_resolution(df, period)
            df = self._apply_start_period(df)

        values = df[self.column].to_numpy(dtype=float)
        if len(values) < n_periods:
            raise ValueError(
                f"from_csv: {self.path.name} has only {len(values)} periods "
                f"available for '{self.column}', but the scenario needs "
                f"{n_periods}. Real data is never wrapped or extrapolated."
            )
        return values[:n_periods]

    def _select_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce a multi-location CSV to the configured source location."""
        if "location" not in df.columns:
            return df
        available = list(df["location"].unique())
        if self.source_location is None:
            if len(available) > 1:
                raise ValueError(
                    f"from_csv: {self.path.name} contains several locations; "
                    f"set source_location to one of {available}."
                )
            return df
        if self.source_location not in available:
            raise ValueError(
                f"from_csv: source_location '{self.source_location}' not in "
                f"{self.path.name}. Available: {available}"
            )
        return df[df["location"] == self.source_location]

    def _check_resolution(self, df: pd.DataFrame, period: str) -> None:
        """Refuse a scenario resolution that doesn't match the source data."""
        first_label = str(df["time_period"].iloc[0])
        if not df["time_period"].astype(str).str.match(_LABEL_SHAPE[period]).all():
            raise ValueError(
                f"from_csv: the scenario is {period}, but {self.path.name} "
                f"has time_period labels like '{first_label}', which do not "
                f"match that resolution."
            )

    def _apply_start_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows before the configured start_period, if one is set."""
        if self.start_period is None:
            return df
        labels = df["time_period"].astype(str).tolist()
        if self.start_period not in labels:
            raise ValueError(
                f"from_csv: start_period '{self.start_period}' not found in "
                f"{self.path.name} (data covers {labels[0]} to {labels[-1]})."
            )
        return df.iloc[labels.index(self.start_period):]
