"""Real-data-backed covariates: read a variable's values from a CHAP CSV.

Semi-synthetic experiments: realistic weather from a real dataset, with the
disease signal still generated synthetically on top.

Hard rule: the generator never invents data. If the CSV holds fewer periods
than the scenario asks for, that is an error — real data is never wrapped,
repeated, or extrapolated.
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import parse_period, periods_per_year

# Expected label shape per resolution — catches a scenario whose resolution
# doesn't match the source data (e.g. weekly scenario, monthly CSV).
_LABEL_SHAPE = {
    "monthly": r"^\d{4}-\d{2}$",
    "weekly": r"^\d{4}-W\d{2}$",
    "daily": r"^\d{8}$",
    "yearly": r"^\d{4}$",
}


@functools.lru_cache(maxsize=8)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    """One parse per file: the engine re-reads the same CSV for every
    location (and again for locations_in). mtime in the key busts the cache
    when the file changes under --watch. Callers must not mutate the result."""
    return pd.read_csv(path)


@register_generator("from_csv")
class FromCsvGenerator(VariableGenerator):
    """Reads a column from a CHAP-format CSV instead of synthesizing it.

    Params: ``file`` (the CSV path), ``column`` (which column to read),
    ``source_location`` (required when the CSV has several locations),
    ``start_period`` (label to start reading from; default: the first row).
    """

    @staticmethod
    def locations_in(file: str) -> list[str]:
        """Distinct ``location`` values in a CSV (empty if none/missing file).

        Lets the engine decide whether to auto-map each output location to
        its own rows; a missing file surfaces its error later, at generation.
        """
        path = Path(file)
        if not path.is_file():
            return []
        try:
            df = _read_csv(str(path), path.stat().st_mtime)
        except pd.errors.EmptyDataError:
            return []
        if "location" not in df.columns:
            return []
        return list(dict.fromkeys(df["location"].tolist()))  # distinct, ordered

    def __init__(
        self,
        file: str,
        column: str,
        source_location: str | None = None,
        start_period: str | None = None,
    ):
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
        try:
            df = _read_csv(str(self.path), self.path.stat().st_mtime)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"from_csv: {self.path.name} is empty (no columns to read)."
            ) from exc
        if len(df) == 0:
            raise ValueError(f"from_csv: {self.path.name} has no data rows.")

        if self.column not in df.columns:
            raise ValueError(
                f"from_csv: column '{self.column}' not in {self.path.name}. "
                f"Available columns: {sorted(df.columns)}"
            )

        df = self._select_location(df)

        if "time_period" in df.columns:
            self._check_resolution(df, period)
            df = self._order_by_period(df, period)
            df = self._apply_start_period(df)
        elif self.start_period is not None:
            # Nothing to align to — fail loudly rather than ignore it.
            raise ValueError(
                f"from_csv: start_period '{self.start_period}' was set, but "
                f"{self.path.name} has no 'time_period' column to align to."
            )

        values = self._numeric_values(df)

        if len(values) < n_periods:
            raise ValueError(
                f"from_csv: {self.path.name} has only {len(values)} periods "
                f"available for '{self.column}', but the scenario needs "
                f"{n_periods}. Real data is never wrapped or extrapolated."
            )
        return values[:n_periods]

    def _numeric_values(self, df: pd.DataFrame) -> np.ndarray:
        """Coerce the column to float; reject non-numeric text and infinities.

        NaN (a blank cell) is allowed — CHAP masks missing values itself.
        """
        original = df[self.column]
        values = pd.to_numeric(original, errors="coerce").to_numpy(dtype=float)
        introduced_nan = values != values  # NaN != NaN
        bad = introduced_nan & ~original.isna().to_numpy()
        if bad.any():
            example = original.to_numpy()[bad][0]
            raise ValueError(
                f"from_csv: column '{self.column}' in {self.path.name} has a "
                f"non-numeric value ({example!r}); covariate columns must be "
                f"numeric."
            )
        if np.isinf(values).any():
            raise ValueError(
                f"from_csv: column '{self.column}' in {self.path.name} has a "
                f"non-finite (infinite) value; covariate values must be finite."
            )
        return values

    def _order_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Sort rows chronologically; reject duplicates and gaps.

        Real CSV exports are often unsorted; duplicates are ambiguous and a
        gap would silently relabel later values onto earlier slots. Sorting
        uses the parsed calendar position, which also validates the labels
        as real dates.
        """
        labels = df["time_period"].astype(str)
        dupes = labels[labels.duplicated()].unique()
        if len(dupes) > 0:
            raise ValueError(
                f"from_csv: {self.path.name} has duplicate time_period values: "
                f"{sorted(dupes)}."
            )
        ppy = periods_per_year(period)

        def absolute_index(label: str) -> int:
            year, offset = parse_period(label, period)
            return year * (ppy if period != "daily" else 366) + offset

        order = labels.map(absolute_index)
        df = df.assign(_order=order.to_numpy()).sort_values(
            "_order", kind="stable"
        )
        ordered = df["_order"].to_numpy()
        # Daily's year*366 base jumps at year ends, so daily consecutiveness
        # is checked from real dates instead of index steps.
        if period == "daily":
            self._check_daily_consecutive(df["time_period"].astype(str).tolist())
        else:
            gaps = np.where(np.diff(ordered) != 1)[0]
            if len(gaps) > 0:
                sorted_labels = df["time_period"].astype(str).tolist()
                i = gaps[0]
                raise ValueError(
                    f"from_csv: {self.path.name} has a gap in time_period "
                    f"between '{sorted_labels[i]}' and '{sorted_labels[i + 1]}'; "
                    f"periods must be consecutive (no missing periods)."
                )
        return df.drop(columns="_order").reset_index(drop=True)

    @staticmethod
    def _check_daily_consecutive(labels: list[str]) -> None:
        """Daily gap check via real dates (handles month/year/leap boundaries)."""
        import datetime

        dates = [datetime.datetime.strptime(d, "%Y%m%d").date() for d in labels]
        for a, b in zip(dates, dates[1:], strict=False):
            if (b - a).days != 1:
                raise ValueError(
                    f"from_csv: daily data has a gap between "
                    f"{a:%Y%m%d} and {b:%Y%m%d}; periods must be consecutive."
                )

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
