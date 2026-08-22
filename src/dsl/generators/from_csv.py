"""Real-data-backed covariates: read a variable's values from a CHAP CSV.

Instead of synthesizing a series, this generator takes it from a real
dataset (e.g. real Laos rainfall), while the disease signal is still
generated synthetically on top — a "semi-synthetic" experiment: realistic
weather, controlled cause→effect.

Hard rule: the generator never invents data. If the CSV holds fewer
periods than the scenario asks for, that is an error — real data is never
wrapped, repeated, or extrapolated beyond its available dates.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from dsl.core.extension.generator_base import VariableGenerator, register_generator
from dsl.core.pipeline.periods import parse_period, periods_per_year

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
    """Reads a column from a CHAP-format CSV instead of synthesizing it.

    Registered as "from_csv" in the generator registry. generate() returns
    the column's first n_periods values, unmodified — never wrapped,
    repeated, or extrapolated beyond the file's real dates.
    Example: array([12.1, 15.4, 9.8, 22.0, ...]).
    """

    @staticmethod
    def locations_in(file: str) -> list[str]:
        """List the distinct location values in a CSV.

        Lets the engine decide whether to auto-map each output location to
        its own rows. Reads only the location column, and tolerates a
        missing file (generation surfaces that error later).

        Args:
            file: Path to the CSV.

        Returns:
            Distinct values from the "location" column, in first-seen
            order. Empty list if the file is missing, empty, or has no
            "location" column.
        """
        path = Path(file)
        if not path.is_file():
            return []
        try:
            col = pd.read_csv(path, usecols=["location"])["location"]
        except (ValueError, pd.errors.EmptyDataError):
            return []  # no location column / empty file
        return list(dict.fromkeys(col.tolist()))  # distinct, order-preserving

    def __init__(
        self,
        file: str,
        column: str,
        source_location: str | None = None,
        start_period: str | None = None,
    ):
        """Store the YAML params: for this variable.

        Args:
            source_location: Which location's rows to use. Required when
                the CSV contains more than one location.
            start_period: A time_period label to start reading from (e.g.
                "2011-01"). Defaults to the first row.

        Errors Caught (raised to caller):
            ValueError: If file does not exist.
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
        """Read the first n_periods real values from the CSV.

        Args:
            n_periods: Number of time periods the scenario needs.
            period: Period type (e.g., "monthly", "daily"), checked against
                the CSV's time_period label format.
            rng: Seeded random generator; unused (the data is real, not
                drawn).

        Returns:
            A float array of length n_periods, read straight from the CSV
            column (never wrapped, repeated, or extrapolated).
            Example: array([12.1, 15.4, 9.8, 22.0, ...]).

        Errors Caught (raised to caller):
            ValueError: If the CSV is empty or missing the column, the
                column has non-numeric or non-finite values, the CSV has
                fewer than n_periods rows, or (via the helpers below) the
                time_period labels don't match, aren't consecutive, or
                don't cover the requested source_location/start_period.
        """
        try:
            df = pd.read_csv(self.path)
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
            # Without a time_period column there is nothing to align to, so a
            # start_period can't be honored — fail loudly rather than ignore it.
            raise ValueError(
                f"from_csv: start_period '{self.start_period}' was set, but "
                f"{self.path.name} has no 'time_period' column to align to."
            )

        # Coerce to float with a clear error for non-numeric content (text,
        # thousands separators, ...) instead of a raw numpy ValueError.
        values = pd.to_numeric(df[self.column], errors="coerce").to_numpy(dtype=float)
        original = df[self.column]
        introduced_nan = np.isnan(values)
        was_blank = original.isna().to_numpy()
        bad = introduced_nan & ~was_blank
        if bad.any():
            example = original.to_numpy()[bad][0]
            raise ValueError(
                f"from_csv: column '{self.column}' in {self.path.name} has a "
                f"non-numeric value ({example!r}); covariate columns must be "
                f"numeric."
            )

        # Reject infinities — numeric, but not a valid covariate value (and
        # CHAP would treat them as data). NaN (a blank cell) is allowed.
        if np.isinf(values).any():
            raise ValueError(
                f"from_csv: column '{self.column}' in {self.path.name} has a "
                f"non-finite (infinite) value; covariate values must be finite."
            )

        if len(values) < n_periods:
            raise ValueError(
                f"from_csv: {self.path.name} has only {len(values)} periods "
                f"available for '{self.column}', but the scenario needs "
                f"{n_periods}. Real data is never wrapped or extrapolated."
            )
        return values[:n_periods]

    def _order_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Sort rows by time_period, reject duplicates, and require no gaps.

        Real CSV exports are often unsorted; reading in file order would map
        values to the wrong periods. Duplicate periods are ambiguous and gaps
        (a missing month) would silently relabel later values onto earlier
        slots — both are errors. Ordering uses the parsed calendar position,
        not string order, so labels are validated as real dates here too.

        Args:
            df: Rows for this variable (already reduced to one location).
            period: Period type (e.g., "monthly", "daily").

        Returns:
            df sorted chronologically by time_period, with the internal
            ordering column dropped and the index reset.

        Errors Caught (raised to caller):
            ValueError: If time_period has duplicates, an unparseable
                label, or a gap between consecutive periods.
        """
        labels = df["time_period"].astype(str)
        dupes = labels[labels.duplicated()].unique()
        if len(dupes) > 0:
            raise ValueError(
                f"from_csv: {self.path.name} has duplicate time_period values: "
                f"{sorted(dupes)}."
            )
        # Map each label to an absolute period index — also validates the
        # calendar (2010-00 / 2010-13 / 20100230 raise) — so sorting by it is
        # chronological.
        ppy = periods_per_year(period)

        def absolute_index(label: str) -> int:
            year, offset = parse_period(label, period)
            return year * (ppy if period != "daily" else 366) + offset

        order = labels.map(absolute_index)
        df = df.assign(_order=order.to_numpy()).sort_values(
            "_order", kind="stable"
        )
        ordered = df["_order"].to_numpy()
        # Consecutive = each step is +1. Daily's year*366 base jumps at year
        # ends, so check daily consecutiveness from real dates instead.
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
        """Check daily labels for gaps using real dates.

        Daily's year*366 base (used by _order_by_period) jumps at year
        boundaries, so consecutiveness is checked via real calendar dates
        instead — this handles month/year/leap-year boundaries correctly.

        Args:
            labels: Sorted YYYYMMDD time_period labels.

        Errors Caught (raised to caller):
            ValueError: If any two consecutive dates are not exactly one
                day apart.
        """
        import datetime
        import itertools

        # Date-only labels, no timezone semantics involved.
        dates = [
            datetime.datetime.strptime(d, "%Y%m%d").date()  # noqa: DTZ007
            for d in labels
        ]
        for a, b in itertools.pairwise(dates):
            if (b - a).days != 1:
                raise ValueError(
                    f"from_csv: daily data has a gap between "
                    f"{a:%Y%m%d} and {b:%Y%m%d}; periods must be consecutive."
                )

    def _select_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce a multi-location CSV to the configured source location.

        Args:
            df: The full CSV, as loaded.

        Returns:
            df unchanged if there's no location column or only one
            location; otherwise just the rows for self.source_location.

        Errors Caught (raised to caller):
            ValueError: If the CSV has several locations but
                source_location is unset, or source_location isn't one of
                the CSV's locations.
        """
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
        """Refuse a scenario resolution that doesn't match the source data.

        Args:
            df: Rows for this variable (already reduced to one location).
            period: Period type (e.g., "monthly", "daily").

        Errors Caught (raised to caller):
            ValueError: If any time_period label doesn't match period's
                expected shape (checked against the first label's format).
        """
        first_label = str(df["time_period"].iloc[0])
        if not df["time_period"].astype(str).str.match(_LABEL_SHAPE[period]).all():
            raise ValueError(
                f"from_csv: the scenario is {period}, but {self.path.name} "
                f"has time_period labels like '{first_label}', which do not "
                f"match that resolution."
            )

    def _apply_start_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows before the configured start_period, if one is set.

        Args:
            df: Rows for this variable, already ordered chronologically.

        Returns:
            df unchanged if start_period is None; otherwise the rows from
            start_period onward.

        Errors Caught (raised to caller):
            ValueError: If start_period is set but not present in df.
        """
        if self.start_period is None:
            return df
        labels = df["time_period"].astype(str).tolist()
        if self.start_period not in labels:
            raise ValueError(
                f"from_csv: start_period '{self.start_period}' not found in "
                f"{self.path.name} (data covers {labels[0]} to {labels[-1]})."
            )
        return df.iloc[labels.index(self.start_period):]
