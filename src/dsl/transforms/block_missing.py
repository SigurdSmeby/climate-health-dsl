"""Block missingness: blank contiguous runs, simulating reporting outages.

Unlike ``missing`` (independent points), real surveillance gaps are usually
contiguous — a facility offline for weeks — so a model that interpolates
single points still faces a hard stretch of nothing.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("block_missing")
class BlockMissingTransform(Transform):
    """Blank n_blocks contiguous runs of length block_len to NaN.

    Registered as "block_missing" in the transform registry. apply()
    returns the series with those runs set to NaN; blocks may overlap
    (fewer effective NaNs than n_blocks*block_len), and a block near the
    end is clipped at the boundary rather than wrapping.
    Example: array([50.5, nan, nan, nan, 72.5, ...]) for n_blocks=1,
    block_len=3.
    """

    def __init__(self, n_blocks: int = 1, block_len: int = 4):
        """Store the YAML params: for this transform. bool is accepted for
        n_blocks/block_len (bool subclasses int in Python — True/False are
        valid values here, not rejected as a wrong type).

        Errors Caught (raised to caller):
            ValueError: If n_blocks isn't an int >= 0, or block_len isn't
                an int >= 1. A float (e.g. a YAML author writing 4.5) is
                rejected here with a clear message rather than failing
                later with a cryptic numpy TypeError from rng.integers()
                or slice indexing.
        """
        if not isinstance(n_blocks, int) or n_blocks < 0:
            raise ValueError(f"n_blocks must be an int >= 0, got {n_blocks!r}")
        if not isinstance(block_len, int) or block_len < 1:
            raise ValueError(f"block_len must be an int >= 1, got {block_len!r}")
        self.n_blocks = n_blocks
        self.block_len = block_len

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Blank n_blocks contiguous runs of the series to NaN.

        Args:
            series: The input time series.
            rng: Seeded random generator, used to pick block start points.

        Returns:
            series with n_blocks runs of block_len values set to NaN.
            Blocks may overlap (fewer effective NaNs), and a block starting
            near the end is clipped at the boundary rather than wrapping.
        """
        result = series.astype(float)  # copy + NaN-capable
        n = len(result)
        if self.n_blocks == 0 or n == 0:
            return result
        starts = rng.integers(0, n, size=self.n_blocks)
        for s in starts:
            result[s : s + self.block_len] = np.nan
        return result
