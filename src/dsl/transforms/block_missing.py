"""Block missingness: blank contiguous runs, simulating reporting outages.

The ``missing`` transform drops points independently (MCAR). Real surveillance
gaps are usually contiguous — a facility offline for weeks, a system migration —
so a model that interpolates single points still faces a hard stretch of
nothing. This blanks ``n_blocks`` runs of length ``block_len`` at random starts.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("block_missing")  # this string is what you write in YAML
class BlockMissingTransform(Transform):
    """Blank ``n_blocks`` contiguous runs of length ``block_len`` to NaN."""

    def __init__(self, n_blocks: int = 1, block_len: int = 4):
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        if block_len < 1:
            raise ValueError(f"block_len must be >= 1, got {block_len}")
        self.n_blocks = n_blocks
        self.block_len = block_len

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a copy with ``n_blocks`` contiguous runs blanked. Blocks may
        overlap or touch (fewer effective NaNs); a block near the end is clipped
        at the boundary rather than wrapping."""
        result = series.astype(float)  # copy + NaN-capable
        n = len(result)
        if self.n_blocks == 0 or n == 0:
            return result
        # Start anywhere a full-or-partial block fits; clip at the end.
        starts = rng.integers(0, n, size=self.n_blocks)
        for s in starts:
            result[s : s + self.block_len] = np.nan
        return result
