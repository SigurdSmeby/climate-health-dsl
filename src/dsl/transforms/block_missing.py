"""Block missingness: blank contiguous runs, simulating reporting outages.

Unlike ``missing`` (independent points), real surveillance gaps are usually
contiguous — a facility offline for weeks — so a model that interpolates
single points still faces a hard stretch of nothing.
"""
import numpy as np

from dsl.core.extension.transform_base import Transform, register_transform


@register_transform("block_missing")
class BlockMissingTransform(Transform):
    """Blank ``n_blocks`` contiguous runs of length ``block_len`` to NaN."""

    def __init__(self, n_blocks: int = 1, block_len: int = 4):
        # isinstance(x, int) excludes bool implicitly is NOT true (bool is an
        # int subclass) but that's fine here — True/False are valid n_blocks.
        # A float (e.g. a YAML author writing 4.5) is what needs rejecting,
        # since rng.integers()/slice indexing both require real ints and
        # otherwise fail later with a cryptic numpy TypeError, not this
        # constructor's friendly ValueError.
        if not isinstance(n_blocks, int) or n_blocks < 0:
            raise ValueError(f"n_blocks must be an int >= 0, got {n_blocks!r}")
        if not isinstance(block_len, int) or block_len < 1:
            raise ValueError(f"block_len must be an int >= 1, got {block_len!r}")
        self.n_blocks = n_blocks
        self.block_len = block_len

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Blocks may overlap (fewer effective NaNs); a block near the end is
        clipped at the boundary rather than wrapping."""
        result = series.astype(float)  # copy + NaN-capable
        n = len(result)
        if self.n_blocks == 0 or n == 0:
            return result
        starts = rng.integers(0, n, size=self.n_blocks)
        for s in starts:
            result[s : s + self.block_len] = np.nan
        return result
