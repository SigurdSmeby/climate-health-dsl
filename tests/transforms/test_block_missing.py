"""Tests for the block_missing transform (contiguous reporting outages).

TDD: written before src/dsl/transforms/block_missing.py. The existing `missing`
transform blanks points independently (MCAR); real reporting gaps come in
contiguous runs (a facility offline for weeks). This blanks whole blocks.
"""
import numpy as np
import pytest

from dsl.transforms.block_missing import BlockMissingTransform


def test_no_blocks_is_identity(rng):
    series = np.arange(20.0)
    result = BlockMissingTransform(n_blocks=0, block_len=5).apply(series, rng)
    assert np.array_equal(result, series)


def test_blanks_a_contiguous_run(rng):
    series = np.arange(50.0)
    result = BlockMissingTransform(n_blocks=1, block_len=6).apply(series, rng)
    nan_idx = np.where(np.isnan(result))[0]
    assert len(nan_idx) == 6
    # Contiguous: the blanked indices form one unbroken run.
    assert np.array_equal(nan_idx, np.arange(nan_idx[0], nan_idx[0] + 6))


def test_multiple_blocks(rng):
    series = np.arange(200.0)
    result = BlockMissingTransform(n_blocks=3, block_len=4).apply(series, rng)
    # Up to 3*4 blanked (blocks could touch/overlap → allow <=).
    assert 0 < np.isnan(result).sum() <= 12


def test_block_clipped_at_end(rng):
    # A block starting near the end must not error, just blank to the boundary.
    series = np.arange(10.0)
    result = BlockMissingTransform(
        n_blocks=1, block_len=8).apply(series, np.random.default_rng(0))
    assert np.isnan(result).sum() >= 1


def test_does_not_modify_input(rng):
    series = np.arange(20.0)
    original = series.copy()
    BlockMissingTransform(n_blocks=1, block_len=3).apply(series, rng)
    assert np.array_equal(series, original)


def test_deterministic_under_seed():
    a = BlockMissingTransform(n_blocks=2, block_len=5).apply(
        np.arange(100.0), np.random.default_rng(0))
    b = BlockMissingTransform(n_blocks=2, block_len=5).apply(
        np.arange(100.0), np.random.default_rng(0))
    assert np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="n_blocks"):
        BlockMissingTransform(n_blocks=-1, block_len=5)
    with pytest.raises(ValueError, match="block_len"):
        BlockMissingTransform(n_blocks=1, block_len=0)


def test_registered_and_reachable():
    from dsl.core.extension.transform_base import get_transform

    assert get_transform("block_missing") is BlockMissingTransform
