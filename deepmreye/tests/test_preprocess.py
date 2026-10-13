"""Preprocessing: the header's TR, the eye crop, and the normalisation.

Registration itself needs ANTs and a real run, so it is exercised end to end
(README, "Check an installation") rather than here. What is pinned here is
what decides whether a new run lands in the corpus's distribution: the crop
geometry against the shipped masks, and the normalisation.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from deepmreye import preprocess
from deepmreye.basis import FULL_MASK_VOXELS


def _image(spacing):
    return SimpleNamespace(spacing=spacing)


def test_read_tr_from_the_fourth_spacing():
    assert preprocess.read_tr(_image((2.0, 2.0, 2.0, 1.5))) == 1.5


def test_read_tr_refuses_a_3d_image_and_a_nonsense_tr():
    with pytest.raises(preprocess.MissingTRError, match="4D"):
        preprocess.read_tr(_image((2.0, 2.0, 2.0)))
    for bad in (0.0, -1.0):
        with pytest.raises(preprocess.MissingTRError, match="invalid"):
            preprocess.read_tr(_image((2.0, 2.0, 2.0, bad)))
    with pytest.raises(preprocess.MissingTRError, match="implausible"):
        preprocess.read_tr(_image((2.0, 2.0, 2.0, 0.044)))
    assert preprocess.read_tr(_image((2.0, 2.0, 2.0, 0.044)), strict=False) == 0.044


def test_crop_of_the_shipped_mask_is_the_corpus_block():
    """The crop must give the corpus's [47, 29, 18] block with every one of the
    14,236 eye-mask voxels inside it, or no basis applies to the result."""
    masks = preprocess.get_masks()
    mask = masks["eyemask_small"].numpy()
    block = preprocess.crop_eyes(np.repeat(mask[..., None], 3, axis=-1), masks)
    assert block.shape == (47, 29, 18, 3)
    assert int((block[..., 0] > 0).sum()) == FULL_MASK_VOXELS


def test_crop_zeroes_everything_outside_the_eye_mask():
    masks = preprocess.get_masks()
    mask = masks["eyemask_small"].numpy()
    ones = np.ones(mask.shape + (2,), dtype=np.float32)
    block = preprocess.crop_eyes(ones, masks)
    assert int((block[..., 0] > 0).sum()) == FULL_MASK_VOXELS


def test_normalize_zscores_voxels_and_keeps_zeros():
    rng = np.random.default_rng(0)
    block = rng.normal(5.0, 2.0, size=(4, 3, 2, 50)).astype(np.float32)
    block[0, 0, 0] = 0.0                       # a voxel outside the field of view
    out = preprocess.normalize(block)
    assert out.shape == block.shape
    assert (out[0, 0, 0] == 0).all()
    assert np.isfinite(out).all()
    # Each volume is standardised over its non-zero voxels.
    vol = out[..., 7].reshape(-1)[1:]
    assert abs(vol.mean()) < 1e-5 and abs(vol.std() - 1) < 1e-4


def test_normalize_clips_outliers():
    block = np.random.default_rng(1).normal(size=(4, 4, 4, 40)).astype(np.float32)
    block[1, 1, 1, 5] = 1e4
    out = preprocess.normalize(block, std_cut=5)
    assert np.abs(out).max() <= 5 * np.nanstd(out) + 1.0
