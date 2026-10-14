"""The two places where the synthetic world cannot stand in for the real one.

- **Registration** (ANTs): a run made from the template itself must come out of
  ``extract_eye_block`` with every eye-mask voxel covered. A few seconds.
- **The published release** (``network``, opt-in with
  ``DEEPMREYE_NETWORK_TESTS=1``): the decoder on the Hub, applied through this
  code to a published labelled participant, must decode it, and must carry
  the evaluation it was released with. This is what breaks if the feature path
  drifts from the one the release was fitted with.
"""
import numpy as np
import pytest

from deepmreye.basis import FULL_MASK_VOXELS


def test_a_run_made_of_the_template_registers_with_full_eye_coverage(tmp_path):
    ants = pytest.importorskip("ants")
    from deepmreye import preprocess

    masks = preprocess.get_masks()
    template = masks["template"]
    rng = np.random.default_rng(0)
    volumes = np.stack([template.numpy() * (1 + 0.01 * rng.normal(size=template.shape))
                        for _ in range(4)], axis=-1).astype(np.float32)
    direction = np.eye(4)
    direction[:3, :3] = template.direction
    run = ants.from_numpy(volumes, origin=list(template.origin) + [0.0],
                          spacing=list(template.spacing) + [2.0], direction=direction)
    ants.image_write(run, str(tmp_path / "run.nii.gz"))

    block, tr = preprocess.extract_eye_block(tmp_path / "run.nii.gz", qa_png=tmp_path / "qa.png")
    assert block.shape == (47, 29, 18, 4) and tr == 2.0
    assert int((np.abs(block).sum(axis=-1) > 0).sum()) == FULL_MASK_VOXELS
    assert np.isfinite(block).all()
    assert (tmp_path / "qa.png").stat().st_size > 0


@pytest.mark.network
def test_the_published_decoder_decodes_a_published_participant():
    from huggingface_hub import hf_hub_download

    from deepmreye.data import DEFAULT_REPO, read_subject
    from deepmreye.model import Decoder

    model = Decoder.pretrained()
    assert model.k == 32 and model.lags == 1 and model.units == "z"
    assert model.meta["evaluation"]["median_subtr"] == pytest.approx(0.771, abs=0.002)
    path = hf_hub_download(DEFAULT_REPO, "dsL02_pursuit/AS_18_08_14.h5", repo_type="dataset")
    block, labels, _attrs = read_subject(path)
    gaze = model.predict(block)
    ok = np.isfinite(labels).all(axis=(1, 2))
    r = np.mean([np.corrcoef(gaze[ok, :, a].ravel(), labels[ok, :, a].ravel())[0, 1]
                 for a in range(2)])
    assert r > 0.8
    assert model.confidence(block) > 0.5
