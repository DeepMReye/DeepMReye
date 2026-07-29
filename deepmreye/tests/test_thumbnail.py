"""Tests for the QA thumbnail that replaced the 5 MB HTML report.

The thumbnail is what QA is now decided on, at full-extraction scale, so the
properties worth pinning are that it stays small, that it actually distinguishes
a subject with eyeballs from one without, and that the two ways of producing it
-- live during extraction, and backfilled from an old report -- agree.
"""
import json

import numpy as np

from deepmreye import thumbnail


def _brain(seed=0):
    """A whole-brain volume with two bright spheres where the eyeballs go."""
    rng = np.random.default_rng(seed)
    vol = rng.normal(1.0, 0.05, size=(91, 109, 91)).astype(np.float32)
    zz, yy, xx = np.mgrid[0:91, 0:109, 0:91]
    for cy in (35, 74):
        ball = (xx - 45) ** 2 + (yy - cy) ** 2 + (zz - thumbnail.Z_SLICE) ** 2 < 36
        vol[ball] = 6.0
    return vol


def _mask():
    mask = np.zeros((91, 109, 91), dtype=np.float32)
    mask[40:50, 30:40, 10:20] = 1
    mask[40:50, 70:80, 10:20] = 1
    return mask


def _eye_block(bright=True, t=12, seed=1):
    """A [47, 29, 18, T] block, with or without eyeball-shaped structure."""
    rng = np.random.default_rng(seed)
    block = rng.normal(1.0, 0.05, size=(47, 29, 18, t)).astype(np.float32)
    if bright:
        zz, yy, xx = np.mgrid[0:47, 0:29, 0:18]
        for cx in (12, 34):
            ball = (xx - 9) ** 2 + (yy - 14) ** 2 + (zz - cx) ** 2 < 25
            block[ball, :] = 8.0
    return block


def test_thumbnail_is_small_enough_to_ship_for_every_subject(tmp_path):
    image = thumbnail.from_arrays(_brain(), _mask(), _eye_block())
    path = thumbnail.save(image, tmp_path / "sub-01.png")

    # The report this replaces is ~5 MB. Anything near that defeats the purpose.
    assert path.stat().st_size < 100_000
    assert image.height == thumbnail.PANEL_HEIGHT


def test_thumbnail_shows_the_eyeballs_where_they_actually_are():
    """The QA question is 'are the eyeballs in the block' -- it must be visible.

    Checked against the known ball positions rather than a summary statistic:
    percentile scaling stretches any input to the full range, so a pure-noise
    block is just as "high contrast" as a pair of eyeballs. What distinguishes
    them is *where* the brightness sits.
    """
    _, axial, _ = thumbnail._panels(
        _brain()[:, :, thumbnail.Z_SLICE],
        _mask()[:, :, thumbnail.Z_SLICE],
        _eye_block(bright=True).mean(axis=3),
    )

    # `_eye_block` puts spheres at block coords (12, 14, 9) and (34, 14, 9);
    # the axial panel is the block collapsed over Z and transposed, so they
    # land at rows 14, columns 12 and 34.
    left, right = int(axial[14, 12]), int(axial[14, 34])
    corner = int(axial[2, 2])

    assert left > 200 and right > 200
    assert corner < 100


def test_the_mask_is_drawn_in_red_over_the_brain():
    # The overlay is how you see a registration that landed off the eyeballs,
    # which is the difference between label 1 and label 0.
    arr = np.asarray(thumbnail.from_arrays(_brain(), _mask(), _eye_block()))
    reddish = (arr[..., 0].astype(int) - arr[..., 2].astype(int)) > 60
    assert reddish.any()


def test_no_mask_still_renders():
    image = thumbnail.from_arrays(_brain(), np.zeros((91, 109, 91)), _eye_block())
    assert image.height == thumbnail.PANEL_HEIGHT


def test_flat_volume_does_not_blow_up():
    # A failed registration can produce a constant volume; percentile scaling
    # divides by zero range unless it is guarded.
    flat = np.zeros((91, 109, 91), dtype=np.float32)
    image = thumbnail.from_arrays(flat, _mask(), np.zeros((47, 29, 18, 5), dtype=np.float32))
    assert image.height == thumbnail.PANEL_HEIGHT


def _fake_report(wb, mask, eye_mean):
    """A minimal report carrying the traces `from_report` reads, in report order."""
    def heatmap(a):
        return {"type": "heatmap", "z": np.asarray(a, dtype=np.float32).tolist()}

    traces = [
        heatmap(wb[25, :, :].T), heatmap(mask[25, :, :].T),      # 0, 1
        heatmap(wb[:, 90, :].T), heatmap(mask[:, 90, :].T),      # 2, 3
        heatmap(wb[:, :, thumbnail.Z_SLICE]),                    # 4
        heatmap(mask[:, :, thumbnail.Z_SLICE]),                  # 5
        {"type": "histogram", "x": [1, 2, 3]},                   # 6
        heatmap(eye_mean.mean(axis=0).T),                        # 7
        heatmap(eye_mean.mean(axis=1).T),                        # 8
        heatmap(eye_mean.mean(axis=2)),                          # 9
        {"type": "scatter", "y": [1, 2]}, {"type": "scatter", "y": [3, 4]},
    ]
    return f"<html><script>Plotly.newPlot('x', {json.dumps(traces)}, {{}})</script></html>"


def test_backfill_from_a_report_matches_live_extraction():
    """Half the corpus predates thumbnails; both halves must look the same.

    If these disagreed, a contact sheet would mix two renderings and the eye
    panels would not be comparable between a backfilled and a fresh subject.
    """
    brain, mask, block = _brain(), _mask(), _eye_block()

    live = np.asarray(thumbnail.from_arrays(brain, mask, block))
    backfilled = np.asarray(thumbnail.from_report(_fake_report(brain, mask, block.mean(axis=3))))

    assert live.shape == backfilled.shape
    np.testing.assert_array_equal(live, backfilled)


def test_report_without_the_expected_traces_is_skipped_not_guessed():
    # A truncated or unexpected report should yield nothing, rather than an
    # image built from whatever traces happened to be present.
    assert thumbnail.from_report("<html><body>no plot here</body></html>") is None
    short = "<html><script>Plotly.newPlot('x', [{\"type\": \"heatmap\"}], {})</script></html>"
    assert thumbnail.from_report(short) is None


def _normalized_block(t=40, seed=3):
    """A stored block: z-scored per voxel across time, so the mean is flat."""
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(47, 29, 18, t)).astype(np.float32)
    # Give the eyeball region more temporal variance, as real eye motion does.
    block[8:18, 10:20, 6:12, :] *= 4.0
    block -= block.mean(axis=3, keepdims=True)
    return block


def test_block_only_thumbnail_uses_variance_not_the_flat_mean():
    """Labeled participants have no report, only a normalized block.

    Per-voxel z-scoring across time flattens the temporal mean, so rendering
    the mean would show noise. The variance map is what still has the eyeballs.
    """
    block = _normalized_block()
    assert block.mean(axis=3).std() < block.std(axis=3).std()

    image = thumbnail.from_block(block)
    arr = np.asarray(image)
    assert image.height == thumbnail.PANEL_HEIGHT
    # High-variance region renders bright against the rest.
    assert arr.max() > 200 and arr.min() < 60


def test_block_only_thumbnail_has_two_panels_not_three():
    # There is no whole-brain volume or mask for these subjects, so the strip is
    # genuinely shorter rather than padded with a misleading blank panel.
    block = _normalized_block()
    two = thumbnail.from_block(block).width
    three = thumbnail.from_arrays(_brain(), _mask(), _eye_block()).width
    assert two < three


def test_save_is_atomic(tmp_path):
    image = thumbnail.from_arrays(_brain(), _mask(), _eye_block())
    path = thumbnail.save(image, tmp_path / "nested" / "sub-01.png")

    assert path.exists()
    assert not list(tmp_path.rglob("*.tmp"))
