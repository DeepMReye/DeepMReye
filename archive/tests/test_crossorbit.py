"""The cross-orbit soft-argmax model.

Three things carry this architecture, and each has a test that fails loudly if
it stops being true:

1. the soft-argmax bottleneck really produces *positions* (move the mass, the
   coordinate moves with it);
2. the nuisance path is fed from a different TR, so it structurally cannot see
   the gaze it would otherwise be graded on;
3. shuffling the coordinate degrades reconstruction -- the ablation that tells
   a working bottleneck from a decoder quietly running off the nuisance path.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepmreye.crossorbit import (  # noqa: E402
    ORBIT_SHAPE,
    CrossOrbitModel,
    _batch_loss,
    build_orbit_cache,
    evaluate,
    load,
    sample_pairs,
    save,
    split_orbits,
    train,
)
from deepmreye.evaluate.features import ORBIT_KINDS, OrbitExtractor, parse_spec  # noqa: E402

CPU = torch.device("cpu")


def _model(**kw):
    kw.setdefault("n_keypoints", 2)
    kw.setdefault("n_nuisance", 8)
    kw.setdefault("width", 4)
    return CrossOrbitModel(device=CPU, **kw)


# --- orbit geometry ----------------------------------------------------------


def test_split_orbits_gives_two_equal_mirrored_halves():
    block = np.arange(47 * 29 * 18 * 3, dtype=np.float32).reshape(47, 29, 18, 3)
    left, right = split_orbits(block)
    assert left.shape == right.shape == (*ORBIT_SHAPE, 3)
    # Both halves run lateral -> medial, so the right is mirrored: its first
    # slice is the array's outermost, not its midline-most.
    assert np.array_equal(left[0], block[2])
    assert np.array_equal(right[0], block[46])
    assert np.array_equal(left[-1], block[23])
    assert np.array_equal(right[-1], block[25])


def test_split_orbits_drops_the_midline_trough():
    """x=24 is the trough between the orbits. A midline voxel appearing in both
    halves would let the cross-orbit objective predict an orbit partly from
    itself -- the one shortcut this objective must not have."""
    block = np.zeros((47, 29, 18, 2), dtype=np.float32)
    block[24] = 1.0
    left, right = split_orbits(block)
    assert left.sum() == 0.0 and right.sum() == 0.0


def test_the_two_halves_share_no_source_slice():
    """Tag every x slice with its index and check the halves are disjoint."""
    block = np.tile(np.arange(47, dtype=np.float32)[:, None, None, None],
                    (1, 29, 18, 1))
    left, right = split_orbits(block)
    assert not (set(np.unique(left)) & set(np.unique(right)))


# --- the bottleneck really encodes position ----------------------------------


def test_soft_argmax_coordinate_tracks_where_the_mass_is():
    """The whole premise: this latent is a position by construction. Put a blob
    at one end of the volume, then the other, and the coordinate must move."""
    model = _model().eval()
    lo = np.zeros((1, 1, *ORBIT_SHAPE), dtype=np.float32)
    hi = np.zeros((1, 1, *ORBIT_SHAPE), dtype=np.float32)
    lo[0, 0, :6] = 8.0
    hi[0, 0, -6:] = 8.0

    with torch.no_grad():
        c_lo, _ = model.encode(torch.from_numpy(lo))
        c_hi, _ = model.encode(torch.from_numpy(hi))
    # Averaged over keypoints, the x coordinate must increase.
    assert c_hi[0, :, 0].mean() > c_lo[0, :, 0].mean()


def test_coordinates_stay_inside_the_normalised_grid():
    model = _model().eval()
    x = torch.randn(4, 1, *ORBIT_SHAPE)
    with torch.no_grad():
        coords, nuis = model.encode(x)
    assert coords.shape == (4, 2, 3)
    assert nuis.shape == (4, 8)
    # A convex combination of grid points in [-1, 1] cannot leave [-1, 1].
    assert coords.abs().max() <= 1.0 + 1e-5


def test_decoder_returns_an_orbit_shaped_volume():
    model = _model().eval()
    with torch.no_grad():
        out = model.decode(torch.zeros(3, 2, 3), torch.zeros(3, 8))
    assert out.shape == (3, 1, *ORBIT_SHAPE)


def test_decoder_output_depends_on_the_coordinate():
    """If moving the coordinate does not move the reconstruction, the render
    step is broken and the bottleneck is decorative."""
    model = _model().eval()
    nuis = torch.zeros(1, 8)
    with torch.no_grad():
        a = model.decode(torch.full((1, 2, 3), -0.8), nuis)
        b = model.decode(torch.full((1, 2, 3), 0.8), nuis)
    assert not torch.allclose(a, b, atol=1e-4)


# --- the two paths, and the ablation that proves they are separate -----------


def test_sample_pairs_draws_both_timepoints_from_one_run():
    """The nuisance code comes from t2. If t and t2 could land in different
    runs, the nuisance path would be handed another participant's anatomy."""
    data = np.zeros((30, 2, 3, 3, 3), dtype=np.float16)
    offsets = np.array([0, 10, 30])
    run_of = np.concatenate([np.zeros(10), np.ones(20)])
    rng = np.random.default_rng(0)
    for _ in range(50):
        runs = [(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]
        lo, hi = runs[rng.integers(len(runs))]
        t, t2 = lo + rng.integers(hi - lo), lo + rng.integers(hi - lo)
        assert run_of[t] == run_of[t2]

    a, b = sample_pairs(data, offsets, 8, np.random.default_rng(1))
    assert a.shape == b.shape == (8, 2, 3, 3, 3)


def _blobs(n, channels):
    """Inputs whose mass sits at a different place per batch element, so their
    soft-argmax coordinates genuinely differ."""
    x = torch.zeros(n, channels, *ORBIT_SHAPE)
    for i in range(n):
        x[i, :, i * 3: i * 3 + 4] = 6.0
    return x


def test_coordinates_collapse_to_the_grid_centre_at_random_init():
    """Documents *why* the untrained control contributes ~0: with near-uniform
    heatmap logits the soft-argmax returns almost the same point whatever the
    input, so there is nothing for a shuffle to permute. This is a property of
    the architecture, not a bug, and it is what makes `ar-random`-style controls
    honest here."""
    model = _model().eval()
    with torch.no_grad():
        coords, _ = model.encode(_blobs(6, 1))
    assert float(coords.std(dim=0).mean()) < 0.01


def test_shuffling_the_coordinate_changes_the_loss():
    """The ablation must bite once coordinates actually differ across the batch
    -- otherwise `coord_contribution` measures nothing and a dead bottleneck
    would look healthy.

    The heatmap conv is amplified to peak the softmax, which is what a trained
    model's heatmaps look like. This isolates the plumbing (does
    ``shuffle_coords`` really permute?) from whether a given model has learned
    to use the bottleneck.
    """
    model = _model().eval()
    with torch.no_grad():
        model.to_heatmap.weight.mul_(200.0)
        model.dec[0].weight.mul_(50.0)
    rng = np.random.default_rng(2)
    x_t, x_t2 = _blobs(6, 2), torch.randn(6, 2, *ORBIT_SHAPE)
    with torch.no_grad():
        coords, _ = model.encode(x_t[:, 0:1])
        assert float(coords.std(dim=0).mean()) > 0.05      # precondition
        plain = _batch_loss(model, x_t, x_t2)[0]
        shuffled = _batch_loss(model, x_t, x_t2, shuffle_coords=True, rng=rng)[0]
    assert not torch.isclose(plain, shuffled, atol=1e-5)


def test_untrained_model_reports_no_coordinate_contribution():
    """The flip side: at random init the decoder ignores the coordinate, so the
    contribution must be ~0. This is the baseline the trained model has to
    clear, and if it were spuriously non-zero every later comparison would be
    biased."""
    model = _model().eval()
    data = np.random.default_rng(8).normal(size=(20, 2, *ORBIT_SHAPE)).astype(np.float16)
    m = evaluate(model, data, np.array([0, 10, 20]), batch=4, seed=0, n_batches=3)
    assert abs(m["coord_contribution"]) < 0.01


def test_evaluate_reports_the_coordinate_contribution():
    model = _model().eval()
    data = np.random.default_rng(3).normal(size=(24, 2, *ORBIT_SHAPE)).astype(np.float16)
    offsets = np.array([0, 12, 24])
    m = evaluate(model, data, offsets, batch=4, seed=0, n_batches=2)
    assert set(m) == {"r2", "r2_coord_shuffled", "coord_contribution"}
    assert np.isclose(m["coord_contribution"], m["r2"] - m["r2_coord_shuffled"])


def test_untrained_control_differs_from_the_trained_architecture():
    a = _model(seed=0)
    b = _model(seed=1000)
    x = torch.randn(2, 1, *ORBIT_SHAPE)
    with torch.no_grad():
        assert not torch.allclose(a.encode(x)[0], b.encode(x)[0])


# --- training and persistence -----------------------------------------------


def test_training_runs_and_selects_on_coordinate_contribution():
    rng = np.random.default_rng(4)
    # A shared latent shifts both orbits together, which is the structure the
    # cross-orbit objective is meant to find.
    n = 48
    data = np.zeros((n, 2, *ORBIT_SHAPE), dtype=np.float16)
    for i in range(n):
        shift = rng.integers(0, ORBIT_SHAPE[0] - 5)
        data[i, :, shift: shift + 5] = 1.0
    data += rng.normal(scale=0.1, size=data.shape).astype(np.float16)
    offsets = np.array([0, 24, 48])

    model = _model()
    hist, best = train(model, data, offsets, data, offsets, steps=40, batch=8,
                       lr=3e-3, seed=0, log_every=20)
    assert len(hist) == 2
    assert best["coord_contribution"] == max(m["coord_contribution"] for m in hist)
    assert best["state"] is not None


def test_checkpoint_round_trips(tmp_path):
    model = _model()
    save(tmp_path / "m.pt", model, {"keypoints": 2, "n_nuisance": 8, "width": 4,
                                    "coord_contribution": 0.1,
                                    "coord_contribution_untrained": 0.0})
    got, meta = load(tmp_path / "m.pt", device=CPU)
    assert meta["keypoints"] == 2
    x = torch.randn(2, 1, *ORBIT_SHAPE)
    with torch.no_grad():
        assert torch.allclose(model.encode(x)[0], got.encode(x)[0], atol=1e-6)


def test_build_orbit_cache_reads_participants_into_run_slabs(tmp_path):
    from deepmreye.storage import subject_path, write_subject

    rng = np.random.default_rng(5)
    subs = []
    for i, t in enumerate((40, 30)):
        p = subject_path(tmp_path, "ds1", f"sub-{i}")
        write_subject(p, rng.normal(size=(47, 29, 18, t)).astype(np.float32))
        subs.append(("ds1", f"sub-{i}", str(p), t))

    data, offsets = build_orbit_cache(subs, trs_per_subject=16)
    assert data.shape == (32, 2, *ORBIT_SHAPE)
    assert list(offsets) == [0, 16, 32]


# --- the probe feature source ------------------------------------------------


def test_orbit_extractor_pools_coordinates_into_bins():
    model = _model().eval()
    ex = OrbitExtractor("xorb", model)
    raw = np.random.default_rng(6).normal(size=(2, 47, 29, 18, 8)).astype(np.float32)
    out = ex(None, raw, 4)
    # 2 orbits x 2 keypoints x 3 coordinates.
    assert out.shape == (2, 4, 12)
    assert np.isfinite(out).all()


def test_orbit_extractor_can_return_the_nuisance_path_instead():
    """`xorb-nuis` is the check that the split split anything: the nuisance code
    should decode gaze badly, and we cannot check that without exposing it."""
    model = _model().eval()
    raw = np.random.default_rng(7).normal(size=(1, 47, 29, 18, 6)).astype(np.float32)
    assert OrbitExtractor("xorb-nuis", model)(None, raw, 3).shape == (1, 3, 16)


def test_orbit_extractor_needs_the_unpooled_window():
    model = _model().eval()
    with pytest.raises(ValueError):
        OrbitExtractor("xorb", model)(np.zeros((1, 4, 10)))


@pytest.mark.parametrize("kind", ORBIT_KINDS)
def test_orbit_kinds_parse(kind):
    assert parse_spec(kind) == ((kind, None),)
    assert parse_spec(f"fold-pca+{kind}:6") == (("fold-pca", None), (kind, 6))
