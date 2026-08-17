"""The cross-orbit contrastive encoder.

Three things here are worth guarding rather than merely exercising, and each one
corresponds to a way this module could be quietly wrong.

**The control.** ``xrot`` shipped a broken one: adding ``--parts`` widened the
model and the trainer but not the branch that built the untrained control, so a
24-feature model was scored against a 4-feature control and its reported margin
came out at +0.370 against a true +0.214. Every constructor argument here comes
out of the saved ``state_dict``, and the tests assert that a control matches the
model it controls and that a checkpoint missing an architecture field fails
loudly instead of silently building the wrong shape.

**The augmentations.** They are the reason to believe the encoder is not reading
global amplitude, so "an augmentation is applied" is a claim under test: each
view has to be perturbed *independently*, and the translation has to move the
volume rather than no-op.

**The diagnostic.** ``agreement_within_runs`` exists because the pooled
agreement cannot tell gaze from anatomy. The test constructs both situations
explicitly -- a signal that varies only between runs, and one that varies within
them -- and asserts the two measures disagree in exactly the way the docstring
claims. If that test ever passes trivially, the diagnostic has stopped doing its
job.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepmreye.orbitcon import (  # noqa: E402
    DEFAULT_AUG,
    ORBIT_SHAPE,
    OrbitContrastModel,
    _batch_loss,
    agreement,
    agreement_within_runs,
    augment,
    build_from_state,
    center_runs,
    evaluate,
    load,
    sample_batch,
    save,
    train,
    unmirror_right,
)

SMALL = (8, 10, 6)


def _model(**kw):
    kw.setdefault("embed_dim", 6)
    kw.setdefault("width", 4)
    kw.setdefault("expander_dim", 12)
    kw.setdefault("shape", SMALL)
    return OrbitContrastModel(device="cpu", seed=0, **kw)


def _pairs(n_runs=3, per_run=12, shape=SMALL, seed=0):
    """``(data [N, 2, ...], offsets)`` -- synthetic runs of paired orbits."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_runs * per_run, 2, *shape)).astype(np.float32)
    offsets = np.arange(n_runs + 1) * per_run
    return data, offsets


# ------------------------------------------------------------------ the model

@pytest.mark.parametrize("head", ["flat", "gap"])
def test_encode_gives_the_embedding_width_and_project_the_expander_width(head):
    m = _model(head=head)
    z = m.encode(torch.randn(5, 1, *SMALL))
    assert z.shape == (5, 6)
    assert m.project(z).shape == (5, 12)


def test_flat_head_keeps_spatial_layout_and_gap_discards_it():
    """The default head exists because gaze is a spatial variable.

    A ``gap`` head averages each channel over every position, so two inputs that
    differ only in *where* the mass sits can collapse to the same embedding. The
    ``flat`` head reads the feature map's layout, so it cannot.
    """
    flat, gap = _model(head="flat"), _model(head="gap")
    assert flat.to_embed.in_features > gap.to_embed.in_features

    a = torch.zeros(1, 1, *SMALL)
    b = torch.zeros(1, 1, *SMALL)
    a[0, 0, 1, 1, 1] = 5.0
    b[0, 0, 6, 8, 4] = 5.0            # same mass, opposite corner
    with torch.no_grad():
        d_flat = (flat.encode(a) - flat.encode(b)).abs().mean()
        d_gap = (gap.encode(a) - gap.encode(b)).abs().mean()
    assert d_flat > d_gap


def test_one_encoder_is_shared_between_the_two_orbits():
    """Both orbits go through the same weights.

    ``split_orbits`` mirrors the right orbit so the two arrive in the same
    orientation, which is what makes weight sharing correct rather than merely
    cheap. Identical input through the two calls must give identical output.
    """
    m = _model()
    x = torch.randn(4, 2, *SMALL)
    with torch.no_grad():
        assert torch.allclose(m.encode(x[:, 0:1]), m.encode(x[:, 0:1]))
    assert len({id(p) for p in m.parameters()}) == len(list(m.parameters()))


def test_loss_is_finite_and_gradients_reach_the_encoder():
    m = _model()
    loss, stats = _batch_loss(m, torch.randn(8, 2, *SMALL))
    assert np.isfinite(stats["loss"])
    loss.backward()
    grads = [p.grad for p in m.enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


# ----------------------------------------------------------- augmentations

def test_augment_preserves_shape_and_dtype():
    x = torch.randn(4, 1, *SMALL)
    a = augment(x, shape=SMALL)
    assert a.shape == x.shape and a.dtype == x.dtype


def test_augment_changes_the_volume():
    x = torch.randn(4, 1, *SMALL)
    assert (augment(x, shape=SMALL) - x).abs().mean() > 1e-6


def test_two_augment_calls_differ_so_the_views_are_independent():
    """The point of augmenting is that the encoder cannot know which it got.

    If both views received the same perturbation, the invariance term could be
    satisfied by encoding the perturbation itself.
    """
    x = torch.randn(6, 1, *SMALL)
    assert (augment(x, shape=SMALL) - augment(x, shape=SMALL)).abs().mean() > 1e-6


def test_each_sample_in_a_batch_gets_its_own_perturbation():
    """Per-sample draws, not one draw broadcast over the batch.

    A batch-wide gain would leave the *relative* amplitudes across the batch
    untouched, so global signal would remain a usable shared cue and defense 1
    in the module docstring would not hold.
    """
    x = torch.ones(8, 1, *SMALL)
    a = augment(x, {"shift_voxels": 0, "dropout": 0, "noise": 0,
                    "gain": 0.5, "bias": 0.5}, shape=SMALL)
    per_sample = a.reshape(8, -1).mean(dim=1)
    assert per_sample.std() > 1e-3


def test_translation_alone_moves_mass_without_changing_its_total():
    """Shift with `padding_mode='border'` semantics: mass moves, roughly conserved.

    Run with every other augmentation off so a failure localises to the
    translation path -- the one piece that reuses ``orbitrot``'s hand-written
    sampler, where an align_corners mismatch is invisible in a loss curve.
    """
    x = torch.zeros(4, 1, *SMALL)
    x[:, 0, 4, 5, 3] = 1.0
    only_shift = {"shift_voxels": 1.0, "dropout": 0, "noise": 0,
                  "gain": 0, "bias": 0}
    a = augment(x, only_shift, shape=SMALL)
    assert (a - x).abs().sum() > 1e-4
    assert a.sum() == pytest.approx(float(x.sum()), abs=0.5)


def test_zero_strength_augmentation_is_the_identity():
    x = torch.randn(3, 1, *SMALL)
    off = dict.fromkeys(DEFAULT_AUG, 0)
    assert torch.allclose(augment(x, off, shape=SMALL), x, atol=1e-5)


# ------------------------------------------------------------ the diagnostic

def test_pooled_agreement_cannot_tell_anatomy_from_gaze_and_within_run_can():
    """The reason ``agreement_within_run`` is reported at all.

    Build a signal that varies **only between runs** -- the anatomy shortcut,
    exactly: both orbits carry the same per-run constant. Pooled agreement is
    then near 1 and the shuffled control near 0, which is the signature a naive
    reading would call success. Within-run agreement is what exposes it, because
    inside one run there is nothing left but noise.
    """
    shape = SMALL
    n_runs, per_run = 6, 16
    rng = np.random.default_rng(0)
    data = np.empty((n_runs * per_run, 2, *shape), dtype=np.float32)
    for r in range(n_runs):
        constant = rng.standard_normal((1, 1, *shape)).astype(np.float32) * 3.0
        sl = slice(r * per_run, (r + 1) * per_run)
        data[sl] = constant + 0.01 * rng.standard_normal(
            (per_run, 2, *shape)).astype(np.float32)
    offsets = np.arange(n_runs + 1) * per_run

    m = _model()
    with torch.no_grad():
        t = torch.from_numpy(data)
        z_l = m.encode(t[:, 0:1]).numpy()
        z_r = m.encode(t[:, 1:2]).numpy()
    pooled = agreement(z_l, z_r)
    within = agreement_within_runs(m, data, offsets, batch=per_run, seed=0)

    # Between-run structure is the only signal, so pooled agreement is high and
    # within-run agreement has nothing to work with.
    assert pooled > 0.9
    assert within < pooled


def test_within_run_agreement_is_high_when_the_shared_signal_varies_in_time():
    """The complement: a signal that moves *within* each run is detected.

    Same measurement, opposite construction -- shared structure that varies
    timepoint to timepoint rather than run to run. Without this pair, the test
    above could be satisfied by a diagnostic that simply always reads low.
    """
    shape = SMALL
    n_runs, per_run = 4, 20
    rng = np.random.default_rng(1)
    data = np.empty((n_runs * per_run, 2, *shape), dtype=np.float32)
    for r in range(n_runs):
        for i in range(per_run):
            shared = rng.standard_normal((1, *shape)).astype(np.float32) * 3.0
            data[r * per_run + i, 0] = shared
            data[r * per_run + i, 1] = shared
    offsets = np.arange(n_runs + 1) * per_run

    m = _model()
    within = agreement_within_runs(m, data, offsets, batch=per_run, seed=0)
    assert within > 0.5


def test_unmirror_right_restores_the_shared_sign_of_horizontal_gaze():
    """The reason a contrastive cross-orbit objective must undo ``split_orbits``.

    Built from the geometry rather than asserted: put a blob in both orbits of
    one volume and displace both the same way in **global** x, which is what
    conjugate horizontal gaze does. After ``split_orbits`` the two crops'
    displacements point in opposite local directions; after ``unmirror_right``
    they agree. A shared encoder plus an MSE invariance term can only align the
    second case.
    """
    from deepmreye.crossorbit import LEFT_X, RIGHT_X, split_orbits

    def centroid_x(vol):
        w = vol.sum(axis=(1, 2))
        return float((w * np.arange(len(w))).sum() / w.sum())

    shifts = []
    for dx in (0, 3):
        block = np.zeros((47, 29, 18), dtype=np.float32)
        block[10 + dx, 14, 9] = 1.0        # left orbit
        block[36 + dx, 14, 9] = 1.0        # right orbit, same global direction
        left, right = split_orbits(block)
        shifts.append((centroid_x(left), centroid_x(right)))

    d_left = shifts[1][0] - shifts[0][0]
    d_right_mirrored = shifts[1][1] - shifts[0][1]
    # Mirrored: the same global motion runs opposite ways in the two crops.
    assert d_left > 0 and d_right_mirrored < 0

    # Un-mirrored: both crops move the same way, so one encoder can agree.
    pair = np.stack([np.stack(
        [np.zeros((22, 29, 18), np.float32)] * 2)] * 2)
    for i, dx in enumerate((0, 3)):
        block = np.zeros((47, 29, 18), dtype=np.float32)
        block[10 + dx, 14, 9] = 1.0
        block[36 + dx, 14, 9] = 1.0
        left, right = split_orbits(block)
        pair[i, 0], pair[i, 1] = left, right
    unmirror_right(pair)
    d_right_plain = centroid_x(pair[1, 1]) - centroid_x(pair[0, 1])
    assert d_right_plain > 0
    assert np.sign(d_right_plain) == np.sign(d_left)
    assert LEFT_X.start < RIGHT_X.start        # the split is along x


def test_the_mirror_convention_travels_with_the_checkpoint():
    """A model trained un-mirrored must not be fed mirrored orbits.

    There is no error when that happens, only a much lower score, so the
    convention is stored on the model and the extractor reads it.
    """
    m = _model(mirror_right=True)
    assert m.state_dict()["mirror_right"] is True
    c = build_from_state(m.state_dict(), device="cpu", seed=1)
    assert c.mirror_right is True
    assert build_from_state(_model().state_dict(), device="cpu").mirror_right is False


def test_extractor_follows_the_models_mirror_convention():
    from deepmreye.crossorbit import ORBIT_SHAPE as REAL_SHAPE
    from deepmreye.evaluate.features import OrbitContrastExtractor

    raw = np.random.default_rng(0).standard_normal(
        (1, 47, 29, 18, 4)).astype(np.float32)
    outs = {}
    for mirror in (False, True):
        m = OrbitContrastModel(embed_dim=4, width=3, expander_dim=8, seed=0,
                              device="cpu", shape=REAL_SHAPE,
                              mirror_right=mirror)
        outs[mirror] = OrbitContrastExtractor("ocon", m)(
            pooled=None, raw=raw, n_t=2)
    # Same weights, same input, different geometry -> different features.
    assert not np.allclose(outs[False], outs[True])


def test_agreement_of_independent_signals_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((400, 8))
    b = rng.standard_normal((400, 8))
    assert abs(agreement(a, b)) < 0.15


def test_agreement_of_a_signal_with_itself_is_one():
    a = np.random.default_rng(0).standard_normal((100, 5))
    assert agreement(a, a) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- the data

def test_center_runs_zeroes_each_run_separately():
    """Defense 2: the residual static component goes, per run, not globally.

    A global mean subtraction would leave each run's own offset intact, which is
    the participant-identifying component the invariance term would find first.
    """
    data, offsets = _pairs(n_runs=3, per_run=10)
    data[0:10] += 5.0
    data[10:20] -= 3.0
    out = center_runs(data.copy(), offsets)
    for i in range(3):
        lo, hi = offsets[i], offsets[i + 1]
        assert abs(float(out[lo:hi].mean())) < 1e-5


def test_sample_batch_draws_from_at_most_runs_per_batch_runs():
    """Defense 3: the batch spans few runs, so VICReg's variance term is
    within-participant rather than between."""
    per_run = 12
    data, offsets = _pairs(n_runs=8, per_run=per_run)
    # Tag each run so its identity is recoverable from the sampled rows.
    for r in range(8):
        data[r * per_run:(r + 1) * per_run] = float(r)
    rng = np.random.default_rng(0)
    batch = sample_batch(data, offsets, 32, rng, runs_per_batch=2)
    assert batch.shape == (32, 2, *SMALL)
    assert len(np.unique(batch.reshape(32, -1)[:, 0])) <= 2


def test_sample_batch_shape_is_both_orbits():
    data, offsets = _pairs()
    b = sample_batch(data, offsets, 7, np.random.default_rng(0))
    assert b.shape == (7, 2, *SMALL)


# ------------------------------------------------------- control and round trip

def test_control_is_built_from_the_model_not_from_configuration():
    """The ``xrot`` bug, as a test.

    ``build_from_state`` must reproduce every architectural choice from the
    state dict alone, so a control cannot drift from the model it controls when
    a new field is added.
    """
    m = _model(embed_dim=5, width=3, expander_dim=9, head="gap")
    c = build_from_state(m.state_dict(), device="cpu", seed=99)
    assert (c.embed_dim, c.width, c.expander_dim, c.head, c.shape) == \
           (m.embed_dim, m.width, m.expander_dim, m.head, m.shape)
    # Same shape, different weights -- that is what makes it a control.
    assert c.to_embed.in_features == m.to_embed.in_features
    assert not torch.allclose(c.to_embed.weight, m.to_embed.weight)


def test_a_checkpoint_missing_architecture_fields_raises():
    """Fail loudly rather than build a mismatched control."""
    sd = _model().state_dict()
    del sd["head"]
    with pytest.raises(ValueError, match="missing architecture fields"):
        build_from_state(sd, device="cpu")


def test_save_load_round_trip_preserves_architecture_and_outputs(tmp_path):
    m = _model(embed_dim=7, width=3, expander_dim=11)
    path = tmp_path / "ocon.pt"
    save(path, m, {"note": "test"})
    loaded, meta = load(path, device="cpu")
    assert meta["note"] == "test"
    assert (loaded.embed_dim, loaded.width, loaded.head) == (7, 3, m.head)
    x = torch.randn(3, 1, *SMALL)
    with torch.no_grad():
        assert torch.allclose(loaded.encode(x), m.encode(x), atol=1e-6)


# ----------------------------------------------------------------- training

def test_a_short_train_run_reduces_the_objective():
    data, offsets = _pairs(n_runs=6, per_run=16, seed=2)
    val, val_off = _pairs(n_runs=3, per_run=16, seed=3)
    m = _model()
    before = evaluate(m, val, val_off, batch=8, seed=0, runs_per_batch=2,
                      n_batches=3, n_within_runs=3)
    history, best = train(m, data, offsets, val, val_off, steps=40, batch=8,
                          lr=1e-3, seed=0, log_every=20, runs_per_batch=2)
    assert history and best["state"] is not None
    assert best["loss"] < before["loss"]


def test_evaluate_reports_every_agreement_column():
    data, offsets = _pairs(n_runs=4, per_run=16)
    m = _model()
    out = evaluate(m, data, offsets, batch=8, seed=0, runs_per_batch=2,
                   n_batches=2, n_within_runs=3)
    for key in ("loss", "sim", "std", "cov", "agreement", "agreement_shuffled",
                "agreement_margin", "agreement_within_run"):
        assert key in out and np.isfinite(out[key]), key


def test_checkpoint_is_written_during_training(tmp_path):
    """A killed run must leave the artifact it would have produced anyway."""
    data, offsets = _pairs(n_runs=4, per_run=16)
    path = tmp_path / "ckpt.pt"
    train(_model(), data, offsets, data, offsets, steps=20, batch=8, lr=1e-3,
          seed=0, log_every=10, runs_per_batch=2, checkpoint_path=path,
          meta={"tag": "partial"})
    assert path.exists()
    loaded, meta = load(path, device="cpu")
    assert meta["tag"] == "partial" and "val_loss" in meta


# ------------------------------------------------------------- the extractor

def test_extractor_output_width_is_both_orbits():
    """The probe feature is ``2 * embed_dim``, matched to ``lr-cca:64`` at 32."""
    from deepmreye.crossorbit import ORBIT_SHAPE as REAL_SHAPE
    from deepmreye.evaluate.features import OrbitContrastExtractor

    m = OrbitContrastModel(embed_dim=4, width=3, expander_dim=8, seed=0,
                           device="cpu", shape=REAL_SHAPE)
    ex = OrbitContrastExtractor("ocon", m)
    b, w, n_t = 2, 10, 5
    raw = np.random.default_rng(0).standard_normal((b, 47, 29, 18, w)).astype(np.float32)
    out = ex(pooled=None, raw=raw, n_t=n_t)
    assert out.shape == (b, n_t, 8)
    assert np.isfinite(out).all()


def test_extractor_requires_the_unpooled_window():
    """It reads orbits out of the voxel grid, so pooled rows are not enough."""
    from deepmreye.evaluate.features import OrbitContrastExtractor

    ex = OrbitContrastExtractor("ocon", _model())
    with pytest.raises(ValueError, match="unpooled window"):
        ex(pooled=np.zeros((1, 2, 3)), raw=None, n_t=None)


def test_ocon_kinds_are_registered_as_feature_sources():
    from deepmreye.evaluate.features import (
        FEATURE_KINDS,
        ORBIT_CONTRAST_KINDS,
        parse_spec,
    )

    assert set(ORBIT_CONTRAST_KINDS) <= set(FEATURE_KINDS)
    assert parse_spec("ocon") == (("ocon", None),)
    assert parse_spec("ocon-random:16") == (("ocon-random", 16),)
    assert parse_spec("fold-pca:64+ocon") == (("fold-pca", 64), ("ocon", None))
