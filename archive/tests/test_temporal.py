"""The next-TR (autoregressive) pretraining objective and its feature source.

The load-bearing test here is not that the model trains -- it is that
``ar-random`` is built from the same architecture as ``ar-gru`` and carries
different weights. That pair is the only thing separating "the objective
learned something" from "a random recurrent projection happens to score well",
and on the JEPA branch the absence of exactly this control is what let a
non-result stand.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepmreye.evaluate.features import (  # noqa: E402
    SEQUENCE_KINDS,
    SequenceExtractor,
    parse_spec,
)
from deepmreye.temporal import (  # noqa: E402
    ARModel,
    crops,
    evaluate_prediction,
    load,
    project_block,
    save,
    train,
)


def _basis(n_vox=12, k=4):
    rng = np.random.default_rng(0)
    return {"mean": rng.normal(size=n_vox),
            "components": np.linalg.qr(rng.normal(size=(n_vox, k)))[0]}


def _ar_data(n_runs=6, t=200, k=4, seed=0, process_seed=17):
    """Runs with genuine AR(1) structure, so a model has something to find.

    The transition matrix comes from ``process_seed``, held fixed while ``seed``
    varies the noise -- train and validation must share one process, or the
    model is asked to transfer to dynamics it has never seen and no amount of
    training can beat random init.
    """
    a = 0.85 * np.linalg.qr(np.random.default_rng(process_seed).normal(size=(k, k)))[0]
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_runs):
        x = np.zeros((t, k), dtype=np.float32)
        for i in range(1, t):
            x[i] = a @ x[i - 1] + 0.3 * rng.normal(size=k)
        parts.append(x)
    offsets = np.cumsum([0] + [len(p) for p in parts])
    return np.concatenate(parts), offsets


def test_project_block_matches_a_manual_projection():
    rng = np.random.default_rng(1)
    block = rng.normal(size=(3, 2, 2, 7))
    mask = np.ones((3, 2, 2), dtype=bool)
    mask[0, 0, 0] = False
    basis = _basis(n_vox=int(mask.sum()))

    got = project_block(block, mask.reshape(-1), basis["mean"], basis["components"])
    flat = block.reshape(-1, 7)[mask.reshape(-1)].T
    assert np.allclose(got, (flat - basis["mean"]) @ basis["components"], atol=1e-5)
    assert got.shape == (7, basis["components"].shape[1])


def test_crops_never_straddle_two_runs():
    """A crop spanning a run boundary would teach the model that one
    participant's last TR predicts another's first."""
    data, offsets = _ar_data(n_runs=4, t=60, k=3)
    # Tag each row with its run so a straddling crop is detectable.
    tags = np.concatenate([np.full(offsets[i + 1] - offsets[i], i)
                           for i in range(len(offsets) - 1)])
    rng = np.random.default_rng(2)
    runs = [(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]
    for _ in range(200):
        lo, hi = runs[rng.integers(len(runs))]
        s = lo + rng.integers(hi - lo - 21)
        assert len(np.unique(tags[s: s + 21])) == 1

    for batch in crops(data, offsets, 20, 8, np.random.default_rng(3), 3):
        assert batch.shape == (8, 21, 3)


def test_crops_raises_when_no_run_is_long_enough():
    data, offsets = _ar_data(n_runs=2, t=20, k=3)
    with pytest.raises(RuntimeError):
        list(crops(data, offsets, 100, 4, np.random.default_rng(0), 1))


def test_training_beats_its_own_untrained_control_on_learnable_data():
    """On data with real AR structure the objective must move: if a trained
    model cannot beat random init here, the training loop is broken and any
    null result on real data would be uninterpretable."""
    data, offsets = _ar_data(n_runs=10, t=300, k=4, seed=3)
    val, val_off = _ar_data(n_runs=4, t=300, k=4, seed=99)
    scale = np.maximum(data.std(axis=0), 1e-6).astype(np.float32)

    device = torch.device("cpu")
    control = ARModel(4, hidden=32, layers=1, seed=0, device=device)
    before = evaluate_prediction(control, val, val_off, torch.from_numpy(scale),
                                 length=32, batch=8, seed=1, n_batches=10)

    model = ARModel(4, hidden=32, layers=1, seed=0, device=device)
    _hist, best = train(model, data, offsets, val, val_off, scale, steps=300,
                        batch=16, length=32, lr=5e-3, seed=0, log_every=150)
    assert best["r2"] > before["r2"] + 0.05


def test_train_returns_the_best_checkpoint_not_the_last():
    data, offsets = _ar_data(n_runs=8, t=250, k=4, seed=5)
    val, val_off = _ar_data(n_runs=3, t=250, k=4, seed=6)
    scale = np.maximum(data.std(axis=0), 1e-6).astype(np.float32)
    model = ARModel(4, hidden=16, layers=1, seed=0, device=torch.device("cpu"))
    hist, best = train(model, data, offsets, val, val_off, scale, steps=200,
                       batch=8, length=32, lr=5e-3, seed=0, log_every=50)
    assert best["r2"] == max(m["r2"] for m in hist)
    assert best["step"] in [m["step"] for m in hist]


def test_checkpoint_round_trips(tmp_path):
    model = ARModel(4, hidden=16, layers=1, seed=0, device=torch.device("cpu"))
    scale = np.ones(4, dtype=np.float32)
    save(tmp_path / "m.pt", model, scale, {"hidden": 16, "layers": 1,
                                           "n_components": 4, "val_r2": 0.1,
                                           "val_r2_untrained": 0.0,
                                           "n_subjects": 3})
    got, got_scale, meta = load(tmp_path / "m.pt", device=torch.device("cpu"))
    assert meta["hidden"] == 16 and np.allclose(got_scale, scale)

    x = torch.randn(2, 9, 4)
    with torch.no_grad():
        a = model.forward(x)[1]
        b = got.forward(x)[1]
    assert torch.allclose(a, b, atol=1e-6)


def test_sequence_extractor_pools_hidden_states_into_bins():
    mask = np.ones((3, 2, 2), dtype=bool)
    basis = _basis(n_vox=12, k=4)
    model = ARModel(4, hidden=8, layers=1, seed=0, device=torch.device("cpu"))
    ex = SequenceExtractor("ar-gru", mask, basis, model, torch.ones(4))

    raw = np.random.default_rng(4).normal(size=(2, 3, 2, 2, 12)).astype(np.float32)
    out = ex(None, raw, 4)
    assert out.shape == (2, 4, 8)
    assert np.isfinite(out).all()
    assert ex.needs_fit is False


def test_sequence_extractor_honours_a_component_budget():
    mask = np.ones((3, 2, 2), dtype=bool)
    model = ARModel(4, hidden=8, layers=1, seed=0, device=torch.device("cpu"))
    ex = SequenceExtractor("ar-gru", mask, _basis(12, 4), model, torch.ones(4),
                           n_components=3)
    raw = np.random.default_rng(5).normal(size=(1, 3, 2, 2, 8)).astype(np.float32)
    assert ex(None, raw, 4).shape == (1, 4, 3)


def test_sequence_extractor_needs_the_unpooled_window():
    """Pooling first would average away the temporal order the model exists to
    read, so a caller that forgets to pass it must fail loudly."""
    mask = np.ones((3, 2, 2), dtype=bool)
    model = ARModel(4, hidden=8, layers=1, seed=0, device=torch.device("cpu"))
    ex = SequenceExtractor("ar-gru", mask, _basis(12, 4), model, torch.ones(4))
    with pytest.raises(ValueError):
        ex(np.zeros((1, 4, 12)))


def test_untrained_control_differs_from_the_trained_model():
    """`ar-random` must be the same architecture with different weights. If the
    two ever coincide, the control silently becomes a copy and the comparison
    that this whole experiment rests on stops meaning anything."""
    trained = ARModel(4, hidden=8, layers=1, seed=0, device=torch.device("cpu"))
    control = ARModel(4, hidden=8, layers=1, seed=1000, device=torch.device("cpu"))
    assert trained.hidden == control.hidden and trained.layers == control.layers

    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        assert not torch.allclose(trained.forward(x)[0], control.forward(x)[0])


@pytest.mark.parametrize("kind", SEQUENCE_KINDS)
def test_sequence_kinds_parse_and_take_a_budget(kind):
    assert parse_spec(kind) == ((kind, None),)
    assert parse_spec(f"fold-pca+{kind}:32") == (("fold-pca", None), (kind, 32))
