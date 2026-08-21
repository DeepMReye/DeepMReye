"""Tests for the voxel-level gaze network.

Two of these are regression tests for bugs that produced *plausible* numbers rather than
errors, which is the only reason they were caught at all:

- `test_gradient_reaches_the_learned_branch_at_init` -- initialising both the gate and the
  branch head at zero is a saddle (each gradient is proportional to the other), so nothing
  ever trains and the network reports an immaculate +0.0000 margin that reads as "the warm
  start guarantee is working".
- `test_cca_matrix_matches_orbit_projections` -- the warm start is only the incumbent if the
  collapsed `[V, k]` matrix really equals the two-orbit projection it replaces.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepmreye import voxelnet  # noqa: E402
from deepmreye.temporal_probe import make_lags  # noqa: E402


def _basis(v=40, k=4, seed=0):
    rng = np.random.default_rng(seed)
    li = np.arange(0, v // 2)
    ri = np.arange(v // 2, v)
    return {"left_index": li, "right_index": ri,
            "left_weights": rng.normal(size=(len(li), k + 2)),
            "right_weights": rng.normal(size=(len(ri), k + 2)),
            "mean": rng.normal(size=v)}


def test_cca_matrix_matches_orbit_projections():
    """`(x - mu) @ W` must equal the L/R average the incumbent's features are built from."""
    from deepmreye.orbitjepa import orbit_projections

    b = _basis()
    k = 4
    x = np.random.default_rng(1).normal(size=(17, 40))
    w, mu = voxelnet.cca_matrix(b, k=k)
    got = (x - mu) @ w
    zl, zr = orbit_projections(x, b, m=k)
    want = 0.5 * (zl[:, :k] + zr[:, :k])
    assert np.allclose(got, want, atol=1e-10)


def test_make_lags_torch_matches_numpy():
    z = np.random.default_rng(2).normal(size=(23, 5))
    for lags in (0, 1, 2, 4):
        got = voxelnet.make_lags_torch(torch.as_tensor(z)[None], lags).numpy()[0]
        assert np.allclose(got, make_lags(z, lags))


def _net(k=4, v=40, lags=1, rank=3):
    b = _basis(v=v, k=k)
    w, mu = voxelnet.cca_matrix(b, k=k)
    return voxelnet.build_net(w, mu, lags=lags, encoder="lowrank", rank=rank, dropout=0.0)


def test_untrained_net_is_exactly_the_linear_branch():
    """`head_nl` is zero at init, so the network equals its own linear branch."""
    net = _net()
    x = torch.as_tensor(np.random.default_rng(3).normal(size=(1, 12, 40)), dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        assert torch.allclose(net(x), net.linear_only(x), atol=1e-6)


def test_gradient_reaches_the_learned_branch_at_init():
    """Regression test for the saddle: zeroing the gate AND the head trains nothing.

    With `alpha = 0` and `head_nl = 0`, d(loss)/d(alpha) is proportional to the branch output
    (zero) and d(loss)/d(head_nl) is proportional to alpha (zero), so the model is stuck at
    initialisation while looking exactly like a successful warm start.
    """
    net = _net()
    x = torch.as_tensor(np.random.default_rng(4).normal(size=(1, 12, 40)), dtype=torch.float32)
    target = torch.as_tensor(np.random.default_rng(5).normal(size=(1, 12, 20)),
                             dtype=torch.float32)
    torch.nn.functional.mse_loss(net(x), target).backward()
    assert net.head_nl.weight.grad is not None
    assert float(net.head_nl.weight.grad.abs().max()) > 0, \
        "no gradient reaches the learned branch -- the network cannot train"


def test_warm_start_reproduces_the_ridge_fit():
    from sklearn.linear_model import RidgeCV

    b = _basis()
    k, lags = 4, 1
    w, mu = voxelnet.cca_matrix(b, k=k)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 40))
    z = (x - mu) @ w
    y = make_lags(z, lags) @ rng.normal(size=(k * (2 * lags + 1), 20))
    ridge = RidgeCV(alphas=np.logspace(-2, 2, 5)).fit(make_lags(z, lags), y)

    net = voxelnet.build_net(w, mu, lags=lags, encoder="lowrank", rank=3, dropout=0.0)
    voxelnet.warm_start(net, ridge)
    err = voxelnet.assert_warm_start(net, ridge, x[:50], lags)
    assert err < 1e-4


def test_assert_warm_start_rejects_a_mismatched_head():
    from sklearn.linear_model import RidgeCV

    b = _basis()
    k, lags = 4, 1
    w, mu = voxelnet.cca_matrix(b, k=k)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 40))
    z = (x - mu) @ w
    y = rng.normal(size=(120, 20))
    ridge = RidgeCV(alphas=np.logspace(-2, 2, 5)).fit(make_lags(z, lags), y)
    net = voxelnet.build_net(w, mu, lags=lags, encoder="lowrank", rank=3, dropout=0.0)
    # deliberately NOT warm-started
    with pytest.raises(AssertionError):
        voxelnet.assert_warm_start(net, ridge, x[:40], lags)


def test_shift_augment_is_a_permutation_of_the_grid():
    """A roll moves values; it must not create or destroy any."""
    grid_shape = (4, 5, 3)
    n_grid = int(np.prod(grid_shape))
    mask_idx = torch.as_tensor(np.sort(
        np.random.default_rng(8).choice(n_grid, 30, replace=False)), dtype=torch.long)
    x = torch.as_tensor(np.random.default_rng(9).normal(size=(2, 6, 30)), dtype=torch.float32)
    out = voxelnet.shift_augment(x, mask_idx, grid_shape, 1, np.random.default_rng(10))
    assert out.shape == x.shape
    # Values can leave the mask, but nothing new may appear.
    kept = set(np.round(out.numpy().ravel(), 5)) - {0.0}
    orig = set(np.round(x.numpy().ravel(), 5))
    assert kept <= orig


def test_shift_augment_zero_is_identity():
    grid_shape = (4, 5, 3)
    mask_idx = torch.arange(20, dtype=torch.long)
    x = torch.as_tensor(np.random.default_rng(11).normal(size=(2, 3, 20)), dtype=torch.float32)
    out = voxelnet.shift_augment(x, mask_idx, grid_shape, 0, np.random.default_rng(12))
    assert torch.allclose(out, x)


def test_mixup_is_label_consistent_for_a_linear_generator():
    """Gaze is near-linear in these features, so a mixed input has the mixed target."""
    rng = np.random.default_rng(13)
    w = torch.as_tensor(rng.normal(size=(8, 20)), dtype=torch.float32)
    x = torch.as_tensor(rng.normal(size=(4, 6, 8)), dtype=torch.float32)
    y = x @ w
    xm, ym = voxelnet.mixup(x, y, rng, alpha=0.5)
    assert torch.allclose(xm @ w, ym, atol=1e-4)


def test_mixup_zero_alpha_is_identity():
    rng = np.random.default_rng(14)
    x = torch.as_tensor(rng.normal(size=(3, 4, 5)), dtype=torch.float32)
    y = torch.as_tensor(rng.normal(size=(3, 4, 20)), dtype=torch.float32)
    xm, ym = voxelnet.mixup(x, y, rng, alpha=0.0)
    assert torch.allclose(xm, x) and torch.allclose(ym, y)


def test_shift_augment_per_sample_gives_each_chunk_its_own_shift():
    """One shift per batch is one augmentation per optimizer step, which is nearly none.

    The batched form draws a single roll and applies it to every chunk, so two identical
    chunks stay identical after augmentation. `per_sample=True` must break that tie, or the
    augmentation is not doing the job its flag advertises.
    """
    import torch

    grid_shape = (4, 4, 4)
    mask_idx = np.arange(64)
    x = torch.arange(2 * 3 * 64, dtype=torch.float32).reshape(2, 3, 64)
    x[1] = x[0]                                   # two identical chunks

    same = voxelnet.shift_augment(x, torch.as_tensor(mask_idx), grid_shape, 2,
                                  np.random.default_rng(0))
    assert torch.allclose(same[0], same[1]), "batched mode must move both chunks together"

    apart = voxelnet.shift_augment(x, torch.as_tensor(mask_idx), grid_shape, 2,
                                   np.random.default_rng(0), per_sample=True)
    assert not torch.allclose(apart[0], apart[1]), "per-sample mode must decorrelate chunks"
    # Still a permutation of the same voxels, per chunk.
    for i in range(2):
        assert torch.allclose(apart[i].sort(dim=-1).values, x[i].sort(dim=-1).values)
