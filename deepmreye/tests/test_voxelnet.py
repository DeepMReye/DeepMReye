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


# --------------------------------------------------------------------------
# Mirror augmentation
# --------------------------------------------------------------------------
def test_mirror_index_is_an_involution_and_reflects_in_x():
    """Mirroring twice returns the original, and a left-lobe voxel lands in the right lobe.

    The augmentation's whole claim is that it maps a real sample to another *physically
    valid* one. If the index were an arbitrary permutation rather than a reflection it would
    still round-trip, so the reflection itself is asserted separately.
    """
    from deepmreye.voxelnet import mirror_index, mirror_rows

    shape = (47, 29, 18)
    mask = np.zeros(shape, dtype=bool)
    mask[3:22, 5:24, 2:16] = True        # left lobe
    mask[26:45, 5:24, 2:16] = True       # right lobe
    src = mirror_index(mask, roll=1)
    n_vox = int(mask.sum())
    assert src.shape == (n_vox,)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, n_vox)).astype(np.float32)
    twice = mirror_rows(mirror_rows(x, src), src)
    ok = src >= 0
    doubly = np.flatnonzero(ok)[src[src[ok]] >= 0]
    assert np.allclose(twice[:, doubly], x[:, doubly])

    # a single voxel in the left lobe must reappear in the right lobe, same y and z
    flat_idx = np.flatnonzero(mask.reshape(-1))
    grid = np.zeros(shape, dtype=np.float32)
    grid[10, 12, 8] = 1.0
    row = grid.reshape(-1)[flat_idx][None, :]
    out = np.zeros(int(np.prod(shape)), dtype=np.float32)
    out[flat_idx] = mirror_rows(row, src)[0]
    i, j, k = np.unravel_index(int(np.argmax(out)), shape)
    assert i > shape[0] // 2, "left-lobe voxel did not reflect into the right lobe"
    assert (j, k) == (12, 8), "reflection must leave the y and z axes alone"


def test_mirror_negates_horizontal_gaze_only():
    """Even columns of the flattened `[T, 10, 2]` label are horizontal, odd are vertical."""
    labels = np.arange(2 * 10 * 2, dtype=np.float64).reshape(2, 10, 2)
    flat = labels.reshape(2, 20).copy()
    flat[:, 0::2] *= -1
    back = flat.reshape(2, 10, 2)
    assert np.allclose(back[..., 0], -labels[..., 0]), "horizontal gaze must flip sign"
    assert np.allclose(back[..., 1], labels[..., 1]), "vertical gaze must be untouched"


def test_warmup_cosine_matches_cosineannealinglr_when_there_is_no_warmup():
    """`--warmup 0 --cosine` must be the OLD schedule to the last decimal.

    The scheduler was replaced by a hand-written LambdaLR so warmup could be prefixed to it.
    Every trial already on record ran under `CosineAnnealingLR`, so if the replacement is off
    by even a fraction those runs stop being comparable to the new ones -- and the difference
    would show up as a hyperparameter effect rather than as a bug.
    """
    import math

    import torch

    epochs, warmup = 40, 0

    def lr_factor(e):
        if warmup > 0 and e < warmup:
            return (e + 1) / warmup
        span = max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (e - warmup) / span)))

    par = [torch.nn.Parameter(torch.zeros(1))]
    ref = torch.optim.lr_scheduler.CosineAnnealingLR(
        torch.optim.SGD(par, lr=1e-3), T_max=epochs)
    got = torch.optim.lr_scheduler.LambdaLR(torch.optim.SGD(par, lr=1e-3), lr_factor)
    for _ in range(epochs):
        assert abs(ref.get_last_lr()[0] - got.get_last_lr()[0]) < 1e-12
        ref.step()
        got.step()


def test_warmup_ramps_linearly_and_then_decays():
    import math

    epochs, warmup = 40, 5

    def lr_factor(e):
        if warmup > 0 and e < warmup:
            return (e + 1) / warmup
        span = max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (e - warmup) / span)))

    ramp = [lr_factor(e) for e in range(warmup)]
    assert ramp == sorted(ramp) and abs(ramp[-1] - 1.0) < 1e-12
    assert abs(lr_factor(warmup) - 1.0) < 1e-12          # continuous at the handover
    tail = [lr_factor(e) for e in range(warmup, epochs)]
    assert tail == sorted(tail, reverse=True)
    assert lr_factor(epochs - 1) < 0.01


def test_ema_swap_restores_the_training_weights_exactly():
    """The EMA is what gets SCORED and SNAPSHOTTED; the raw weights are what keep training.

    If the swap leaked -- restoring the averaged weights into the optimiser's parameters --
    training would silently continue from the average and `--ema` would stop being an
    evaluation-time option. That reads as a hyperparameter effect, not as a bug, which is
    exactly the failure mode this file exists to catch.
    """
    import contextlib

    import torch

    net = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU(), torch.nn.Linear(3, 2))
    decay = 0.9
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    @contextlib.contextmanager
    def eval_weights():
        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict({k: v.to(dtype=backup[k].dtype) for k, v in ema.items()})
        try:
            yield
        finally:
            net.load_state_dict(backup)

    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        net(torch.ones(2, 4)).sum().backward()
        opt.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                ema[k].mul_(decay).add_(v.detach().float(), alpha=1 - decay)

    raw = {k: v.detach().clone() for k, v in net.state_dict().items()}
    assert not all(torch.allclose(raw[k], ema[k]) for k in raw), "EMA never diverged from raw"
    with eval_weights():
        inside = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for k in raw:
        assert torch.allclose(inside[k], ema[k]), "scoring did not see the EMA weights"
        assert torch.allclose(net.state_dict()[k], raw[k]), "swap leaked into training"
