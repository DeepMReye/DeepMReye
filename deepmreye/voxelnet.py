"""Voxel-level gaze network, warm-started at the linear incumbent.

The argument for this arm, stated so it can be checked rather than believed: a network is a
universal function approximator, so there **exists** a network computing
`RidgeHead(make_lags(W_cca @ (x - mu), L))` exactly -- `W_cca` is a linear layer, `make_lags`
is a fixed convolution, the ridge head is a linear layer. Universal approximation is an
existence claim about representation, and says nothing about whether SGD finds that point or
whether a richer function class generalises from 337 participants. This module makes the
existence claim **constructive**: the network is *initialised* at the incumbent, so it starts
at the incumbent's score by construction and training can only move away from a known-good
point, with early stopping on held-out participants to keep the better of the two.

That is also why this is not refuted by `analyze_temporal_ceiling_supervised.py`. That gate
measured readouts over **frozen, unsupervised** projections -- `lr-cca` fitted blind to gaze,
at ranks 32..256 -- and found more of them monotonically harmful. It never measured a
**supervised learned** projection from voxels, which optimises the exact quantity scored. The
learned branch here is precisely that missing arm.

    pred(x) = head_lin(make_lags(z_cca(x), L))        # == the incumbent, warm-started
            + clamp(alpha) * head_nl(make_lags(g(x), L))   # zero-init; must earn its place

`g` is the learned encoder: a low-rank linear map over voxels (directly comparable to
`lr-cca:32`) or a 3-D conv over the eye block. `alpha` is ReZero-style, initialised at 0.
"""
import json
from pathlib import Path

import numpy as np

CACHE_VERSION = 1


# --------------------------------------------------------------------------------------
# Voxel cache: one fp16 memmap over every labeled TR, plus a per-participant index.
#
# 337 participants x ~406k TRs x 14236 voxels is 11.6 GB at fp16 -- too large to hold as
# fp32 in RAM alongside a trainer, small enough to memmap and slice. Kept masked (not on the
# [47,29,18] grid) because the grid is 42% zeros; `scatter_grid` puts it back when the conv
# needs it, which costs less than storing the zeros.
# --------------------------------------------------------------------------------------

def build_voxel_cache(root, mask, out_dir, dtype=np.float16, verbose=True, labeled=True,
                      max_participants=None):
    """Write `voxels.npy` (memmap `[N, V]`), `labels.npy` (`[N, 10, 2]`) and `index.json`.

    `labeled=False` builds the **unlabeled** corpus instead -- every participant *without* a
    `labels` dataset, under its own accession. Those are strictly disjoint from the probe set
    by construction, which is the property any pretraining cache has to have: a `dsL*` file
    leaking in would put the same participant on both sides of a leave-one-dataset-out split.
    `labels.npy` is written all-NaN there so the two caches share a loader.
    """
    import h5py

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = mask.reshape(-1)
    n_vox = int(flat.sum())

    if labeled:
        dirs = sorted(p for p in Path(root).glob("dsL*") if p.is_dir())
    else:
        dirs = sorted(p for p in Path(root).iterdir()
                      if p.is_dir() and not p.name.startswith(("dsL", "_", ".")))

    parts = []
    total = 0
    for ds_dir in dirs:
        for path in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(path, "r") as f:
                    has = "labels" in f
                    if has != labeled:
                        continue
                    t = f["eye_block"].shape[-1]
                    if t < 60:
                        continue
                    if labeled and not np.isfinite(f["labels"][:]).any():
                        continue
            except Exception:
                continue
            parts.append({"dataset": ds_dir.name, "subject": path.stem,
                          "path": str(path), "start": total, "n": t})
            total += t
            if max_participants and len(parts) >= max_participants:
                break
        if max_participants and len(parts) >= max_participants:
            break
    if not parts:
        raise SystemExit("[!] no participants found")
    if not labeled:
        assert not any(p_["dataset"].startswith("dsL") for p_ in parts), \
            "a labeled dataset leaked into the pretraining cache"

    vox = np.lib.format.open_memmap(out_dir / "voxels.npy", mode="w+",
                                    dtype=dtype, shape=(total, n_vox))
    lab = np.lib.format.open_memmap(out_dir / "labels.npy", mode="w+",
                                    dtype=np.float32, shape=(total, 10, 2))
    for i, p in enumerate(parts):
        with h5py.File(p["path"], "r") as f:
            block = f["eye_block"][:]
            labels = f["labels"][:] if labeled else None
        t = p["n"]
        vox[p["start"]:p["start"] + t] = block.reshape(-1, t).T[:, flat].astype(dtype)
        lab[p["start"]:p["start"] + t] = (labels[:t].astype(np.float32) if labeled
                                          else np.nan)
        if verbose and (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(parts)} participants", flush=True)
    vox.flush()
    lab.flush()

    meta = {"version": CACHE_VERSION, "n_rows": total, "n_vox": n_vox,
            "parts": [{k: p[k] for k in ("dataset", "subject", "start", "n")} for p in parts],
            "mask_sum": int(flat.sum()), "mask_shape": list(mask.shape)}
    (out_dir / "index.json").write_text(json.dumps(meta))
    if verbose:
        print(f"[+] {len(parts)} participants, {total} TRs, "
              f"{vox.nbytes / 1e9:.1f} GB -> {out_dir}", flush=True)
    return meta


def load_voxel_cache(out_dir, mask):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "index.json").read_text())
    if meta["version"] != CACHE_VERSION:
        raise SystemExit(f"[!] voxel cache version {meta['version']} != {CACHE_VERSION}")
    if meta["mask_sum"] != int(mask.sum()) or meta["mask_shape"] != list(mask.shape):
        raise SystemExit("[!] voxel cache built against a different mask; rebuild")
    vox = np.load(out_dir / "voxels.npy", mmap_mode="r")
    lab = np.load(out_dir / "labels.npy", mmap_mode="r")
    return vox, lab, meta


# --------------------------------------------------------------------------------------
# The frozen linear branch, as one matrix
# --------------------------------------------------------------------------------------

def cca_matrix(basis, k=32):
    """`(W, mu)` such that `(x - mu) @ W` equals `0.5 * (z_left[:k] + z_right[:k])`.

    `orbit_projections` centres, splits the two orbit index sets, projects each with its own
    weights and the caller averages. All of that is linear in `x`, so it collapses to one
    `[V, k]` matrix -- which is what lets the incumbent live inside a network as a single
    frozen layer, and is the whole basis of the warm start.
    """
    li = np.asarray(basis["left_index"])
    ri = np.asarray(basis["right_index"])
    wl = np.asarray(basis["left_weights"])[:, :k]
    wr = np.asarray(basis["right_weights"])[:, :k]
    mu = np.asarray(basis["mean"])
    w = np.zeros((mu.shape[0], k), dtype=np.float64)
    w[li] = 0.5 * wl
    w[ri] = 0.5 * wr
    return w, mu


def make_lags_torch(z, lags):
    """Torch twin of `temporal_probe.make_lags`: block `l` at row `t` is `z[clip(t+l,0,T-1)]`.

    Index-and-clamp rather than `F.pad` + `Conv1d`: `nn.Conv1d` defaults to **zero** padding
    while `make_lags` edge-pads, and torch cross-correlates rather than convolving, so a conv
    written to imitate this differs from the incumbent only on the first and last `lags` rows
    of every participant -- invisible in every downstream number. Parity is asserted in the
    tests against the numpy implementation.
    """
    import torch

    if lags == 0:
        return z
    t_n = z.shape[1]
    base = torch.arange(t_n, device=z.device)
    return torch.cat([z.index_select(1, (base + lag).clamp(0, t_n - 1))
                      for lag in range(-lags, lags + 1)], dim=-1)


def _module():
    import torch
    import torch.nn as nn
    torch.set_num_threads(max(1, min(8, __import__("os").cpu_count() or 1)))
    return torch, nn


class LowRankEncoder:
    """Marker for the supervised learned projection: `Linear(V, rank, bias=False)`.

    Directly comparable to `lr-cca:32` and the arm the temporal gate never tested -- that gate
    varied the rank of a *frozen, unsupervised* projection, which is a different object from a
    projection fitted against gaze.
    """


def build_net(w_cca, mu, lags=1, encoder="lowrank", rank=32, width=16, dropout=0.2,
              grid_shape=(47, 29, 18), mask_idx=None, seed=0):
    torch, nn = _module()
    torch.manual_seed(seed)

    class VoxelGazeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lags = int(lags)
            self.encoder_kind = encoder
            self.rank = int(rank)
            n_lag = 2 * self.lags + 1
            k = w_cca.shape[1]

            self.register_buffer("w_cca", torch.as_tensor(w_cca, dtype=torch.float32))
            self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
            # The incumbent. Warm-started from the fold's own RidgeCV fit; never randomised.
            self.head_lin = nn.Linear(k * n_lag, 20)

            if encoder == "lowrank":
                self.enc = nn.Linear(w_cca.shape[0], self.rank, bias=False)
                nn.init.normal_(self.enc.weight, std=1e-3)
                enc_out = self.rank
            elif encoder == "cnn":
                if mask_idx is None:
                    raise ValueError("cnn encoder needs mask_idx to scatter onto the grid")
                self.register_buffer("mask_idx", torch.as_tensor(mask_idx, dtype=torch.long))
                self.grid_shape = tuple(grid_shape)
                # NO global average pooling. Gaze is *where* the eyeball is, so pooling the
                # spatial axes away destroys the signal the branch exists to find: with
                # `AdaptiveAvgPool3d(1)` the training loss was flat (0.464 / 0.490 / 0.465)
                # and selection r never moved off the incumbent's 0.8265. DeepMReye 1.0
                # flattens its feature map into dense layers for the same reason. Three
                # stride-2 convs take [47,29,18] to [6,4,3], so the flattened map keeps 72
                # spatial positions per channel.
                self.conv = nn.Sequential(
                    nn.Conv3d(1, width, 3, stride=2, padding=1), nn.GroupNorm(4, width), nn.GELU(),
                    nn.Conv3d(width, 2 * width, 3, stride=2, padding=1),
                    nn.GroupNorm(4, 2 * width), nn.GELU(),
                    nn.Conv3d(2 * width, 4 * width, 3, stride=2, padding=1),
                    nn.GroupNorm(4, 4 * width), nn.GELU(),
                    nn.Flatten())
                with torch.no_grad():
                    probe = torch.zeros(1, 1, *tuple(grid_shape))
                    n_feat = self.conv(probe).shape[1]
                self.enc = nn.Linear(n_feat, self.rank)
                enc_out = self.rank
            else:
                raise ValueError(encoder)

            self.drop = nn.Dropout(dropout)
            # Zero-init the last layer ONLY, and start the gate at 1. Zeroing *both* -- the
            # obvious reading of "the branch must start switched off" -- is a saddle: the
            # gradient to `alpha` is proportional to the branch output and the gradient to
            # the branch is proportional to `alpha`, so with both at zero neither ever moves
            # and the net sits at the incumbent forever, reporting a perfect +0.0000 margin
            # that looks like a result. Measured: `alpha` pinned at 0.0000 with val loss
            # identical across epochs. With `head_nl` zero and `alpha` one, the prediction is
            # still exactly the incumbent at step 0 (the branch outputs zero) while gradient
            # reaches `head_nl` immediately, and the encoder one step later.
            self.head_nl = nn.Linear(enc_out * n_lag, 20)
            nn.init.zeros_(self.head_nl.weight)
            nn.init.zeros_(self.head_nl.bias)
            self.alpha = nn.Parameter(torch.ones(1))

        def encode(self, x):
            if self.encoder_kind == "lowrank":
                return self.enc(x)
            b, t_n, _ = x.shape
            grid = x.new_zeros(b * t_n, int(np.prod(self.grid_shape)))
            grid[:, self.mask_idx] = x.reshape(b * t_n, -1)
            grid = grid.view(b * t_n, 1, *self.grid_shape)
            return self.enc(self.conv(grid)).view(b, t_n, -1)

        def forward(self, x):
            z = (x - self.mu) @ self.w_cca
            y_lin = self.head_lin(make_lags_torch(z, self.lags))
            h = self.drop(self.encode(x))
            y_nl = self.head_nl(make_lags_torch(h, self.lags))
            return y_lin + self.alpha.clamp(-1.0, 1.0) * y_nl

        def linear_only(self, x):
            z = (x - self.mu) @ self.w_cca
            return self.head_lin(make_lags_torch(z, self.lags))

        def nonlinear_share(self):
            return float(self.alpha.detach().abs().clamp(max=1.0))

        def arch(self):
            return {"lags": self.lags, "encoder": self.encoder_kind, "rank": self.rank,
                    "dropout": float(dropout), "width": int(width), "seed": int(seed)}

    return VoxelGazeNet()


def warm_start(net, ridge):
    """Copy a fitted RidgeCV into the linear branch, making the net *equal* the incumbent.

    Not a unit test but a per-fold precondition: `assert_warm_start` re-checks it on real
    held-out rows every fold. The `xrot` incident reported a +0.370 margin for a true +0.214
    because a control was assembled from configuration rather than from the artifact.
    """
    import torch

    with torch.no_grad():
        net.head_lin.weight.copy_(torch.as_tensor(ridge.coef_, dtype=torch.float32))
        net.head_lin.bias.copy_(torch.as_tensor(ridge.intercept_, dtype=torch.float32))
        net.alpha.zero_()
    return net


def assert_warm_start(net, ridge, x, lags, tol=1e-4):
    """`net(x)` must equal `ridge.predict(make_lags(z_cca(x), L))` at init."""
    import torch

    from deepmreye.temporal_probe import make_lags

    net.eval()
    with torch.no_grad():
        got = net(torch.as_tensor(x, dtype=torch.float32)[None]).numpy()[0]
    z = (np.asarray(x, dtype=np.float64) - net.mu.numpy()) @ net.w_cca.numpy()
    want = ridge.predict(make_lags(z, lags))
    err = float(np.abs(got - want).max())
    if err > tol:
        raise AssertionError(f"warm start is not the incumbent: max|diff| = {err:.2e}")
    return err


# --------------------------------------------------------------------------------------
# Voxel-space augmentation
#
# The learned branch's failure mode is measured, not guessed: its selection score on a
# held-out *dataset* falls monotonically from the first step (dsL08: 0.365 -> 0.208 -> 0.161)
# while its training loss improves. It is fitting structure that does not cross a dataset
# boundary -- registration, anatomy, protocol. Geometric augmentation is the direct attack on
# that, and it is what DeepMReye 1.0 used (rotation +-5deg, shift +-4 voxels, zoom 0.15) on
# this exact signal.
#
# Integer-voxel shifts are a re-index of the grid, so they cost a gather rather than an
# interpolation and can run per batch on the GPU. Rotation and zoom need trilinear resampling
# and are deliberately left out here rather than approximated badly.
#
# What must NOT be used: any flip. An x-flip negates horizontal gaze while leaving the label
# untouched, which teaches the model that the two are unrelated -- the same trap
# `orbitcon.unmirror_right` exists for.
# --------------------------------------------------------------------------------------

def shift_augment(x, mask_idx, grid_shape, max_shift, gen, per_sample=False):
    """Random integer-voxel translation of each sample, as a grid roll.

    `per_sample=False` draws ONE shift for the whole batch, which at `--batch-chunks 2` is
    one augmentation per optimizer step -- far weaker than DeepMReye 1.0's per-sample jitter,
    and weak enough that turning it on barely changes the input distribution. `per_sample=True`
    draws a shift per chunk instead.

    The shift is per *chunk*, never per TR: a chunk is a contiguous time window from one
    participant and the lag stack reads across it, so jittering each TR independently would
    corrupt the temporal structure the model is being given. One shift per chunk is the
    registration-jitter interpretation, which is the one that leaves gaze unchanged.
    """
    import torch

    if max_shift <= 0:
        return x
    b, t_n, _ = x.shape
    n_grid = int(np.prod(grid_shape))
    grid = x.new_zeros(b * t_n, n_grid)
    grid[:, mask_idx] = x.reshape(b * t_n, -1)
    grid = grid.view(b, t_n, *grid_shape)
    if per_sample:
        rolled = []
        for i in range(b):
            sh = tuple(int(gen.integers(-max_shift, max_shift + 1)) for _ in range(3))
            rolled.append(torch.roll(grid[i], shifts=sh, dims=(1, 2, 3)))
        grid = torch.stack(rolled)
    else:
        shifts = tuple(int(gen.integers(-max_shift, max_shift + 1)) for _ in range(3))
        grid = torch.roll(grid, shifts=shifts, dims=(2, 3, 4))
    return grid.reshape(b * t_n, n_grid)[:, mask_idx].view(b, t_n, -1)


def mixup(x, y, gen, alpha=0.2):
    """Convex combination of two samples, applied to voxels and targets alike.

    Label-consistent here in a way it is not for classification: gaze is very nearly a linear
    function of these voxels, so a mixed input has the mixed target almost exactly. Its job is
    to stop the branch keying on any single participant's anatomy.
    """
    if alpha <= 0 or len(x) < 2:
        return x, y
    lam = float(gen.beta(alpha, alpha))
    perm = list(range(1, len(x))) + [0]
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
