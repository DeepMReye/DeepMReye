"""Cross-orbit prediction through a soft-argmax bottleneck.

The design follows from what the previous two attempts measured rather than
from a new architecture family.

``temporal.py`` failed for a specific, diagnosed reason: a next-TR model has
*one* hidden state, the predictable part of an eye block is drift/motion/global
signal, and so the nuisance evicted gaze (trained 0.530 against 0.686 for its
own untrained control). The fix that follows is not a better predictor -- it is
**somewhere else for the nuisance to go**.

So there are two paths, and each is blocked from carrying what the other owns:

- **the coordinate path** is a soft-argmax bottleneck: the encoder emits ``K``
  heatmaps per orbit, each is softmaxed over the volume and collapsed to its
  spatial expectation. The output is ``K x 3`` numbers that *are* positions by
  construction rather than by hope. This is the "tight bottleneck" of the
  unsupervised-landmark literature (Jakab et al. 2018), and it fits the physics:
  an eyeball rotating in its socket moves mass, so a centroid is the natural
  latent. At ``K=2`` it is 6 numbers per orbit -- far too few to smuggle
  appearance through.

- **the nuisance path** is a wide global vector, and it is encoded from a
  **different TR of the same run**. Anatomy, coil bias, subject and scanner are
  constant across TRs and pass through freely; *this* TR's gaze cannot, because
  the path never sees this TR. That is DrNet's time-invariance trick (Denton &
  Birodkar 2017) and it avoids needing an adversarial loss.

The objective is **cross-orbit**: reconstruct the right orbit at time ``t`` from
the *left* orbit's coordinate at ``t`` plus the right orbit's own nuisance code
from ``t'``. Both eyes rotate together, so a coordinate that helps predict the
other orbit has to encode conjugate gaze; anything private to one orbit is
useless for the other and is pushed into the nuisance path. This is the same
constraint that makes ``lr-cca`` the best-behaved linear arm, applied
non-linearly and without labels.

Deliberately *not* included: a prior on the coordinate's range. The corpus-wide
gaze spread is not a constant -- std(GazeX) is 2.42 deg on dsL03 against 7.05
deg on dsL01 -- so constraining to one range would impose the wrong scale on
half the datasets, which is precisely the calibration failure
``analyze_calibration.py`` already measured. The cross-orbit constraint is the
load-bearing one; the readout downstream is free to rescale.

**The diagnostic that decides whether any of this worked** is not the
reconstruction loss. It is whether the coordinate is *used*: shuffling the
coordinates across the batch must degrade reconstruction. If it does not, the
bottleneck is dead and the decoder is working off the nuisance path alone --
which is the failure mode this architecture is most prone to, and the one that
would otherwise be invisible.
"""
import numpy as np

# The two orbits sit either side of the mask's trough at **x=24**, where the
# per-x voxel count falls to 52 against ~396 at each lobe's centre. That slice
# is dropped from both halves: a midline voxel appearing in both would let the
# cross-orbit objective predict an orbit partly from itself, which is exactly
# the shortcut the objective exists to forbid.
#
# 47 slices minus the trough leaves 46, so the halves are 22 wide once the two
# outermost lateral slices (x=0,1, the thinnest part of the crop) are also
# dropped to make them equal. Equal halves are what let one shared encoder see
# both orbits; the right is mirrored so the two arrive lateral-to-medial in the
# same orientation.
LEFT_X = slice(2, 24)          # x = 2..23, lateral -> medial
RIGHT_X = slice(25, 47)        # x = 25..46, mirrored to lateral -> medial
ORBIT_SHAPE = (22, 29, 18)


def _torch():
    import torch

    # See evaluate/features.py: LightGBM and PyTorch each load their own OpenMP
    # runtime and deadlock when a threaded torch op follows a LightGBM fit.
    torch.set_num_threads(1)
    return torch


def split_orbits(block):
    """``[X, Y, Z, ...]`` -> ``(left, right_mirrored)``, both ``[23, Y, Z, ...]``."""
    block = np.asarray(block)
    left = block[LEFT_X]
    right = block[RIGHT_X][::-1]
    return left, np.ascontiguousarray(right)


def build_orbit_cache(subjects, trs_per_subject=128, progress=None, dtype=np.float16):
    """Both orbits for a TR budget per participant.

    Returns ``(data [N, 2, 23, 29, 18], offsets [n+1])``. float16 because this
    is a 3 GB array that is only ever read back into float32 batches, and the
    blocks are z-scored to roughly unit scale where float16 has ample precision.
    A contiguous slab per participant keeps the two sampled timepoints inside
    one run, which the objective requires.
    """
    import h5py

    chunks, offsets = [], [0]
    for i, (_ds, _sub, path, n_trs) in enumerate(subjects):
        try:
            with h5py.File(path, "r") as f:
                b = f["eye_block"]
                t = min(b.shape[-1], trs_per_subject)
                # From the middle of the run: the first volumes carry the
                # steepest scanner drift and are the least representative.
                start = max(0, (b.shape[-1] - t) // 2)
                block = b[..., start: start + t]
        except Exception:
            continue
        left, right = split_orbits(block)
        # [23, 29, 18, T] -> [T, 2, 23, 29, 18]
        pair = np.stack([left, right], axis=0).transpose(4, 0, 1, 2, 3)
        chunks.append(pair.astype(dtype))
        offsets.append(offsets[-1] + len(pair))
        if progress and i % progress == 0:
            print(f"  [{i + 1}/{len(subjects)}] {offsets[-1]} TRs", flush=True)
    if not chunks:
        raise RuntimeError("no participant produced usable orbits")
    return np.concatenate(chunks), np.array(offsets)


def _grid(shape, device):
    """Normalised coordinate grids in [-1, 1], one per spatial axis."""
    torch = _torch()
    axes = [torch.linspace(-1.0, 1.0, n, device=device) for n in shape]
    return torch.meshgrid(*axes, indexing="ij")


class CrossOrbitModel:
    """Shared-weight orbit encoder (coordinate + nuisance) and a decoder.

    Kept as a plain wrapper rather than an ``nn.Module`` subclass so the
    untrained control is built by the identical code path -- a control
    constructed differently is not a control.
    """

    def __init__(self, n_keypoints=2, n_nuisance=32, width=16, seed=0, device=None,
                 shape=ORBIT_SHAPE):
        torch = _torch()
        import torch.nn as nn

        torch.manual_seed(seed)
        from deepmreye.temporal import device_for

        self.device = device or device_for()
        self.k, self.n_nuisance, self.shape = n_keypoints, n_nuisance, shape

        w = width
        self.enc = nn.Sequential(
            nn.Conv3d(1, w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(2 * w, 4 * w, 3, stride=1, padding=1), nn.GELU(),
        ).to(self.device)
        # Feature-map size after two stride-2 convs.
        self.fmap = tuple((n + 3) // 4 for n in shape)

        self.to_heatmap = nn.Conv3d(4 * w, n_keypoints, 1).to(self.device)
        self.to_nuisance = nn.Linear(4 * w, n_nuisance).to(self.device)

        self.dec = nn.Sequential(
            nn.Conv3d(n_keypoints + n_nuisance, 4 * w, 3, padding=1), nn.GELU(),
        ).to(self.device)
        self.dec2 = nn.Sequential(
            nn.Conv3d(4 * w, 2 * w, 3, padding=1), nn.GELU(),
        ).to(self.device)
        self.dec3 = nn.Conv3d(2 * w, 1, 3, padding=1).to(self.device)

        self._grids = None
        # Width of the Gaussian the decoder renders each keypoint back into.
        # Fixed rather than learned: a learnable width can collapse toward
        # covering the whole volume, which hands the decoder a free global
        # channel and quietly reopens the bypass the bottleneck exists to close.
        self.sigma = 0.3

    def modules_(self):
        return [self.enc, self.to_heatmap, self.to_nuisance,
                self.dec, self.dec2, self.dec3]

    def parameters(self):
        return [p for m in self.modules_() for p in m.parameters()]

    def train(self):
        for m in self.modules_():
            m.train()
        return self

    def eval(self):
        for m in self.modules_():
            m.eval()
        return self

    def encode(self, x):
        """``[B, 1, X, Y, Z]`` -> ``(coords [B, K, 3], nuisance [B, D])``."""
        torch = _torch()
        h = self.enc(x)

        heat = self.to_heatmap(h)                       # [B, K, x, y, z]
        b, k = heat.shape[:2]
        flat = heat.reshape(b, k, -1).softmax(dim=-1)
        if self._grids is None:
            self._grids = [g.reshape(-1) for g in _grid(h.shape[2:], x.device)]
        # Spatial expectation: this is what makes the latent a position rather
        # than an arbitrary 3-vector that we merely hope encodes one.
        coords = torch.stack([(flat * g).sum(-1) for g in self._grids], dim=-1)

        nuisance = self.to_nuisance(h.mean(dim=(2, 3, 4)))
        return coords, nuisance

    def render(self, coords):
        """``[B, K, 3]`` -> ``[B, K, x, y, z]`` Gaussian heatmaps."""
        torch = _torch()
        if self._grids is None:
            self._grids = [g.reshape(-1) for g in _grid(self.fmap, coords.device)]
        d2 = sum((g.view(1, 1, -1) - coords[..., i: i + 1]) ** 2
                 for i, g in enumerate(self._grids))
        heat = torch.exp(-d2 / (2 * self.sigma ** 2))
        return heat.reshape(coords.shape[0], coords.shape[1], *self.fmap)

    def decode(self, coords, nuisance):
        """Reconstruct one orbit from a coordinate and a nuisance code."""
        torch = _torch()
        import torch.nn.functional as F

        heat = self.render(coords)
        nz = nuisance[..., None, None, None].expand(-1, -1, *self.fmap)
        h = self.dec(torch.cat([heat, nz], dim=1))
        h = self.dec2(F.interpolate(h, size=tuple((n + 1) // 2 for n in self.shape),
                                    mode="trilinear", align_corners=False))
        h = F.interpolate(h, size=self.shape, mode="trilinear", align_corners=False)
        return self.dec3(h)

    def state_dict(self):
        return {"enc": self.enc.state_dict(),
                "to_heatmap": self.to_heatmap.state_dict(),
                "to_nuisance": self.to_nuisance.state_dict(),
                "dec": self.dec.state_dict(), "dec2": self.dec2.state_dict(),
                "dec3": self.dec3.state_dict(),
                "k": self.k, "n_nuisance": self.n_nuisance, "shape": self.shape}

    def load_state_dict(self, sd):
        self.enc.load_state_dict(sd["enc"])
        self.to_heatmap.load_state_dict(sd["to_heatmap"])
        self.to_nuisance.load_state_dict(sd["to_nuisance"])
        self.dec.load_state_dict(sd["dec"])
        self.dec2.load_state_dict(sd["dec2"])
        self.dec3.load_state_dict(sd["dec3"])


def sample_pairs(data, offsets, batch, rng):
    """``(x_t [B, 2, ...], x_t2 [B, 2, ...])`` -- two timepoints of one run.

    ``t2`` supplies the nuisance code and ``t`` the coordinate and the target,
    so the nuisance path is structurally unable to see the gaze it would
    otherwise be graded on.
    """
    runs = [(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)
            if offsets[i + 1] - offsets[i] >= 2]
    if not runs:
        raise RuntimeError("no run has two timepoints")
    idx_t = np.empty(batch, dtype=np.int64)
    idx_t2 = np.empty(batch, dtype=np.int64)
    for b in range(batch):
        lo, hi = runs[rng.integers(len(runs))]
        idx_t[b] = lo + rng.integers(hi - lo)
        idx_t2[b] = lo + rng.integers(hi - lo)
    return data[idx_t], data[idx_t2]


def _batch_loss(model, x_t, x_t2, shuffle_coords=False, rng=None):
    """Cross-orbit reconstruction loss and its R^2.

    ``shuffle_coords`` permutes the coordinate across the batch. It is the
    ablation that decides whether the bottleneck is alive: if reconstruction
    does not get worse, the decoder is ignoring the coordinate and working from
    the nuisance path alone.
    """
    torch = _torch()

    left_t, right_t = x_t[:, 0:1], x_t[:, 1:2]
    left_2, right_2 = x_t2[:, 0:1], x_t2[:, 1:2]

    coord_l, _ = model.encode(left_t)
    coord_r, _ = model.encode(right_t)
    _, nuis_l = model.encode(left_2)
    _, nuis_r = model.encode(right_2)

    if shuffle_coords:
        perm = torch.from_numpy(rng.permutation(len(coord_l))).to(coord_l.device)
        coord_l, coord_r = coord_l[perm], coord_r[perm]

    # Each orbit is reconstructed from the OTHER orbit's coordinate.
    pred_r = model.decode(coord_l, nuis_r)
    pred_l = model.decode(coord_r, nuis_l)

    err = ((pred_r - right_t) ** 2).sum() + ((pred_l - left_t) ** 2).sum()
    tot = (((right_t - right_t.mean()) ** 2).sum()
           + ((left_t - left_t.mean()) ** 2).sum())
    return err / (right_t.numel() * 2), 1 - float(err.detach()) / float(tot.detach())


def evaluate(model, data, offsets, batch, seed, n_batches=20):
    """Held-out reconstruction R^2, with and without a usable coordinate."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    intact, ablated = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            a, b = sample_pairs(data, offsets, batch, rng)
            a = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(model.device)
            b = torch.from_numpy(np.asarray(b, dtype=np.float32)).to(model.device)
            intact.append(_batch_loss(model, a, b)[1])
            ablated.append(_batch_loss(model, a, b, True, rng)[1])
    r2, r2_ab = float(np.mean(intact)), float(np.mean(ablated))
    return {"r2": r2, "r2_coord_shuffled": r2_ab, "coord_contribution": r2 - r2_ab}


def train(model, data, offsets, val_data, val_offsets, steps=3000, batch=32,
          lr=1e-3, seed=0, log_every=250, checkpoint_path=None, meta=None):
    """Train cross-orbit reconstruction; returns ``(history, best)``.

    Selection is on **coordinate contribution**, not on reconstruction R^2. A
    model that reconstructs well through the nuisance path alone is exactly the
    degenerate solution here, and picking on reconstruction would reward it.

    ``checkpoint_path`` writes the best state every time it improves. Without it
    a run is all-or-nothing: the best state lives only in memory until the
    function returns, so killing a long job -- or losing it to a sleeping
    laptop -- throws away every step. At >2 s/step that is hours.
    """
    import copy

    torch = _torch()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history, best = [], {"coord_contribution": -np.inf, "step": 0, "state": None}

    model.train()
    for step in range(1, steps + 1):
        a, b = sample_pairs(data, offsets, batch, rng)
        a = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(model.device)
        b = torch.from_numpy(np.asarray(b, dtype=np.float32)).to(model.device)
        loss, _ = _batch_loss(model, a, b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == steps:
            model.eval()
            m = evaluate(model, val_data, val_offsets, batch, seed + 1)
            model.train()
            m["step"], m["train_loss"] = step, float(loss.detach())
            history.append(m)
            marker = ""
            if m["coord_contribution"] > best["coord_contribution"]:
                best = {"coord_contribution": m["coord_contribution"], "step": step,
                        "state": copy.deepcopy(model.state_dict())}
                marker = "  *"
                if checkpoint_path:
                    # Written from the *best* state, not the live one, so an
                    # interrupted run leaves the same artifact it would have
                    # produced had it stopped here cleanly.
                    save(checkpoint_path, model, dict(
                        meta or {}, coord_contribution=m["coord_contribution"],
                        best_step=step, partial=True))
            print(f"  step {step:>5}  loss {m['train_loss']:.4f}  "
                  f"val R2 {m['r2']:+.4f}  shuffled {m['r2_coord_shuffled']:+.4f}  "
                  f"coord contributes {m['coord_contribution']:+.4f}{marker}",
                  flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"  restored best checkpoint from step {best['step']} "
              f"(coord contribution {best['coord_contribution']:+.4f})")
    return history, best


def save(path, model, meta):
    from pathlib import Path

    torch = _torch()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)
    return path


def load(path, device=None):
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model"]
    meta = blob["meta"]
    model = CrossOrbitModel(sd["k"], sd["n_nuisance"], meta.get("width", 16),
                            device=device, shape=tuple(sd["shape"]))
    model.load_state_dict(sd)
    return model.eval(), meta
