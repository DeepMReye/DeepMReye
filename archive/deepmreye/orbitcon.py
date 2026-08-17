"""Cross-orbit *contrastive* learning: VICReg between the two orbits of one TR.

This is the third bottleneck on the same constraint, and the constraint is the
only reason to try again. ``crossorbit.py`` (position) and ``orbitrot.py``
(rotation) both reach the cross-orbit objective through **reconstruction**: a
latent is graded on how well it lets a decoder repaint the other orbit. That
works -- ``xrot`` at matched width is the best self-supervised arm on this corpus
-- and it plateaus far below ``lr-cca``, the *linear* version of the same idea.
The suspicion this module tests is that reconstruction is the wrong grader:
repainting an orbit requires representing everything about it, so the latent is
pulled toward appearance, and the decoder's capacity sets the score. A
contrastive objective grades the latent on *agreement alone* and never asks it
to reconstruct anything.

What makes it a fair successor rather than a fourth architecture: the probe
feature is ``2 x 32 = 64`` dimensions, matched to ``lr-cca:64``, on the same
voxels, through the same extractor plumbing. If a non-linear encoder trained to
agree across orbits cannot beat the linear map trained to correlate across
orbits, the cross-orbit line is finished and that is worth knowing.

**Why the previous contrastive attempt does not bear on this one.**
``models/contrastive_net.py`` paired frames at ``t`` and ``t+dt``, which is the
objective ``temporal.py`` had already diagnosed as counterproductive on this
signal: the predictable part of an eye block is drift, motion and global signal,
so temporal invariance is a direct instruction to encode nuisance. It also ran
with no augmentation, no untrained control, and 100 participants for three
minutes. Its r ~ 0.1-0.35 is not evidence about contrastive learning here. The
two views in this module are the two *orbits of the same volume*, which is the
one pairing on this corpus with a mechanism behind it: both eyes rotate
conjugately, so the shared content across orbits at a single TR is gaze.

**The failure mode this design is mostly about.** Anatomy is also shared between
the two orbits, and it varies across a batch drawn from many participants. So
"encode which participant this is" satisfies VICReg's invariance term perfectly
and its variance term easily -- a collapse to nuisance that would look like a
healthy loss curve. Three independent defenses, because any one of them can be
argued around:

1. **Per-view intensity randomisation** (``gain``, ``bias``). Independent random
   affine rescaling of each view's voxel values makes global amplitude
   uninformative about the other view, so the cheapest shared scalar -- overall
   signal level -- stops being a solution.
2. **Per-run centering** of the cache (``center_runs``). The stored blocks are
   already per-voxel z-scored across the *whole* run, but the cache keeps only a
   128-TR window out of the middle, over which the mean is close to but not
   exactly zero. Re-centering on the cached window removes the residual static
   component, which is precisely the anatomy an invariance term would otherwise
   find first.
3. **Few runs per batch** (``runs_per_batch``). VICReg's variance and covariance
   terms are batch statistics. A batch spanning hundreds of participants has its
   variance dominated by between-participant anatomy, and satisfying the
   variance term by spreading *subjects* across dimensions costs nothing. Drawing
   a batch from a handful of runs makes the available variance within-run --
   gaze, motion, drift -- so the objective has to work with the axis we care
   about.

None of this proves the nuisance is gone, which is why the arbiter is the probe
against ``ocon-random``, and why ``evaluate`` reports agreement against a
shuffled-pairing control. Both orbits sit in one volume and share global signal,
motion and drift, so a *random* encoder already agrees with itself: the measured
floor for the position bottleneck was +0.201. An agreement number without its
untrained control is uninterpretable here.

Deliberately not included: any prior on the latent's scale or range. The corpus
gaze spread is not a constant (std(GazeX) 2.42 deg on dsL03 against 7.05 on
dsL01) and imposing one is the calibration failure ``analyze_calibration.py``
already measured. VICReg's variance term normalises the latent's spread, which
is a batch-level statistic, not a physical prior.
"""
import numpy as np

from deepmreye.crossorbit import ORBIT_SHAPE, split_orbits  # noqa: F401
from deepmreye.orbitrot import _affine_grid, _sample_trilinear

# VICReg's own coefficients, from the paper and from
# ``models/contrastive_net.py``, whose ``VICRegLoss`` this module calls rather
# than reimplementing.
DEFAULT_EMBED = 32
DEFAULT_EXPANDER = 128

# Augmentation strengths. Ranges rather than points because the whole purpose is
# that the encoder cannot know which perturbation it got.
DEFAULT_AUG = {
    "shift_voxels": 1.0,    # sub-voxel to +-1 voxel translation, trilinear
    "dropout": 0.3,         # voxels zeroed, drawn per view
    "noise": 0.2,           # additive Gaussian SD, in z-score units
    "gain": 0.3,            # multiplicative, 1 +- this
    "bias": 0.3,            # additive constant, in z-score units
}


def _torch():
    import torch

    # See evaluate/features.py: LightGBM and PyTorch each load their own OpenMP
    # runtime and deadlock when a threaded torch op follows a LightGBM fit.
    torch.set_num_threads(1)
    return torch


def unmirror_right(data):
    """Undo ``split_orbits``' x-flip of the right orbit, in place-ish.

    **This is the difference between a contrastive and a reconstructive
    cross-orbit objective, and it is not cosmetic.** ``split_orbits`` mirrors the
    right orbit so both crops run lateral-to-medial, which is right for
    ``crossorbit``/``orbitrot``: they *reconstruct* one orbit from the other, and
    that is easiest when the two look anatomically alike.

    For an invariance objective it inverts the signal. Both eyes rotate
    conjugately, so a leftward gaze moves both eyeballs the same way in *global*
    x. After the right orbit is flipped, that same motion runs toward increasing
    local x in the left crop and *decreasing* local x in the right. One shared
    encoder computes one function of both crops, so a feature that tracks
    horizontal gaze comes out with opposite sign on the two orbits -- and
    VICReg's invariance term is an MSE between them, so it penalises exactly
    that feature. Vertical and through-plane gaze are unaffected, an x-flip
    leaving them alone.

    Measured, before this was fixed: trained ``ocon`` scored r_y **0.829**
    against its untrained control's 0.512 while r_x collapsed to **0.390**
    against the control's 0.768 -- the vertical axis its best number and the
    horizontal one far *below* random. That asymmetry is the fingerprint.

    Un-mirroring also has a second effect that happens to point the right way:
    the two crops become anatomical mirror images, so matching *anatomy* across
    them gets harder while matching *gaze* gets easier. For an objective whose
    degenerate solution is anatomy, that is the useful direction.
    """
    data[:, 1] = np.ascontiguousarray(data[:, 1][:, ::-1])
    return data


def center_runs(data, offsets):
    """Subtract each run's per-voxel temporal mean, in place.

    Defense 2 in the module docstring. Cheap, and strictly in the direction of
    removing the static component that an invariance term would otherwise latch
    onto first. Returns ``data`` for chaining.
    """
    for i in range(len(offsets) - 1):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi - lo < 2:
            continue
        chunk = data[lo:hi]
        data[lo:hi] = chunk - chunk.mean(axis=0, keepdims=True)
    return data


def augment(x, aug=None, shape=ORBIT_SHAPE):
    """One view of ``[B, 1, X, Y, Z]``, independently perturbed per sample.

    Translation goes through ``orbitrot``'s hand-written ``_affine_grid`` /
    ``_sample_trilinear`` rather than ``F.grid_sample``, because MPS has no
    ``grid_sampler_3d_backward`` and the CPU fallback costs ~5 s a step. Those
    two are equivalence-tested against ``torch.nn.functional``, so reusing them
    is free; writing a second sampler here would not be.

    Randomness is device-native (``rand_like``), seeded by the global torch seed
    the model sets at construction. Drawing on the CPU and transferring would
    make every step wait on a copy for no benefit.
    """
    torch = _torch()
    cfg = dict(DEFAULT_AUG, **(aug or {}))
    b = x.shape[0]
    device, dtype = x.device, x.dtype

    def scalar(lo, hi):
        """``[B, 1, 1, 1, 1]`` uniform in ``[lo, hi]``, one draw per sample."""
        u = torch.rand(b, 1, 1, 1, 1, device=device, dtype=dtype)
        return lo + (hi - lo) * u

    if cfg["shift_voxels"]:
        # Identity rotation, random translation. `_affine_grid` orders the output
        # axes (x, y, z) indexing (W, H, D), so row j's offset is scaled by the
        # size of the axis it indexes -- getting this backwards shifts the
        # narrow axis by the wide axis's step and is invisible in the loss.
        theta = torch.zeros(b, 3, 4, device=device, dtype=dtype)
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1.0
        d, h, w = shape
        for j, n in enumerate((w, h, d)):
            u = torch.rand(b, device=device, dtype=dtype) * 2 - 1
            theta[:, j, 3] = u * (2.0 * cfg["shift_voxels"] / n)
        grid = _affine_grid(theta, shape, device, dtype)
        x = _sample_trilinear(x, grid)

    if cfg["gain"] or cfg["bias"]:
        # Defense 1: global amplitude carries no information about the other
        # view once each view is rescaled and offset independently.
        x = x * scalar(1.0 - cfg["gain"], 1.0 + cfg["gain"])
        x = x + scalar(-cfg["bias"], cfg["bias"])

    if cfg["dropout"]:
        keep = (torch.rand_like(x) >= cfg["dropout"]).to(dtype)
        x = x * keep

    if cfg["noise"]:
        x = x + torch.randn_like(x) * cfg["noise"]

    return x


class OrbitContrastModel:
    """Shared encoder over both orbits, plus a VICReg expander head.

    A plain wrapper rather than an ``nn.Module`` subclass, matching
    ``CrossOrbitModel`` and ``RotationOrbitModel``: the untrained control has to
    be built by the identical code path, and a control constructed differently
    is not a control.

    One encoder for both orbits. ``split_orbits`` mirrors the right orbit, so
    both arrive lateral-to-medial in the same orientation and sharing weights is
    the correct inductive bias rather than a saving -- it also halves the
    parameter count, which is regularisation the small corpus wants.

    The expander is where VICReg's loss is computed, not the embedding. Standard
    practice, and load-bearing rather than cosmetic: the variance and covariance
    terms force decorrelation *at the loss surface*, and applying them directly
    to a 32-dimensional probe feature would spend those dimensions satisfying a
    whitening constraint instead of carrying gaze.
    """

    def __init__(self, embed_dim=DEFAULT_EMBED, width=16, expander_dim=DEFAULT_EXPANDER,
                 seed=0, device=None, shape=ORBIT_SHAPE, head="flat",
                 mirror_right=False):
        torch = _torch()
        import torch.nn as nn

        torch.manual_seed(seed)
        from deepmreye.temporal import device_for

        self.device = device or device_for()
        self.embed_dim, self.width = embed_dim, width
        self.expander_dim, self.shape = expander_dim, tuple(shape)
        self.head = head
        # Whether this model was trained on `split_orbits`' mirrored right orbit.
        # Recorded on the model because the feature extractor has to reproduce
        # the training geometry exactly -- encoding a mirrored orbit with weights
        # trained on an un-mirrored one silently halves the score.
        self.mirror_right = mirror_right

        w = width
        self.enc = nn.Sequential(
            nn.Conv3d(1, w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(2 * w, 4 * w, 3, stride=1, padding=1), nn.GELU(),
        ).to(self.device)
        # Feature-map size after two stride-2 convs, as in `CrossOrbitModel`.
        self.fmap = tuple((n + 3) // 4 for n in self.shape)

        # `flat` keeps the feature map's spatial layout and lets one linear layer
        # read it; `gap` averages each channel over all 240 positions first.
        #
        # `flat` is the default, and the reason is the signal rather than a
        # preference for capacity. An eyeball rotating in its socket *moves mass
        # spatially* -- that is the entire measurement -- and a channel mean
        # discards where the mass went, keeping only how much of each filter
        # fired. It also has to be judged against `lr-cca`, which is a linear map
        # over all 7156 voxels of an orbit, i.e. fully spatial; handing this
        # encoder a spatially blind head would lose that comparison for a reason
        # that has nothing to do with the objective. Measured at init, `gap` also
        # collapses: pooling 240 positions leaves the embedding near-constant
        # across a batch (VICReg's variance hinge sits at its 2.0 maximum), which
        # would additionally make the untrained control a crippled one rather
        # than the honest random projection it needs to be.
        n_in = 4 * w if head == "gap" else 4 * w * int(np.prod(self.fmap))
        self.to_embed = nn.Linear(n_in, embed_dim).to(self.device)
        self.expander = nn.Sequential(
            nn.Linear(embed_dim, expander_dim), nn.GELU(),
            nn.Linear(expander_dim, expander_dim),
        ).to(self.device)

    def modules_(self):
        return [self.enc, self.to_embed, self.expander]

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
        """``[B, 1, X, Y, Z]`` -> ``[B, embed_dim]``. The probe feature."""
        h = self.enc(x)
        pooled = (h.mean(dim=(2, 3, 4)) if self.head == "gap"
                  else h.reshape(h.shape[0], -1))
        return self.to_embed(pooled)

    def project(self, z):
        """``[B, embed_dim]`` -> ``[B, expander_dim]``. Where the loss lives."""
        return self.expander(z)

    def state_dict(self):
        return {"enc": self.enc.state_dict(),
                "to_embed": self.to_embed.state_dict(),
                "expander": self.expander.state_dict(),
                "embed_dim": self.embed_dim, "width": self.width,
                "expander_dim": self.expander_dim, "shape": self.shape,
                "head": self.head, "mirror_right": self.mirror_right}

    def load_state_dict(self, sd):
        self.enc.load_state_dict(sd["enc"])
        self.to_embed.load_state_dict(sd["to_embed"])
        self.expander.load_state_dict(sd["expander"])


def sample_batch(data, offsets, batch, rng, runs_per_batch=4):
    """``[B, 2, X, Y, Z]`` -- both orbits at ``B`` timepoints.

    Defense 3 in the module docstring: the timepoints come from at most
    ``runs_per_batch`` runs, so the variance VICReg is asked to spread across
    dimensions is variance *within* a few participants rather than between
    hundreds of them. With one run per batch the batch statistics are purely
    within-subject; the default of 4 keeps the covariance estimate from being
    one participant's idiosyncrasy while staying far from the many-subject
    regime where encoding identity is the cheapest solution.
    """
    runs = [(int(offsets[i]), int(offsets[i + 1])) for i in range(len(offsets) - 1)
            if offsets[i + 1] - offsets[i] >= 2]
    if not runs:
        raise RuntimeError("no run has two timepoints")
    k = min(runs_per_batch, len(runs))
    chosen = [runs[i] for i in rng.choice(len(runs), size=k, replace=False)]
    idx = np.empty(batch, dtype=np.int64)
    for b in range(batch):
        lo, hi = chosen[b % k]
        idx[b] = lo + rng.integers(hi - lo)
    return data[idx]


def _views(model, x, aug):
    """``(z_left, z_right)`` on independently augmented views of one TR."""
    left, right = x[:, 0:1], x[:, 1:2]
    return (model.encode(augment(left, aug, model.shape)),
            model.encode(augment(right, aug, model.shape)))


def _batch_loss(model, x, aug=None):
    """VICReg between the two orbits' expanded embeddings."""
    from deepmreye.models.contrastive_net import VICRegLoss

    z_l, z_r = _views(model, x, aug)
    loss_fn = VICRegLoss()
    total, sim, std, cov = loss_fn(model.project(z_l), model.project(z_r))
    return total, {"loss": float(total.detach()), "sim": float(sim.detach()),
                   "std": float(std.detach()), "cov": float(cov.detach())}


def agreement(z_l, z_r):
    """Mean per-dimension Pearson r between the two orbits' embeddings.

    The cross-orbit objective's own success criterion, and invisible in the
    probe table. Reported against a shuffled-pairing control in ``evaluate``,
    without which it means nothing: both orbits live in one volume and share
    global signal, motion and drift, so a random encoder self-agrees at a
    substantial positive value (+0.201, measured for the position bottleneck).
    """
    a = np.asarray(z_l, dtype=np.float64)
    b = np.asarray(z_r, dtype=np.float64)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    ok = denom > 1e-12
    if not ok.any():
        return float("nan")
    return float(((a * b).sum(axis=0)[ok] / denom[ok]).mean())


def agreement_within_runs(model, data, offsets, batch, seed, n_runs=24):
    """L/R agreement computed **inside** each run, then averaged.

    This is the measurement that separates gaze from anatomy, and the pooled
    ``agreement`` cannot do it. If the encoder learned nothing but "which
    participant is this", then ``z_L`` and ``z_R`` are both a function of the
    subject: pooled across runs they agree almost perfectly, and the shuffled
    control still reads ~0 because a random re-pairing crosses subjects. So a
    high pooled agreement with a null shuffled control is exactly as consistent
    with the degenerate solution as with the intended one.

    Within a single run the participant is constant, so anatomy contributes no
    variance and whatever agreement remains has to come from something that
    moves during the run -- gaze, motion or drift. Against its untrained control
    that is the honest statement of what the objective learned.
    """
    torch = _torch()
    rng = np.random.default_rng(seed)
    runs = [(int(offsets[i]), int(offsets[i + 1])) for i in range(len(offsets) - 1)
            if offsets[i + 1] - offsets[i] >= 8]
    if not runs:
        return float("nan")
    chosen = [runs[i] for i in rng.choice(len(runs), size=min(n_runs, len(runs)),
                                          replace=False)]
    scores = []
    with torch.no_grad():
        for lo, hi in chosen:
            take = min(batch, hi - lo)
            idx = lo + rng.choice(hi - lo, size=take, replace=False)
            t = torch.from_numpy(np.asarray(data[np.sort(idx)],
                                            dtype=np.float32)).to(model.device)
            z_l = model.encode(t[:, 0:1]).cpu().numpy()
            z_r = model.encode(t[:, 1:2]).cpu().numpy()
            s = agreement(z_l, z_r)
            if np.isfinite(s):
                scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(model, data, offsets, batch, seed, n_batches=20, aug=None,
             runs_per_batch=4, n_within_runs=24):
    """Held-out VICReg loss and three agreement numbers.

    Agreement is measured on **unaugmented** volumes: the augmentations exist to
    shape what the encoder learns, not to define the quantity being reported.

    - ``agreement`` pooled across runs, and ``agreement_shuffled``, which pairs
      each left orbit with another timepoint's right orbit. The shuffled column
      rules out a constant offset and nothing more -- see
      ``agreement_within_runs`` for why it does not rule out anatomy.
    - ``agreement_within_run`` is the one to read.
    """
    torch = _torch()
    rng = np.random.default_rng(seed)
    stats, zs_l, zs_r = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x = sample_batch(data, offsets, batch, rng, runs_per_batch)
            t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(model.device)
            stats.append(_batch_loss(model, t, aug)[1])
            zs_l.append(model.encode(t[:, 0:1]).cpu().numpy())
            zs_r.append(model.encode(t[:, 1:2]).cpu().numpy())
    z_l, z_r = np.concatenate(zs_l), np.concatenate(zs_r)
    perm = np.random.default_rng(seed + 7).permutation(len(z_r))
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out["agreement"] = agreement(z_l, z_r)
    out["agreement_shuffled"] = agreement(z_l, z_r[perm])
    out["agreement_margin"] = out["agreement"] - out["agreement_shuffled"]
    out["agreement_within_run"] = agreement_within_runs(
        model, data, offsets, batch, seed + 3, n_within_runs)
    return out


def train(model, data, offsets, val_data, val_offsets, steps=3000, batch=64,
          lr=1e-3, weight_decay=1e-2, seed=0, log_every=250, aug=None,
          runs_per_batch=4, checkpoint_path=None, meta=None):
    """Train cross-orbit VICReg; returns ``(history, best)``.

    Selection is on **held-out VICReg loss**, the objective's own
    generalisation, not on agreement. Selecting on agreement would reward
    exactly the degenerate solution the module docstring is about: encoding
    anatomy maximises L/R agreement while carrying no gaze. The probe is the
    arbiter and is measured separately, by ``eval_probe.py`` against
    ``ocon-random``.

    ``weight_decay`` defaults to 1e-2 rather than AdamW's usual 1e-4, and a
    cosine schedule decays the learning rate to zero. The corpus is large but
    the *number of independent acquisitions* is not, and every trained arm on
    this project so far has been beaten by a linear map -- if this one wins it
    should be because the objective is right, not because the encoder had room
    to memorise runs.

    ``checkpoint_path`` writes the best state whenever it improves, so a killed
    or slept-through run leaves the artifact it would have produced had it
    stopped cleanly.
    """
    import copy

    torch = _torch()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    history, best = [], {"loss": np.inf, "step": 0, "state": None}

    model.train()
    for step in range(1, steps + 1):
        x = sample_batch(data, offsets, batch, rng, runs_per_batch)
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(model.device)
        loss, _ = _batch_loss(model, t, aug)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps:
            model.eval()
            m = evaluate(model, val_data, val_offsets, batch, seed + 1, aug=aug,
                         runs_per_batch=runs_per_batch)
            model.train()
            m["step"], m["train_loss"] = step, float(loss.detach())
            history.append(m)
            marker = ""
            if m["loss"] < best["loss"]:
                best = {"loss": m["loss"], "step": step,
                        "state": copy.deepcopy(model.state_dict())}
                marker = "  *"
                if checkpoint_path:
                    save(checkpoint_path, model, dict(
                        meta or {}, val_loss=m["loss"], best_step=step,
                        agreement=m["agreement"],
                        agreement_within_run=m["agreement_within_run"],
                        agreement_margin=m["agreement_margin"], partial=True))
            print(f"  step {step:>5}  train {m['train_loss']:.3f}  "
                  f"val {m['loss']:.3f}  (sim {m['sim']:.3f} std {m['std']:.3f} "
                  f"cov {m['cov']:.3f})  agree {m['agreement']:+.3f} "
                  f"shuf {m['agreement_shuffled']:+.3f} "
                  f"within-run {m['agreement_within_run']:+.3f}{marker}",
                  flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"  restored best checkpoint from step {best['step']} "
              f"(val loss {best['loss']:.4f})")
    return history, best


def save(path, model, meta):
    from pathlib import Path

    torch = _torch()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)
    return path


def load(path, device=None):
    """Rebuild a trained model, and its architecture, from the checkpoint alone.

    Every constructor argument comes out of the saved ``state_dict``, never from
    a caller's configuration. This is the ``xrot`` lesson written into the
    loader: adding ``--parts`` there updated the model and the trainer but not
    the place that built the untrained control, which then compared a 4-feature
    control against a 24-feature model and inflated the reported margin to
    +0.370 against a true +0.214. A control assembled from configuration rather
    than from the thing it controls drifts the next time a field is added.
    """
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd, meta = blob["model"], blob["meta"]
    model = build_from_state(sd, device=device)
    model.load_state_dict(sd)
    return model.eval(), meta


def build_from_state(sd, device=None, seed=0):
    """The untrained control's factory: architecture from a state dict."""
    missing = [k for k in ("embed_dim", "width", "expander_dim", "shape", "head",
                           "mirror_right") if k not in sd]
    if missing:
        raise ValueError(
            f"checkpoint is missing architecture fields {missing}; it cannot "
            f"build a matching control. Retrain, or fix state_dict().")
    return OrbitContrastModel(sd["embed_dim"], sd["width"], sd["expander_dim"],
                              seed=seed, device=device, shape=tuple(sd["shape"]),
                              head=sd["head"], mirror_right=sd["mirror_right"])
