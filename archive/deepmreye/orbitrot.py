"""Cross-orbit prediction through a **rotation** bottleneck.

``crossorbit.py`` put the bottleneck on a soft-argmax *position*, following the
unsupervised-landmark literature (Jakab et al. 2018). It trains, it beats its
own untrained control on 6/6 folds, and it still loses badly to a linear basis
(0.389 against ``lr-cca``'s 0.798). The diagnosis is that the latent has the
wrong inductive bias, and it is measurable:

- the trained coordinate moves **0.187 voxels** over a whole run;
- a parameter-free soft-argmax over the same voxels at full resolution decodes
  gaze at r **0.367** within subject, while a *linear* projection of those same
  voxels reaches **0.904**;
- so the failure is not resolution and not a per-subject offset (between-subject
  coordinate spread is 0.20x the within-subject spread -- five times too small
  to be the problem).

Gaze does not **translate** the eyeball, it **rotates** it. The globe stays put
and intensity redistributes inside it as the lens and cornea sweep through. A
centroid is nearly blind to that, which is exactly what 0.187 voxels of travel
means. Soft-argmax keypoints are right for objects that move in frame; they are
the wrong summary for an orientation.

So the latent here *is* a rotation. The decoder holds a learned canonical orbit
appearance and renders it by rotating it -- ``affine_grid`` + ``grid_sample``,
differentiable -- about a learned centre. The bottleneck is **two angles**,
which is the true dimensionality of gaze. That is the "size-2 bottleneck
constrained to be X and Y" that motivated this line, except the constraint is
now mechanical rather than a penalty we hope holds: two numbers cannot express
anything but an orientation, because rotation is the only thing the decoder can
do with them.

Everything else is deliberately inherited from ``crossorbit`` so the comparison
is clean:

- the **cross-orbit objective** -- each orbit is reconstructed from the *other*
  orbit's angle plus its own nuisance code taken from a different TR. Both eyes
  rotate conjugately, so an angle that helps predict the other orbit has to
  encode gaze rather than anything private to one socket.
- the **nuisance path**, wide and read from ``t'``, so anatomy and coil bias
  pass freely while this TR's gaze cannot (DrNet's time-invariance trick).
- ``train``/``evaluate``/``_batch_loss`` are imported unchanged, so selection is
  still on **angle contribution** and the shuffle ablation still decides whether
  the bottleneck is alive rather than bypassed.
- the untrained control is the same class with the same seed and no training,
  built by the identical code path.

**Axis convention.** ``affine_grid`` for a 5D input ``[N, C, D, H, W]`` returns
sampling coordinates ordered ``(w, h, d)``, so rotation component 0 acts on the
*last* array axis. The angles are therefore not "horizontal" and "vertical" gaze
in any anatomical sense -- they are two orthogonal rotations of the socket, and
the readout downstream is free to map them onto gaze x and y. Do not read
``angles[:, 0]`` as gaze x.
"""
import numpy as np

from deepmreye.crossorbit import (  # noqa: F401  (re-exported for the scripts)
    ORBIT_SHAPE,
    _batch_loss,
    _torch,
    build_orbit_cache,
    evaluate,
    sample_pairs,
    save,
    split_orbits,
    train,
)

# Gaze rarely exceeds ~15 deg of visual angle, and the eyeball's rotation in the
# socket is of that order. Capping via tanh keeps the sampling grid inside the
# volume -- an unbounded angle would rotate most of the orbit out of frame and
# the reconstruction gradient would vanish.
MAX_ANGLE = 0.6                       # radians, ~34 deg


def _rotation(angles):
    """``[B, n]`` angles -> ``[B, 3, 3]`` rotation matrices, ``n`` in (2, 3).

    Composed about orthogonal axes so two angles span the gaze sphere. See the
    module docstring on why these axes are not anatomical.
    """
    torch = _torch()
    b = angles.shape[0]
    dev, dt = angles.device, angles.dtype
    zero = torch.zeros(b, device=dev, dtype=dt)
    one = torch.ones(b, device=dev, dtype=dt)

    def rot(axis, a):
        c, s = torch.cos(a), torch.sin(a)
        if axis == 0:
            rows = [(one, zero, zero), (zero, c, -s), (zero, s, c)]
        elif axis == 1:
            rows = [(c, zero, s), (zero, one, zero), (-s, zero, c)]
        else:
            rows = [(c, -s, zero), (s, c, zero), (zero, zero, one)]
        return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)

    out = rot(2, angles[:, 0]) @ rot(0, angles[:, 1])
    if angles.shape[1] > 2:
        out = out @ rot(1, angles[:, 2])
    return out


def _sample_trilinear(volume, grid):
    """``grid_sample(..., mode='trilinear', padding_mode='border')``, by hand.

    Written out rather than calling ``F.grid_sample`` because MPS does not
    implement ``grid_sampler_3d_backward``, and the CPU fallback costs 5.4 s per
    step against 0.1 s on device -- six hours a run. Everything here is gather
    and arithmetic, which every backend supports, so the module trains on MPS,
    CUDA and CPU without a device branch.

    ``volume`` is ``[B, C, D, H, W]``; ``grid`` is ``[B, D, H, W, 3]`` in
    ``[-1, 1]`` with the last axis ordered ``(x, y, z)`` indexing ``(W, H, D)``
    -- ``affine_grid``'s convention, kept so the two are interchangeable.
    """
    torch = _torch()
    b, c, d, h, w = volume.shape
    g = grid.reshape(b, -1, 3)

    # align_corners=False: coord = ((g + 1) * size - 1) / 2
    xs = ((g[..., 0] + 1) * w - 1) / 2
    ys = ((g[..., 1] + 1) * h - 1) / 2
    zs = ((g[..., 2] + 1) * d - 1) / 2

    x0, y0, z0 = torch.floor(xs), torch.floor(ys), torch.floor(zs)
    fx, fy, fz = xs - x0, ys - y0, zs - z0
    flat = volume.reshape(b, c, -1)

    def corner(zi, yi, xi):
        idx = ((zi.clamp(0, d - 1).long() * h + yi.clamp(0, h - 1).long()) * w
               + xi.clamp(0, w - 1).long())
        return torch.gather(flat, 2, idx.unsqueeze(1).expand(-1, c, -1))

    out = 0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                weight = ((fz if dz else 1 - fz) * (fy if dy else 1 - fy)
                          * (fx if dx else 1 - fx))
                out = out + corner(z0 + dz, y0 + dy, x0 + dx) * weight.unsqueeze(1)
    return out.reshape(b, c, d, h, w)


def _affine_grid(theta, shape, device, dtype):
    """``F.affine_grid`` for a 5D output, as plain arithmetic.

    Same reason as ``_sample_trilinear``: keeping the whole render path on
    primitives avoids a backend-specific fallback in the middle of training.
    """
    torch = _torch()
    d, h, w = shape

    def axis(n):
        # align_corners=False puts samples at cell CENTRES: -1 + (2i+1)/n.
        # Using linspace(-1, 1, n) here instead is the align_corners=True
        # convention, and pairing it with the False pixel mapping in
        # _sample_trilinear silently shifts every sample by half a cell.
        i = torch.arange(n, device=device, dtype=dtype)
        return (2 * i + 1) / n - 1

    gz, gy, gx = torch.meshgrid(axis(d), axis(h), axis(w), indexing="ij")
    base = torch.stack([gx, gy, gz, torch.ones_like(gx)], dim=-1)   # [D,H,W,4]
    flat = base.reshape(-1, 4)
    out = flat @ theta.transpose(1, 2)                              # [B,N,3]
    return out.reshape(theta.shape[0], d, h, w, 3)


def _smooth_init(shape, channels, generator, device, scale=0.5):
    """A low-frequency random field to initialise the canonical appearance.

    White noise would work against ``grid_sample``'s trilinear interpolation:
    with no spatial structure below the voxel scale, a small rotation changes
    the rendered volume almost randomly and the angle gets no usable gradient.
    A smooth field gives rotation something to act on from step one.
    """
    torch = _torch()
    import torch.nn.functional as F

    coarse = torch.randn(1, channels, *[max(2, n // 4) for n in shape],
                         generator=generator, device=device)
    return F.interpolate(coarse, size=tuple(shape), mode="trilinear",
                         align_corners=False) * scale


class RotationOrbitModel:
    """Shared-weight orbit encoder (angle + nuisance) and a rotating decoder.

    A plain wrapper rather than an ``nn.Module`` subclass, matching
    ``CrossOrbitModel``: the untrained control has to be built by the identical
    code path or it is not a control.

    ``encode``/``decode`` keep ``CrossOrbitModel``'s signatures exactly, so the
    training loop, the shuffle ablation and ``OrbitExtractor`` all apply
    unchanged and the two bottlenecks differ in nothing but the bottleneck.
    """

    def __init__(self, n_angles=2, n_nuisance=32, width=16, seed=0, device=None,
                 shape=ORBIT_SHAPE, template_channels=8, max_angle=MAX_ANGLE,
                 n_parts=1):
        torch = _torch()
        import torch.nn as nn

        torch.manual_seed(seed)
        from deepmreye.temporal import device_for

        self.device = device or device_for()
        # ``n_parts`` rotating templates, each with its own angles and its own
        # centre. The latent stays a set of rotations -- the inductive bias is
        # unchanged -- but widens to 2*n_parts, which is the capacity control
        # for "is 4 dimensions too few, or is the encoder the limit?". The orbit
        # is not one rigid body: globe, lens and surrounding tissue move
        # differently, so more than one rotating part is defensible rather than
        # merely convenient. ``n_parts=1`` is exactly the original model.
        self.n_parts = int(n_parts)
        self.n_angles = int(n_angles)
        self.k = self.n_angles * self.n_parts
        self.n_nuisance, self.shape = n_nuisance, tuple(shape)
        self.template_channels, self.max_angle = template_channels, max_angle

        w = width
        self.enc = nn.Sequential(
            nn.Conv3d(1, w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(2 * w, 4 * w, 3, stride=1, padding=1), nn.GELU(),
        ).to(self.device)

        self.to_angle = nn.Linear(4 * w, self.k).to(self.device)
        self.to_nuisance = nn.Linear(4 * w, n_nuisance).to(self.device)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        # The canonical orbit appearance, ``n_parts`` blocks of
        # ``template_channels``. Everything the angles can do is rotate these,
        # which is what makes the latent an orientation by construction.
        self.template = nn.Parameter(
            _smooth_init(self.shape, self.n_parts * template_channels,
                         gen, "cpu").to(self.device))
        # Rotation centre per part, in normalised coords. Learned, because the
        # eyeball is not at the centre of the cropped socket and rotating about
        # the wrong point turns a rotation into a rotation plus a translation.
        self.centre = nn.Parameter(torch.zeros(self.n_parts, 3, device=self.device))

        self.dec = nn.Sequential(
            nn.Conv3d(self.n_parts * template_channels + n_nuisance,
                      4 * w, 3, padding=1), nn.GELU(),
            nn.Conv3d(4 * w, 2 * w, 3, padding=1), nn.GELU(),
        ).to(self.device)
        self.dec_out = nn.Conv3d(2 * w, 1, 3, padding=1).to(self.device)

    # -- plumbing, mirroring CrossOrbitModel ------------------------------
    def modules_(self):
        return [self.enc, self.to_angle, self.to_nuisance, self.dec, self.dec_out]

    def parameters(self):
        return ([p for m in self.modules_() for p in m.parameters()]
                + [self.template, self.centre])

    def train(self):
        for m in self.modules_():
            m.train()
        return self

    def eval(self):
        for m in self.modules_():
            m.eval()
        return self

    # -- the bottleneck ---------------------------------------------------
    def encode(self, x):
        """``[B, 1, X, Y, Z]`` -> ``(angles [B, n], nuisance [B, D])``."""
        torch = _torch()
        h = self.enc(x)
        pooled = h.mean(dim=(2, 3, 4))
        angles = self.max_angle * torch.tanh(self.to_angle(pooled))
        return angles, self.to_nuisance(pooled)

    def render(self, angles):
        """``[B, n_parts * n_angles]`` -> each template block, rotated.

        Parts are folded into the batch so one ``grid_sample`` covers all of
        them; the output is ``[B, n_parts * template_channels, X, Y, Z]``.
        """
        torch = _torch()
        b, p, c_ch = angles.shape[0], self.n_parts, self.template_channels
        ang = angles.reshape(b * p, self.n_angles)
        rot = _rotation(ang)                                  # [B*P, 3, 3]
        # Rotate about each part's own centre rather than the volume origin:
        #   x' = R(x - c) + c  =>  translation = c - R c
        cen = self.centre.unsqueeze(0).expand(b, -1, -1).reshape(b * p, 3, 1)
        theta = torch.cat([rot, cen - rot @ cen], dim=2)      # [B*P, 3, 4]
        tmpl = (self.template.view(1, p, c_ch, *self.shape)
                .expand(b, -1, -1, -1, -1, -1).reshape(b * p, c_ch, *self.shape))
        grid = _affine_grid(theta, self.shape, angles.device, angles.dtype)
        out = _sample_trilinear(tmpl, grid)
        return out.reshape(b, p * c_ch, *self.shape)

    def decode(self, angles, nuisance):
        """Reconstruct one orbit from the other orbit's angle and its own
        nuisance code."""
        torch = _torch()
        rotated = self.render(angles)
        nz = nuisance[..., None, None, None].expand(-1, -1, *self.shape)
        return self.dec_out(self.dec(torch.cat([rotated, nz], dim=1)))

    # -- persistence ------------------------------------------------------
    def state_dict(self):
        return {"enc": self.enc.state_dict(),
                "to_angle": self.to_angle.state_dict(),
                "to_nuisance": self.to_nuisance.state_dict(),
                "dec": self.dec.state_dict(),
                "dec_out": self.dec_out.state_dict(),
                "template": self.template.detach().cpu(),
                "centre": self.centre.detach().cpu(),
                "k": self.k, "n_nuisance": self.n_nuisance,
                "shape": self.shape, "template_channels": self.template_channels,
                "max_angle": self.max_angle, "n_parts": self.n_parts,
                "n_angles": self.n_angles}

    def load_state_dict(self, sd):
        torch = _torch()
        self.enc.load_state_dict(sd["enc"])
        self.to_angle.load_state_dict(sd["to_angle"])
        self.to_nuisance.load_state_dict(sd["to_nuisance"])
        self.dec.load_state_dict(sd["dec"])
        self.dec_out.load_state_dict(sd["dec_out"])
        with torch.no_grad():
            self.template.copy_(sd["template"].to(self.device))
            self.centre.copy_(sd["centre"].to(self.device))
        return self


def load(path, device=None):
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd, meta = blob["model"], blob["meta"]
    model = RotationOrbitModel(
        # Older checkpoints predate n_parts and stored k == n_angles.
        sd.get("n_angles", sd["k"]), sd["n_nuisance"], meta.get("width", 16),
        device=device, shape=tuple(sd["shape"]),
        template_channels=sd.get("template_channels", 8),
        max_angle=sd.get("max_angle", MAX_ANGLE),
        n_parts=sd.get("n_parts", 1))
    model.load_state_dict(sd)
    return model.eval(), meta
