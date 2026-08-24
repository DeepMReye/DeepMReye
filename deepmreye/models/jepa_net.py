"""Orbit-JEPA: a cross-orbit joint-embedding predictive architecture that
*contains linear CCA as a special case* and is initialised at it.

The point of this design, and why it differs from the first attempt
--------------------------------------------------------------------
Every trained arm on this corpus has lost to a linear map (see `CLAUDE.md`:
JEPA, next-TR, CompositeNet, `xorb`, `xrot`, `ocon`). The reason recorded
there is structural rather than incidental -- a 64-dimensional linear subspace
of a 14236-voxel eye mask is easy to estimate, and leave-one-dataset-out
punishes anything fitted to the training folds' gaze. So a non-linear model
that *starts from scratch* is starting behind and has to climb back to a
baseline it can only tie.

This architecture removes that handicap by construction:

1. **Frozen canonical pre-projection.** Each orbit's ~7100 voxels are first
   projected onto the `left_weights` / `right_weights` of the frozen corpus
   `lr-cca` basis (`M=256` directions per orbit, already on disk, fitted on
   unlabeled participants only). The network never sees a raw voxel. This is
   the "CCA -> JEPA adapter" the handover asked for, and it is what makes a
   non-linear fit tractable: DeepCCA-style objectives need the input dimension
   to be small relative to the batch, or the in-batch covariance is
   rank-deficient and the canonical directions come back as noise at
   correlation 1.0 (Wang et al. 2015; the same reason `fit_lr_cca` reduces to
   `n_reduce` directions before whitening).

2. **A linear identity path, initialised at the linear solution.** Each
   encoder is ``s = z @ W_lin + MLP(z)`` with ``W_lin`` initialised to
   ``I[:, :k]`` -- i.e. "take the first k canonical variates" -- and the MLP's
   last layer initialised to **zero** (Fixup/ReZero-style residual init, which
   is also what BatchNorm does implicitly: it biases residual blocks toward the
   identity at init, De & Smith 2020).

   The consequence is exact, not approximate: at step 0 the averaged latent
   ``0.5 (s_L + s_R)`` equals ``project("lr-cca", basis, x, k)`` **bit for
   bit**. `test_untrained_jepa_reproduces_lr_cca_exactly` asserts it.

   So the untrained control is not a random projection -- it is the 0.825 arm
   itself, which is the strongest control available and the only one that makes
   "did the non-linearity earn anything" a well-posed question. Every gain over
   it is a gain over `lr-cca:k` measured on identical folds, identical windows
   and identical readout.

3. **Both encoders are trained, and there is no cross-space EMA.** The first
   implementation froze the right encoder and dragged it toward the *left*
   encoder's weights by EMA. The two orbits are different voxel sets, so those
   parameter matrices index different anatomy and the EMA copied a column
   prefix between unrelated inputs -- the prediction target was noise. A
   BYOL/I-JEPA momentum encoder is only meaningful when context and target
   share an input space (two masked views of one image); here they do not, so
   the objective is made **symmetric** instead, with a stop-gradient on
   whichever side is the target. With linear encoders and an isotropy
   constraint, that objective's optimum *is* CCA, which is consistent with
   starting there.

`SIGRegLoss` and the collapse bug it used to have
-------------------------------------------------
SIGReg (Balestriero & LeCun 2025, LeJEPA) replaces BYOL's heuristics with a
goodness-of-fit test: project the batch onto random 1D directions and push each
marginal toward N(0, 1) via the Epps-Pulley statistic. Correct, and the
statistic must be **minimised** by N(0, 1).

The previous implementation had the two exponent denominators swapped, which
inverts it: it scored N(0, 1) at 0.285 and total collapse at 0.163, so gradient
descent walked into collapse. The saved training history of
`models/orbitjepa_n1039.pt` sits at 0.16314 -- the analytic collapse value,
``1 - sqrt(2) + 1/sqrt(3)`` -- from the first epoch, with the prediction loss
falling to 3e-5 because a constant is trivially predictable. That model is
fully collapsed and its probe score is whatever leaked through its residual
linear path.

The statistic below is the textbook form (Epps & Pulley 1983), which evaluates
to ~0 at N(0, 1) and 0.1631 at collapse. `test_sigreg_is_minimised_by_its_own
_target_distribution` pins that ordering, including collapse *toward zero* --
the direction the old test missed by using ``ones * 5.0``, an offset the buggy
statistic happens to penalise.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1 - sqrt(2) + 1/sqrt(3): the value the Epps-Pulley statistic takes when every
# embedding is identical. Exported because it is the number that identifies a
# collapsed run at a glance in a training log.
SIGREG_COLLAPSE_VALUE = 1.0 - math.sqrt(2.0) + 1.0 / math.sqrt(3.0)


class SIGRegLoss(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero & LeCun 2025).

    Projects embeddings ``[B, D]`` onto ``n_sketches`` random unit directions and
    scores each 1D marginal against N(0, 1) with the Epps-Pulley characteristic
    function statistic

        T = mean_jk exp(-(z_j - z_k)^2 / 2) - sqrt(2) mean_j exp(-z_j^2 / 4)
            + 1 / sqrt(3)

    which is ~0 for a standard Gaussian and ``SIGREG_COLLAPSE_VALUE`` (0.1631)
    when the batch is constant. **The exponent denominators are 2 and 4 in that
    order and swapping them inverts the objective** -- see the module docstring.
    """

    def __init__(self, n_sketches: int = 64):
        super().__init__()
        self.n_sketches = n_sketches

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D tensor [B, D], got shape {embeddings.shape}")
        batch_size, dim = embeddings.shape
        if batch_size < 4:
            return embeddings.sum() * 0.0

        u = F.normalize(torch.randn(dim, self.n_sketches, device=embeddings.device,
                                    dtype=embeddings.dtype), p=2, dim=0)
        proj = embeddings @ u                                     # [B, M]

        sq_diff = (proj.unsqueeze(0) - proj.unsqueeze(1)) ** 2     # [B, B, M]
        term1 = torch.exp(-0.5 * sq_diff).mean(dim=(0, 1))         # [M]
        term2 = math.sqrt(2.0) * torch.exp(-0.25 * proj ** 2).mean(dim=0)
        return (term1 - term2 + 1.0 / math.sqrt(3.0)).mean()


class ResidualEncoder(nn.Module):
    """``z [B, M] -> s [B, k]``, as a linear map plus a zero-initialised MLP
    and optional causal 1D spatiotemporal convolution stream.

    ``s = z @ W_lin + alpha_spatial * MLP(z) + alpha_temporal * Conv1D(z)``.
    ``W_lin`` starts as ``I[:, :k]`` (select the first ``k`` canonical variates)
    and the MLP / Conv1D output layers start at zero with zero ReZero scalars,
    so the block is *exactly* the linear selection at initialisation and the
    non-linear / dynamic temporal capacity has to be earned by gradient descent.
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 256,
                 depth: int = 2, dropout: float = 0.1, train_linear: bool = True,
                 temp_kernel: int = 1, alpha_gate: float = 1.0):
        super().__init__()
        if latent_dim > in_dim:
            raise ValueError(
                f"latent_dim={latent_dim} exceeds the pre-projection width "
                f"in_dim={in_dim}; the identity initialisation needs k <= M")
        self.in_dim, self.latent_dim = in_dim, latent_dim
        self.temp_kernel = int(temp_kernel)
        self.alpha_gate = float(alpha_gate)

        self.linear = nn.Linear(in_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[:, :latent_dim] = torch.eye(latent_dim)
        self.linear.weight.requires_grad_(bool(train_linear))

        # Spatial non-linear MLP branch
        layers, d = [], in_dim
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        out = nn.Linear(d, latent_dim)
        with torch.no_grad():                     # Fixup / ReZero residual init
            out.weight.zero_()
            out.bias.zero_()
        layers.append(out)
        self.mlp = nn.Sequential(*layers)

        # Spatiotemporal dynamic sequence branch (causal 1D convolution over consecutive TRs)
        if self.temp_kernel > 1:
            self.temp_conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=self.temp_kernel, padding=0)
            self.temp_norm = nn.LayerNorm(hidden_dim)
            self.temp_gelu = nn.GELU()
            self.temp_drop = nn.Dropout(dropout)
            self.temp_conv2 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
            with torch.no_grad():
                self.temp_conv2.weight.zero_()
                self.temp_conv2.bias.zero_()
        else:
            self.temp_conv1 = None
            self.temp_norm = None
            self.temp_gelu = None
            self.temp_drop = None
            self.temp_conv2 = None

        # Learnable ReZero gating parameters
        self.alpha_spatial = nn.Parameter(torch.tensor(float(alpha_gate)))
        self.alpha_temporal = nn.Parameter(torch.tensor(float(alpha_gate)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            # 2D tensor [T, M]
            lin = self.linear(z)
            spatial = self.mlp(z)
            a_spat = torch.clamp(self.alpha_spatial, -1.0, 1.0)
            if self.temp_conv1 is None:
                return lin + a_spat * spatial

            # Causal 1D convolution
            T, M = z.shape
            z_trans = z.T.unsqueeze(0)  # [1, M, T]
            z_pad = F.pad(z_trans, (self.temp_kernel - 1, 0), mode="replicate")
            c = self.temp_conv1(z_pad)  # [1, hidden, T]
            c = self.temp_norm(c.squeeze(0).T).T.unsqueeze(0)
            c = self.temp_gelu(c)
            c = self.temp_drop(c)
            c = self.temp_conv2(c).squeeze(0).T  # [T, latent_dim]
            a_temp = torch.clamp(self.alpha_temporal, -1.0, 1.0)
            return lin + a_spat * spatial + a_temp * c

        elif z.ndim == 3:
            # 3D tensor [B, T, M]
            B, T, M = z.shape
            lin = self.linear(z)
            spatial = self.mlp(z)
            a_spat = torch.clamp(self.alpha_spatial, -1.0, 1.0)
            if self.temp_conv1 is None:
                return lin + a_spat * spatial

            z_trans = z.permute(0, 2, 1)  # [B, M, T]
            z_pad = F.pad(z_trans, (self.temp_kernel - 1, 0), mode="replicate")
            c = self.temp_conv1(z_pad)  # [B, hidden, T]
            c = self.temp_norm(c.permute(0, 2, 1)).permute(0, 2, 1)
            c = self.temp_gelu(c)
            c = self.temp_drop(c)
            c = self.temp_conv2(c).permute(0, 2, 1)  # [B, T, latent_dim]
            a_temp = torch.clamp(self.alpha_temporal, -1.0, 1.0)
            return lin + a_spat * spatial + a_temp * c
        else:
            raise ValueError(f"expected 2D or 3D tensor, got shape {z.shape}")

    def nonlinear_share(self, z: torch.Tensor) -> float:
        """Fraction of the output's norm contributed by the non-linear branches."""
        with torch.no_grad():
            lin = self.linear(z)
            full = self.forward(z)
            non = full - lin
            denom = lin.norm() + non.norm()
            return float(non.norm() / denom) if denom > 0 else 0.0


class OrbitJEPA(nn.Module):
    """Symmetric cross-orbit Dual-Stream Spatiotemporal JEPA over the frozen canonical pre-projection.

    Parameters
    ----------
    in_dim : int
        Width of the frozen pre-projection per orbit (``M``, e.g. 256 canonical
        directions from the corpus `lr-cca` basis).
    latent_dim : int
        Latent width ``k``. Matched to the `lr-cca:k` arm being compared
        against, so the probe sees the same number of features.
    hidden_dim, depth, dropout
        The MLP / Conv1D branches.
    temp_kernel : int
        Temporal kernel size for causal 1D convolution (default 1 = purely spatial).
    sigreg_weight : float
        Weight on the isotropy term.
    train_linear : bool
        Whether the linear identity path itself is trainable.
    """

    def __init__(self, in_dim: int, latent_dim: int = 32, hidden_dim: int = 256,
                 depth: int = 2, dropout: float = 0.1, sigreg_weight: float = 1.0,
                 n_sketches: int = 64, train_linear: bool = True,
                 temp_kernel: int = 1, alpha_gate: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.sigreg_weight = sigreg_weight
        self.train_linear = train_linear
        self.temp_kernel = int(temp_kernel)
        self.alpha_gate = float(alpha_gate)

        kw = dict(hidden_dim=hidden_dim, depth=depth, dropout=dropout,
                  train_linear=train_linear, temp_kernel=temp_kernel,
                  alpha_gate=alpha_gate)
        self.left_encoder = ResidualEncoder(in_dim, latent_dim, **kw)
        self.right_encoder = ResidualEncoder(in_dim, latent_dim, **kw)

        # Predictors, one per direction
        self.predict_lr = self._identity_predictor(latent_dim, hidden_dim, dropout)
        self.predict_rl = self._identity_predictor(latent_dim, hidden_dim, dropout)

        self.sigreg = SIGRegLoss(n_sketches=n_sketches)

    @staticmethod
    def _identity_predictor(latent_dim, hidden_dim, dropout):
        return ResidualEncoder(latent_dim, latent_dim, hidden_dim=hidden_dim,
                               depth=2, dropout=dropout, train_linear=True,
                               temp_kernel=1, alpha_gate=0.0)

    def arch(self):
        """Every constructor argument, for the checkpoint."""
        return {"in_dim": self.in_dim, "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim, "depth": self.depth,
                "sigreg_weight": self.sigreg_weight,
                "train_linear": self.train_linear,
                "temp_kernel": self.temp_kernel,
                "alpha_gate": self.alpha_gate}

    def encode(self, z_left, z_right):
        return self.left_encoder(z_left), self.right_encoder(z_right)

    def forward(self, z_left: torch.Tensor, z_right: torch.Tensor):
        """Symmetric cross-orbit prediction plus isotropy."""
        s_L, s_R = self.encode(z_left, z_right)

        # Reshape for SIGReg if sequence [B, T, k]
        s_L_flat = s_L.reshape(-1, self.latent_dim) if s_L.ndim == 3 else s_L
        s_R_flat = s_R.reshape(-1, self.latent_dim) if s_R.ndim == 3 else s_R

        pred_loss = 0.5 * (F.smooth_l1_loss(self.predict_lr(s_L), s_R.detach())
                           + F.smooth_l1_loss(self.predict_rl(s_R), s_L.detach()))
        sigreg_loss = 0.5 * (self.sigreg(s_L_flat) + self.sigreg(s_R_flat))

        return {"loss": pred_loss + self.sigreg_weight * sigreg_loss,
                "pred_loss": pred_loss, "sigreg_loss": sigreg_loss,
                "s_L": s_L, "s_R": s_R}

    # -- numpy export -----------------------------------------------------
    def to_numpy_weights(self):
        """Plain-array weights for `deepmreye.orbitjepa.encode_numpy`."""
        def enc(e):
            out = {
                "linear": e.linear.weight.detach().cpu().numpy().T.copy(),
                "layers": [],
                "temp_kernel": e.temp_kernel,
                "alpha_spatial": float(torch.clamp(e.alpha_spatial, -1.0, 1.0).item()),
                "alpha_temporal": float(torch.clamp(e.alpha_temporal, -1.0, 1.0).item()),
            }
            for m in e.mlp:
                if isinstance(m, nn.Linear):
                    out["layers"].append(
                        ("linear", m.weight.detach().cpu().numpy().T.copy(),
                         None if m.bias is None else m.bias.detach().cpu().numpy().copy()))
                elif isinstance(m, nn.LayerNorm):
                    out["layers"].append(
                        ("layernorm", m.weight.detach().cpu().numpy().copy(),
                         m.bias.detach().cpu().numpy().copy(), float(m.eps)))
                elif isinstance(m, nn.GELU):
                    out["layers"].append(("gelu",))
                elif isinstance(m, nn.Dropout):
                    out["layers"].append(("identity",))   # eval mode
                else:
                    raise TypeError(f"no numpy path for {type(m).__name__}")

            if e.temp_conv1 is not None:
                out["temp_conv1"] = (
                    e.temp_conv1.weight.detach().cpu().numpy().copy(),
                    None if e.temp_conv1.bias is None else e.temp_conv1.bias.detach().cpu().numpy().copy()
                )
                out["temp_norm"] = (
                    e.temp_norm.weight.detach().cpu().numpy().copy(),
                    e.temp_norm.bias.detach().cpu().numpy().copy(),
                    float(e.temp_norm.eps)
                )
                out["temp_conv2"] = (
                    e.temp_conv2.weight.detach().cpu().numpy().copy(),
                    None if e.temp_conv2.bias is None else e.temp_conv2.bias.detach().cpu().numpy().copy()
                )
            return out

        return {"left": enc(self.left_encoder), "right": enc(self.right_encoder)}


def untrained_like(model: OrbitJEPA) -> OrbitJEPA:
    """A fresh model with the same architecture -- the mandatory control.

    Built from ``model.arch()`` rather than from configuration, so a field added
    to the constructor cannot silently produce a control of the wrong width.
    Because of the identity initialisation this control *is* `lr-cca:k`, so it
    is a genuine baseline rather than a random projection.
    """
    fresh = OrbitJEPA(**model.arch())
    fresh.eval()
    return fresh


def gelu_numpy(x):
    """Exact GELU, matching torch's default (erf) formulation."""
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
