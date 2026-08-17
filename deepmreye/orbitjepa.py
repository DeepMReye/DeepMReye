"""Orbit-JEPA pretraining over the unlabeled corpus, and its numpy feature path.

The expensive part of every earlier trained arm on this project was reading
1039 participants' voxels once per configuration. Here the corpus is reduced
**once** to its canonical pre-projection -- ``[T, 2, M]`` per run, M=256
directions per orbit from the frozen `lr-cca` basis -- and cached. That is
~280 MB for the whole corpus against 6.0 GB for `orbitcon`'s raw orbit cache,
and it turns a pretraining run into seconds on CPU, which is what makes a real
hyperparameter sweep affordable.

The cache records the basis it was built from and refuses to load against a
different one. `CLAUDE.md` records why: `orbitcon`'s raw cache silently
survived a change to the orbit split, and a stale cache is invisible in every
number downstream.

Feature extraction here is **pure numpy** (`encode_numpy`). See the note in
`models/jepa_net.py`: torch in the feature path deadlocks against LightGBM's
OpenMP runtime with no traceback.
"""
import numpy as np
import torch

from deepmreye.gauge import as_rows, motion_proxy, regress_out
from deepmreye.models.jepa_net import OrbitJEPA, gelu_numpy

CACHE_VERSION = 2
MIN_TR = 100


# --------------------------------------------------------------------------
# The frozen canonical pre-projection
# --------------------------------------------------------------------------
def orbit_projections(rows, basis, m=256, regress_motion=False):
    """Masked voxel rows ``[T, 14236]`` or ``[B, T, 14236]`` -> two orbits' canonical coords."""
    rows = np.asarray(rows, dtype=np.float64)
    is_3d = (rows.ndim == 3)
    if is_3d:
        B, T, V = rows.shape
        flat = rows.reshape(B * T, V)
    else:
        flat = rows

    li, ri = np.asarray(basis["left_index"]), np.asarray(basis["right_index"])
    wl, wr = basis["left_weights"], basis["right_weights"]
    m = int(min(m, wl.shape[1], wr.shape[1]))

    centred = flat - basis["mean"]
    left, right = centred[:, li], centred[:, ri]
    if regress_motion:
        conf = motion_proxy(flat)
        left, right = regress_out(left, conf), regress_out(right, conf)

    zl = left @ wl[:, :m]
    zr = right @ wr[:, :m]
    if is_3d:
        return zl.reshape(B, T, m), zr.reshape(B, T, m)
    return zl, zr


def build_corpus_cache(paths, mask, basis, m=256, max_files=None,
                       regress_motion=False, min_tr=MIN_TR, verbose=True):
    """Reduce unlabeled participants to ``(z [N, 2, m], run_id [N])``.

    Skips anything carrying `labels` -- the labeled datasets are the evaluation
    set and must not be pretrained on -- plus short runs and runs with
    degenerate variance.
    """
    import h5py

    zs, run_ids, kept = [], [], []
    for path in list(paths)[: max_files or len(list(paths))]:
        try:
            with h5py.File(path, "r") as f:
                if "labels" in f:
                    continue
                block = f["eye_block"][:]
        except Exception:
            continue
        if block.shape[-1] < min_tr:
            continue

        rows = as_rows(block, mask)
        if (rows.std(axis=0) > 1e-6).sum() < 10000:
            continue

        zl, zr = orbit_projections(rows, basis, m=m, regress_motion=regress_motion)
        zs.append(np.stack([zl, zr], axis=1).astype(np.float32))
        run_ids.append(np.full(len(rows), len(kept), dtype=np.int32))
        kept.append(str(path))
        if verbose and len(kept) % 100 == 0:
            print(f"    cached {len(kept)} runs", flush=True)

    if not zs:
        raise ValueError("no usable unlabeled corpus runs found")
    return (np.concatenate(zs, axis=0), np.concatenate(run_ids), kept)


def save_cache(path, z, run_id, runs, basis_path, m, regress_motion):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, z=z, run_id=run_id, n_runs=np.array([len(runs)]),
             version=np.array([CACHE_VERSION]), m=np.array([m]),
             regress_motion=np.array([int(regress_motion)]),
             basis=np.array(str(basis_path)))


def load_cache(path, basis_path, m, regress_motion):
    """Load a cache, refusing any geometry other than the one requested."""
    d = np.load(path, allow_pickle=False)
    got = (int(d["version"][0]), int(d["m"][0]), bool(d["regress_motion"][0]),
           str(d["basis"]))
    want = (CACHE_VERSION, int(m), bool(regress_motion), str(basis_path))
    if got != want:
        raise ValueError(
            f"cache at {path} was built for {got} but {want} was requested; "
            f"delete it and rebuild rather than training on stale geometry")
    return d["z"], d["run_id"], int(d["n_runs"][0])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def split_runs(run_id, val_frac=0.1, seed=0):
    """Train/val split **by run**, never by TR.

    Neighbouring TRs of one run are near-duplicates, so a TR-level split scores
    the model on frames it effectively trained on and every loss curve looks
    better than it is -- the same grouped-CV argument `combine.py` makes for
    choosing regularisation.
    """
    runs = np.unique(run_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(runs)
    n_val = max(int(round(len(runs) * val_frac)), 1)
    val_runs = set(perm[:n_val].tolist())
    is_val = np.array([r in val_runs for r in run_id])
    return ~is_val, is_val


def augment_batch(z, noise=0.0, dropout=0.0, gain=0.0, rng=None):
    """Per-view jitter on the canonical coordinates.

    Note that **independent per-view gain is off by default**, unlike the first
    implementation. It was there to stop the encoder keying on amplitude, but a
    gaze coordinate *is* a signed amplitude along a canonical direction, so
    rescaling the two orbits independently corrupts precisely the conjugate
    relationship the objective exists to exploit. Kept as a flag so the claim
    is measured rather than asserted.
    """
    rng = rng or np.random
    out = z
    if gain > 0:
        g = rng.uniform(1.0 - gain, 1.0 + gain, size=(z.shape[0], z.shape[1], 1))
        out = out * g.astype(np.float32)
    if noise > 0:
        out = out + rng.normal(0, noise, size=out.shape).astype(np.float32)
    if dropout > 0:
        keep = (rng.uniform(size=out.shape) >= dropout).astype(np.float32)
        out = out * keep / max(1.0 - dropout, 1e-6)
    return out


def train_orbit_jepa(model, z, run_id, epochs=30, batch_size=512, lr=1e-3,
                     weight_decay=1e-2, noise=0.0, dropout=0.0, gain=0.0,
                     val_frac=0.1, seed=0, device="cpu", verbose=True,
                     on_epoch=None):
    """Train the symmetric cross-orbit objective. Returns ``(model, history)``.

    Selection is on **held-out-run** total loss, and the returned model is the
    best-scoring epoch's weights rather than the last -- the score to beat is a
    warm start, so a run that drifts away from it must not be silently reported
    as its final state.

    ``on_epoch(epoch, model, row)`` is called after every epoch, and is how the
    caller snapshots intermediate weights. **This is not a convenience.** The
    `ocon` result on this corpus is that the cross-orbit objective improves
    monotonically with training while gaze decoding *peaks early and then
    falls* -- val loss and probe r moved in opposite directions. Selecting on
    the objective alone would therefore report whichever epoch is worst at the
    thing being measured, so the probe has to be evaluated as a function of
    training progress rather than only at the objective's optimum.
    """
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr, va = split_runs(run_id, val_frac=val_frac, seed=seed)
    z_tr, z_va = z[tr], z[va]
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-2)

    # Validation is scored in batches of the *training* batch size, never in one
    # pass. SIGReg forms a [B, B, M] pairwise table, so its memory is quadratic
    # in the batch: at 512 that is 67 MB, but over a 30k-TR validation split it
    # is ~226 GB and the process is SIGKILLed with no traceback. The batch size
    # also has to match training's, since the statistic is batch-dependent and a
    # value computed at a different B is not comparable.
    def validate():
        rows, n = np.zeros(3), 0
        with torch.no_grad():
            for start in range(0, len(z_va) - batch_size + 1, batch_size):
                t = torch.from_numpy(np.ascontiguousarray(z_va[start:start + batch_size])).to(device)
                v = model(t[:, 0], t[:, 1])
                rows += [float(v["loss"]), float(v["pred_loss"]), float(v["sigreg_loss"])]
                n += 1
        if n == 0:
            raise ValueError(
                f"validation split has {len(z_va)} TRs, fewer than one batch of "
                f"{batch_size}; lower --batch-size or raise --val-frac")
        return dict(zip(("loss", "pred_loss", "sigreg_loss"), rows / n))

    probe = torch.from_numpy(np.ascontiguousarray(z_va[:batch_size])).to(device)
    history, best = [], {"val_loss": float("inf"), "epoch": 0,
                         "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(z_tr))
        sums, n = np.zeros(3), 0
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = z_tr[order[start:start + batch_size]]
            batch = augment_batch(batch, noise=noise, dropout=dropout, gain=gain, rng=rng)
            t = torch.from_numpy(np.ascontiguousarray(batch)).to(device)

            opt.zero_grad()
            res = model(t[:, 0], t[:, 1])
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sums += [res["loss"].item(), res["pred_loss"].item(), res["sigreg_loss"].item()]
            n += 1
        sched.step()

        model.eval()
        val = validate()
        share = 0.5 * (model.left_encoder.nonlinear_share(probe[:, 0])
                       + model.right_encoder.nonlinear_share(probe[:, 1]))

        row = {"epoch": epoch, "train_loss": sums[0] / max(n, 1),
               "train_pred": sums[1] / max(n, 1), "train_sigreg": sums[2] / max(n, 1),
               "val_loss": val["loss"], "val_pred": val["pred_loss"],
               "val_sigreg": val["sigreg_loss"], "nonlinear_share": share,
               "lr": sched.get_last_lr()[0]}
        history.append(row)

        if on_epoch is not None:
            on_epoch(epoch, model, row)

        if val["loss"] < best["val_loss"]:
            best = {"val_loss": val["loss"], "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"  [{epoch:>3}/{epochs}] train {row['train_loss']:.5f} "
                  f"(pred {row['train_pred']:.5f}, sigreg {row['train_sigreg']:.5f})  "
                  f"val {val['loss']:.5f}  nonlin {share:.3f}", flush=True)

    model.load_state_dict(best["state"])
    model.eval()
    if verbose:
        print(f"  best epoch {best['epoch']} (val {best['val_loss']:.5f})")
    return model, {"history": history, "best_epoch": best["epoch"],
                   "best_val_loss": best["val_loss"]}


# --------------------------------------------------------------------------
# numpy feature path
# --------------------------------------------------------------------------
def encode_numpy(weights, z):
    """Apply one exported encoder to ``z [N, M]`` or ``z [B, T, M]`` -> ``[N, k]`` or ``[B, T, k]``."""
    z = np.asarray(z)
    is_3d = (z.ndim == 3)
    if is_3d:
        B, T, M = z.shape
        z_2d = z.reshape(B * T, M)
    else:
        T, M = z.shape
        z_2d = z

    out_lin = z_2d @ weights["linear"]

    # Spatial MLP path
    h = z_2d
    for layer in weights.get("layers", []):
        if layer[0] == "linear":
            h = h @ layer[1] + (0.0 if layer[2] is None else layer[2])
        elif layer[0] == "layernorm":
            mu = h.mean(axis=-1, keepdims=True)
            sd = np.sqrt(h.var(axis=-1, keepdims=True) + layer[3])
            h = (h - mu) / sd * layer[1] + layer[2]
        elif layer[0] == "gelu":
            h = gelu_numpy(h)
        elif layer[0] == "identity":
            pass
        else:
            raise ValueError(f"unknown exported layer {layer[0]!r}")

    a_spat = float(weights.get("alpha_spatial", 1.0))
    out = out_lin + a_spat * h

    # Spatiotemporal Conv1D path (if present)
    if "temp_conv1" in weights and weights["temp_conv1"] is not None:
        w1, b1 = weights["temp_conv1"]  # [hidden, M, K]
        norm_w, norm_b, norm_eps = weights["temp_norm"]
        w2, b2 = weights["temp_conv2"]  # [latent, hidden, 1]
        k_size = int(w1.shape[2])

        if is_3d:
            z_pad = np.pad(z, [(0, 0), (k_size - 1, 0), (0, 0)], mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(z_pad, (k_size, M), axis=(1, 2)).squeeze(2)  # [B, T, K, M]
            w1_perm = np.transpose(w1, (0, 2, 1))  # [hidden, K, M]
            c1 = np.tensordot(windows, w1_perm, axes=((2, 3), (1, 2)))  # [B, T, hidden]
            if b1 is not None:
                c1 = c1 + b1
            mu_c = c1.mean(axis=-1, keepdims=True)
            sd_c = np.sqrt(c1.var(axis=-1, keepdims=True) + norm_eps)
            c1 = (c1 - mu_c) / sd_c * norm_w + norm_b
            c1 = gelu_numpy(c1)
            w2_mat = w2.squeeze(-1).T  # [hidden, latent]
            c2 = c1 @ w2_mat
            if b2 is not None:
                c2 = c2 + b2
            c2_flat = c2.reshape(B * T, -1)
        else:
            z_pad = np.pad(z, [(k_size - 1, 0), (0, 0)], mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(z_pad, (k_size, M)).squeeze(1)  # [T, K, M]
            w1_perm = np.transpose(w1, (0, 2, 1))  # [hidden, K, M]
            c1 = np.tensordot(windows, w1_perm, axes=((1, 2), (1, 2)))  # [T, hidden]
            if b1 is not None:
                c1 = c1 + b1
            mu_c = c1.mean(axis=-1, keepdims=True)
            sd_c = np.sqrt(c1.var(axis=-1, keepdims=True) + norm_eps)
            c1 = (c1 - mu_c) / sd_c * norm_w + norm_b
            c1 = gelu_numpy(c1)
            w2_mat = w2.squeeze(-1).T  # [hidden, latent]
            c2 = c1 @ w2_mat
            if b2 is not None:
                c2 = c2 + b2
            c2_flat = c2

        a_temp = float(weights.get("alpha_temporal", 1.0))
        out = out + a_temp * c2_flat

    if is_3d:
        return out.reshape(B, T, -1)
    return out


def jepa_features(weights, rows, basis, m=256, head="avg", regress_motion=False):
    """Masked voxel rows ``[T, 14236]`` or ``[B, T, 14236]`` -> Orbit-JEPA features."""
    zl, zr = orbit_projections(rows, basis, m=m, regress_motion=regress_motion)
    s_l = encode_numpy(weights["left"], zl)
    s_r = encode_numpy(weights["right"], zr)
    if head == "avg":
        return 0.5 * (s_l + s_r)
    if head == "concat":
        return np.concatenate([s_l, s_r], axis=-1)
    raise ValueError(f"unknown head {head!r}")


def save_checkpoint(path, model, basis_path, m, head, regress_motion, meta=None):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": model.arch(),
                "numpy_weights": model.to_numpy_weights(), "basis": str(basis_path),
                "m": int(m), "head": head, "regress_motion": bool(regress_motion),
                "meta": meta or {}}, path)


def load_checkpoint(path, untrained=False):
    """Load a checkpoint, or build its untrained twin from its own ``arch``.

    ``untrained=True`` is the control, and it is derived from the stored
    architecture rather than from CLI defaults for the reason `CLAUDE.md` gives
    under the `xrot` entry: a control assembled from configuration drifts the
    next time a field is added, and it inflated a reported margin to +0.370
    once already.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("arch", "basis", "m", "head", "regress_motion"):
        if key not in ckpt:
            raise ValueError(f"checkpoint {path} is missing {key!r}; refusing to "
                             f"guess an architecture for it")
    model = OrbitJEPA(**ckpt["arch"])
    if not untrained:
        model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ckpt["model"] = model
    ckpt["weights"] = model.to_numpy_weights() if untrained else ckpt["numpy_weights"]
    return ckpt
