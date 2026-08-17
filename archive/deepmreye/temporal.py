"""Next-TR prediction: a causal (language-model-style) objective on eye blocks.

The unlabeled corpus has already failed to help through three *linear* bases
(``unsupervised.py``) and, on the ``pytorch-jepa`` branch, through masked
spatial prediction. What none of those tried is a **causal temporal** objective:
predict TR *t+1* from TRs <= *t*, and use the sequence model's hidden state as
the feature. Eye position is part of the state you would need in order to
predict the next volume, so in principle it has to be encoded.

Two measurements shape the design, both made before any model was written
(``predictability`` diagnostic):

1. **The next TR is genuinely predictable.** A linear AR(4) over the corpus-PCA
   coordinates reaches R^2 0.32 on held-out runs, against 0.13 for persistence
   and 0.0 for the mean. So there is temporal structure to learn -- this is not
   a vacuous objective.
2. **Almost all of that structure sits in the leading components.** Components
   0-8 hold 38% of the variance and are predicted at R^2 0.59; components
   128-256 are predicted at 0.09. The leading components of an eye-region block
   are global signal, motion and drift. An unweighted MSE would therefore spend
   nearly all its capacity on exactly the nuisance the probe does not want.

Hence ``whiten=True`` by default: targets are standardised per component, so
every direction contributes equally to the loss and the model cannot buy its
score purely by tracking the global signal. That is the single most important
knob here, and ``--no-whiten`` exists to show what happens without it.

The model runs on the corpus-PCA coordinates rather than on 14236 raw voxels.
The basis is fitted on the same unlabeled participants and is close to lossless
for the variance that exists, so this costs little and makes the whole thing a
few minutes on a laptop instead of a cluster job. It also keeps the comparison
clean: ``corpus-pca`` is exactly this model's *input*, so any gain is
attributable to the temporal model rather than to the projection.

**The control that matters is an untrained model of identical architecture**
(``ar-random``). On the JEPA branch a random encoder scored the same as every
trained one, and that is what showed the objective was doing nothing. Any claim
made from ``ar-gru`` has to clear ``ar-random`` first.
"""
import numpy as np

# Sequence length used for training crops. Long enough to carry several
# autocorrelation times at any TR in the corpus, short enough that a batch fits
# comfortably and the GRU does not have to backpropagate through a whole run.
CROP = 128


def _torch():
    import torch

    # LightGBM and PyTorch each load their own OpenMP runtime and deadlock when
    # a threaded torch op follows a LightGBM fit in one process (see
    # `evaluate/features.py`). The models here are small enough that
    # single-threaded costs nothing, and it keeps `--readouts lgbm` usable
    # alongside an `ar-*` feature source.
    torch.set_num_threads(1)
    return torch


def device_for(name="auto"):
    torch = _torch()
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def project_block(block, mask_flat, mean, components):
    """``[X, Y, Z, T]`` -> ``[T, k]`` corpus-PCA coordinates."""
    x = np.asarray(block).reshape(-1, np.asarray(block).shape[-1])[mask_flat].T
    return ((x.astype(np.float64) - mean) @ components).astype(np.float32)


def build_sequences(subjects, mask, basis, max_trs=None, progress=None):
    """Project every participant to PCA coordinates.

    Returns ``(data [N, k] float32, offsets [n+1])`` -- one flat array plus run
    boundaries, so crops never straddle two participants.
    """
    import h5py

    mask_flat = mask.reshape(-1)
    mean = basis["mean"].astype(np.float64)
    comps = basis["components"].astype(np.float64)

    chunks, offsets = [], [0]
    for i, (_ds, _sub, path, _n) in enumerate(subjects):
        try:
            with h5py.File(path, "r") as f:
                b = f["eye_block"]
                t = b.shape[-1] if not max_trs else min(b.shape[-1], max_trs)
                block = b[..., :t]
        except Exception:
            continue
        chunks.append(project_block(block, mask_flat, mean, comps))
        offsets.append(offsets[-1] + len(chunks[-1]))
        if progress and i % progress == 0:
            print(f"  [{i + 1}/{len(subjects)}] {offsets[-1]} TRs", flush=True)
    return np.concatenate(chunks), np.array(offsets)


class ARModel:
    """A causal GRU over PCA coordinates, predicting the next TR.

    Kept as a thin wrapper rather than an ``nn.Module`` subclass so that
    building, saving and loading go through one place and the untrained control
    is constructed by exactly the same code path as the trained model -- if the
    control were built differently, it would not be a control.
    """

    def __init__(self, n_in, hidden=256, layers=1, seed=0, device=None):
        torch = _torch()
        import torch.nn as nn

        torch.manual_seed(seed)
        self.n_in, self.hidden, self.layers = n_in, hidden, layers
        self.device = device or device_for()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True).to(self.device)
        self.head = nn.Linear(hidden, n_in).to(self.device)

    def parameters(self):
        return list(self.gru.parameters()) + list(self.head.parameters())

    def forward(self, x):
        """``[B, T, k]`` -> ``(hidden [B, T, H], prediction of x[:, t+1])``."""
        h, _ = self.gru(x)
        return h, self.head(h)

    def state_dict(self):
        return {"gru": self.gru.state_dict(), "head": self.head.state_dict(),
                "n_in": self.n_in, "hidden": self.hidden, "layers": self.layers}

    def load_state_dict(self, sd):
        self.gru.load_state_dict(sd["gru"])
        self.head.load_state_dict(sd["head"])

    def eval(self):
        self.gru.eval()
        self.head.eval()
        return self


def crops(data, offsets, length, batch, rng, n_batches):
    """Random within-run crops. Yields ``[B, length, k]`` float32."""
    runs = [(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)
            if offsets[i + 1] - offsets[i] > length + 1]
    if not runs:
        raise RuntimeError("no run is long enough to crop")
    for _ in range(n_batches):
        out = np.empty((batch, length + 1, data.shape[1]), dtype=np.float32)
        for b in range(batch):
            lo, hi = runs[rng.integers(len(runs))]
            s = lo + rng.integers(hi - lo - length - 1)
            out[b] = data[s: s + length + 1]
        yield out


def evaluate_prediction(model, data, offsets, scale, length, batch, seed, n_batches=40):
    """Held-out next-TR R^2, against persistence and the mean.

    R^2 is computed on the *whitened* targets, i.e. per component and then
    pooled, so it is not dominated by the handful of leading components that
    carry most of the raw variance.
    """
    torch = _torch()
    rng = np.random.default_rng(seed)
    num = den = num_p = 0.0
    with torch.no_grad():
        for batch_np in crops(data, offsets, length, batch, rng, n_batches):
            w = torch.from_numpy(batch_np).to(model.device) / scale
            x, y = w[:, :-1], w[:, 1:]
            _h, pred = model.forward(x)
            num += float(((y - pred) ** 2).sum())
            num_p += float(((y - x) ** 2).sum())      # persistence
            den += float(((y - y.mean(dim=(0, 1))) ** 2).sum())
    return {"r2": 1 - num / den, "r2_persistence": 1 - num_p / den}


def train(model, data, offsets, val_data, val_offsets, scale, steps=3000,
          batch=64, length=CROP, lr=1e-3, seed=0, log_every=250):
    """Train next-TR prediction; returns ``(history, best_state)``.

    Validation R^2 peaks early and then drifts down as the model starts fitting
    run-specific structure, so the *best* state is kept rather than the last.
    Probing a checkpoint that is past its own optimum would understate the
    objective, which is the one thing this experiment must not do.
    """
    import copy

    torch = _torch()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    best = {"r2": -np.inf, "step": 0, "state": None}
    scale_t = (scale if torch.is_tensor(scale)
               else torch.from_numpy(scale).to(model.device))

    for step, batch_np in enumerate(
            crops(data, offsets, length, batch, rng, steps), start=1):
        w = torch.from_numpy(batch_np).to(model.device) / scale_t
        x, y = w[:, :-1], w[:, 1:]
        _h, pred = model.forward(x)
        loss = ((y - pred) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == steps:
            model.gru.eval(); model.head.eval()
            m = evaluate_prediction(model, val_data, val_offsets, scale_t,
                                    length, batch, seed + 1)
            model.gru.train(); model.head.train()
            m["step"] = step
            m["train_loss"] = float(loss.detach())
            history.append(m)
            marker = ""
            if m["r2"] > best["r2"]:
                best = {"r2": m["r2"], "step": step,
                        "state": copy.deepcopy(model.state_dict())}
                marker = "  *"
            print(f"  step {step:>5}  train {float(loss):.4f}  "
                  f"val R2 {m['r2']:+.4f}  (persistence {m['r2_persistence']:+.4f})"
                  f"{marker}", flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"  restored best checkpoint from step {best['step']} "
              f"(val R2 {best['r2']:+.4f})")
    return history, best


def save(path, model, scale, meta):
    from pathlib import Path

    torch = _torch()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "scale": scale, "meta": meta}, path)
    return path


def load(path, device=None):
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model"]
    model = ARModel(sd["n_in"], sd["hidden"], sd["layers"], device=device)
    model.load_state_dict(sd)
    return model.eval(), blob["scale"], blob["meta"]
