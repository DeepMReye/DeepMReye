"""Voxels -> gaze, trained from scratch. No warm start, no incumbent in the graph.

The warm-started arm (`train_voxelnet.py`) puts a zero-initialised branch on top of a frozen
RidgeCV head, so the network can only fit what ridge left over. That guarantees "at least as
good as the incumbent" and buys it by construction -- but it also fixes the representation
the head reads from at step 0, so gradient descent never reorganises the whole map. A branch
that can only decorate a good linear solution is not a test of whether a network can decode
gaze from voxels.

This script is that test. The model sees voxels and predicts the 20 sub-TR gaze coordinates.
The incumbent is computed per fold and printed, but it is a **reference line on the plot**,
never a term in the loss and never added to a prediction.

    pred(x) = head( make_lags( g(x), L ) )        # g = 3-D conv or low-rank linear

Protocol is otherwise identical to `deepmreye/temporal_probe.lodo_subtr`: leave one dataset
out, per-training-dataset target z-scoring, selection on held-out training *datasets* scored
by the reported metric (sub-TR r), median over test participants.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepmreye.temporal_probe import (ALPHAS, MAX_TRAIN_ROWS, fold_median, make_lags,
                                      subject_scores)
from deepmreye.voxelnet import (cca_matrix, load_voxel_cache, make_lags_torch, mixup,
                                shift_augment)


def device_for(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_scratch_net(n_vox, grid_shape, mask_idx, lags=1, encoder="cnn", rank=64, width=16,
                      dropout=0.2, hidden=256, seed=0):
    """Encoder over one TR, a lag stack over time, then a dense head to 20 outputs."""
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class ScratchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lags = int(lags)
            self.encoder_kind = encoder
            self.rank = int(rank)
            n_lag = 2 * self.lags + 1

            if encoder == "lowrank":
                self.enc = nn.Linear(n_vox, self.rank, bias=False)
            elif encoder == "cnn":
                self.register_buffer("mask_idx", torch.as_tensor(mask_idx, dtype=torch.long))
                self.grid_shape = tuple(grid_shape)
                # No global pooling: gaze IS the eyeball's spatial position, so pooling the
                # spatial axes away deletes the signal. Flatten instead, as DeepMReye 1.0 does.
                self.conv = nn.Sequential(
                    nn.Conv3d(1, width, 3, stride=2, padding=1),
                    nn.GroupNorm(4, width), nn.GELU(),
                    nn.Conv3d(width, 2 * width, 3, stride=2, padding=1),
                    nn.GroupNorm(4, 2 * width), nn.GELU(),
                    nn.Conv3d(2 * width, 4 * width, 3, stride=2, padding=1),
                    nn.GroupNorm(4, 4 * width), nn.GELU(),
                    nn.Flatten())
                with torch.no_grad():
                    n_feat = self.conv(torch.zeros(1, 1, *tuple(grid_shape))).shape[1]
                self.enc = nn.Linear(n_feat, self.rank)
            else:
                raise ValueError(encoder)

            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(self.rank * n_lag, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 20))

        def encode(self, x):
            if self.encoder_kind == "lowrank":
                return self.enc(x)
            b, t_n, _ = x.shape
            grid = x.new_zeros(b * t_n, int(np.prod(self.grid_shape)))
            grid[:, self.mask_idx] = x.reshape(b * t_n, -1)
            return self.enc(self.conv(grid.view(b * t_n, 1, *self.grid_shape))).view(b, t_n, -1)

        def forward(self, x):
            return self.head(make_lags_torch(self.drop(self.encode(x)), self.lags))

        def arch(self):
            return {"lags": self.lags, "encoder": self.encoder_kind, "rank": self.rank,
                    "width": int(width), "hidden": int(hidden), "dropout": float(dropout)}

    return ScratchNet()


def zscore_targets(parts, lab, train_idx_by_ds):
    stats = {}
    for ds, rows in train_idx_by_ds.items():
        y = np.asarray(lab[rows]).reshape(len(rows), 20)
        ok = np.isfinite(y).all(axis=1)
        if ok.sum() < 10:
            continue
        y = y[ok]
        sd = y.std(axis=0)
        sd[sd < 1e-9] = 1.0
        stats[ds] = (y.mean(axis=0), sd)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", default="results/subtr/voxels")
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("--k", type=int, default=32, help="Only for the printed incumbent reference.")
    p.add_argument("--encoder", default="cnn", choices=("cnn", "lowrank"))
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--batch-chunks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps-per-epoch", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--cosine", action="store_true", help="Cosine-decay lr over --epochs.")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-datasets", type=int, default=3)
    p.add_argument("--val-subjects", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--vox-dropout", type=float, default=0.0)
    p.add_argument("--shift", type=int, default=0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--folds", nargs="*", default=None)
    p.add_argument("--fold-index", type=int, default=None,
                   help="Run only the Nth dataset (0-based, sorted). For SLURM array tasks: "
                        "--fold-index $SLURM_ARRAY_TASK_ID. Mutually exclusive with --folds.")
    p.add_argument("--init-encoder", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="", help="Free-text label for this run, stored in the "
                   "result JSON and surfaced by scripts/summarize_voxelnet_trials.py.")
    p.add_argument("--out", default="results/subtr/voxelnet_scratch.json")
    args = p.parse_args()

    import torch
    from sklearn.linear_model import RidgeCV

    from deepmreye.datasource import resolve
    from deepmreye.unsupervised import corpus_mask, load_basis

    dev = device_for(args.device)
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    _m, bases, _meta = load_basis(Path(args.basis))
    w_cca, mu = cca_matrix(bases["lr-cca"], k=args.k)

    vox, lab, meta = load_voxel_cache(args.voxels, mask)
    parts = meta["parts"]
    datasets = sorted({q["dataset"] for q in parts})
    print(f"[*] {len(parts)} participants, {len(datasets)} datasets, "
          f"{meta['n_rows']} TRs, device {dev}", flush=True)

    z_all = np.load(Path(args.voxels) / f"z_cca_k{args.k}.npy")
    mask_idx = np.flatnonzero(mask.reshape(-1))
    if args.fold_index is not None:
        if args.folds:
            raise SystemExit("[!] pass --folds or --fold-index, not both")
        if not 0 <= args.fold_index < len(datasets):
            raise SystemExit(f"[!] --fold-index {args.fold_index} outside 0..{len(datasets)-1}")
        folds = [datasets[args.fold_index]]
    else:
        folds = args.folds or datasets
    results = {}

    for held in folds:
        t_fold = time.time()
        train_parts = [q for q in parts if q["dataset"] != held]
        test_parts = [q for q in parts if q["dataset"] == held]

        # ---- incumbent, for reference only -------------------------------------------
        xs, ys = [], []
        for ds in sorted({q["dataset"] for q in train_parts}):
            members = [q for q in train_parts if q["dataset"] == ds]
            x = np.concatenate([make_lags(z_all[m["start"]:m["start"] + m["n"]].astype(np.float64),
                                          args.lags) for m in members])
            y = np.concatenate([lab[m["start"]:m["start"] + m["n"]].reshape(m["n"], 20)
                                for m in members])
            ok = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            x, y = x[ok], y[ok]
            sd = y.std(axis=0)
            sd[sd < 1e-9] = 1.0
            xs.append(x)
            ys.append((y - y.mean(axis=0)) / sd)
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > MAX_TRAIN_ROWS:
            idx = np.random.default_rng(args.seed).choice(len(x_tr), MAX_TRAIN_ROWS, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]
        ridge = RidgeCV(alphas=ALPHAS).fit(x_tr, y_tr)
        base = []
        for m in test_parts:
            sl = slice(m["start"], m["start"] + m["n"])
            base.append(subject_scores(ridge.predict(
                make_lags(z_all[sl].astype(np.float64), args.lags)), np.asarray(lab[sl]))[0])
        base_r = fold_median(base)

        stats = zscore_targets(train_parts, lab,
                               {ds: np.concatenate([np.arange(m["start"], m["start"] + m["n"])
                                                    for m in train_parts if m["dataset"] == ds])
                                for ds in sorted({q["dataset"] for q in train_parts})})

        tr_ds = sorted({q["dataset"] for q in train_parts})
        pick = np.random.default_rng(args.seed).permutation(len(tr_ds))[:args.val_datasets]
        val_ds = sorted(tr_ds[i] for i in pick)
        val_parts = [q for d in val_ds
                     for q in [r for r in train_parts if r["dataset"] == d][:args.val_subjects]]
        fit_parts = [q for q in train_parts if q["dataset"] not in set(val_ds)]
        print(f"[{held}] incumbent {base_r:.4f} | selection: {', '.join(val_ds)} "
              f"({len(val_parts)} participants) | fit pool {len(fit_parts)}", flush=True)

        net = build_scratch_net(int(mask.sum()), tuple(meta["mask_shape"]), mask_idx,
                                lags=args.lags, encoder=args.encoder, rank=args.rank,
                                width=args.width, dropout=args.dropout, hidden=args.hidden,
                                seed=args.seed).to(dev)
        net.mask_idx_aug = torch.as_tensor(mask_idx, dtype=torch.long, device=dev)
        if args.init_encoder:
            ck = torch.load(args.init_encoder, map_location="cpu", weights_only=False)
            net.load_state_dict(ck["encoder"], strict=False)
            print(f"    encoder initialised from {args.init_encoder}", flush=True)

        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
                 if args.cosine else None)

        long_fit = [m for m in fit_parts if m["n"] > args.chunk] or fit_parts

        def sample(gen):
            xs_, ys_ = [], []
            for _ in range(args.batch_chunks):
                m = long_fit[int(gen.integers(len(long_fit)))]
                off = int(gen.integers(max(1, m["n"] - args.chunk)))
                n_ = min(args.chunk, m["n"] - off)
                if n_ < args.chunk:
                    continue
                sl = slice(m["start"] + off, m["start"] + off + n_)
                y = np.asarray(lab[sl]).reshape(n_, 20)
                mn, sd = stats.get(m["dataset"], (np.zeros(20), np.ones(20)))
                xs_.append(vox[sl].astype(np.float32))
                ys_.append((y - mn) / sd)          # the FULL target, not a residual
            if not xs_:
                return None, None
            return np.stack(xs_), np.stack(ys_)

        def run_batch(x, y, gen):
            if x is None:
                return None
            xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
            tgt = torch.as_tensor(y, dtype=torch.float32, device=dev)
            if args.shift > 0:
                xt = shift_augment(xt, net.mask_idx_aug, tuple(meta["mask_shape"]),
                                   args.shift, gen, per_sample=True)
            if args.mixup > 0:
                xt, tgt = mixup(xt, tgt, gen, args.mixup)
            if args.noise > 0:
                xt = xt + args.noise * torch.randn_like(xt)
            if args.vox_dropout > 0:
                keep = (torch.rand_like(xt) > args.vox_dropout).float()
                xt = xt * keep / max(1e-6, 1 - args.vox_dropout)
            pred = net(xt)
            ok = torch.isfinite(tgt).all(dim=-1)
            if ok.sum() < 5:
                return None
            loss = torch.nn.functional.mse_loss(pred[ok], tgt[ok])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            return float(loss.detach())

        def score_parts(part_list, cap=4000):
            out = []
            with torch.no_grad():
                for m in part_list:
                    n_ = min(m["n"], cap)
                    sl = slice(m["start"], m["start"] + n_)
                    outs = []
                    for i in range(0, n_, 512):
                        xt = torch.as_tensor(vox[sl][i:i + 512].astype(np.float32),
                                             device=dev)[None]
                        outs.append(net(xt).cpu().numpy()[0])
                    a, _ = subject_scores(np.concatenate(outs), np.asarray(lab[sl]))
                    if np.isfinite(a):
                        out.append(a)
            return fold_median(out)

        gen = np.random.default_rng(args.seed)
        history, bad = [], 0
        best_val, best_state = -np.inf, None
        # A handful of TRAINING participants scored the same way. Without this, a flat
        # validation curve cannot be told apart from a model that is not fitting at all --
        # the first is a generalisation failure, the second is an optimisation failure, and
        # they call for opposite fixes.
        probe_fit = long_fit[:6]
        for ep in range(args.epochs):
            net.train()
            losses = [run_batch(*sample(gen), gen) for _ in range(args.steps_per_epoch)]
            losses = [q for q in losses if q is not None]
            tr_loss = float(np.mean(losses)) if losses else float("nan")
            net.eval()
            v = score_parts(val_parts)
            f = score_parts(probe_fit, cap=2000)
            if sched:
                sched.step()
            history.append({"epoch": ep, "train": tr_loss, "val_r": float(v),
                            "fit_r": float(f), "lr": opt.param_groups[0]["lr"]})
            print(f"    ep{ep:>3} loss {tr_loss:.4f}  fit r {f:.4f}  val r {v:.4f}"
                  f"  (best {max(best_val, -1):.4f})", flush=True)
            if v > best_val + 1e-4:
                best_val, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"    early stop at epoch {ep}", flush=True)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        r_sub, r_1tr = [], []
        with torch.no_grad():
            for m in test_parts:
                sl = slice(m["start"], m["start"] + m["n"])
                outs = []
                for i in range(0, m["n"], 512):
                    xt = torch.as_tensor(vox[sl][i:i + 512].astype(np.float32), device=dev)[None]
                    outs.append(net(xt).cpu().numpy()[0])
                a, b = subject_scores(np.concatenate(outs), np.asarray(lab[sl]))
                if np.isfinite(a):
                    r_sub.append(a)
                if np.isfinite(b):
                    r_1tr.append(b)
        got = fold_median(r_sub)
        results[held] = {"incumbent": base_r, "net": got, "net_1tr": fold_median(r_1tr),
                         "best_val": float(best_val), "history": history,
                         "n_test": len(test_parts)}
        print(f"[{held}] incumbent {base_r:.4f} -> scratch net {got:.4f} "
              f"({got - base_r:+.4f})  ({time.time() - t_fold:.0f}s)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "results": results}, indent=1))

    inc = fold_median([v["incumbent"] for v in results.values()])
    net_r = fold_median([v["net"] for v in results.values()])
    won = sum(v["net"] > v["incumbent"] for v in results.values())
    print(f"\nmedian incumbent {inc:.4f}  scratch net {net_r:.4f}  "
          f"({net_r - inc:+.4f})  folds won {won}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
