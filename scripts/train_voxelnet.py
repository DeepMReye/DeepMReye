#!/usr/bin/env python3
"""Voxel-level gaze network warm-started at the linear incumbent.

The claim being tested: a network is a universal function approximator, so it should be at
least as good as `lr-cca:32 + lags` read straight off voxels. That is true about
*representation* and says nothing about optimisation or generalisation -- so the network is
**initialised at** the incumbent rather than asked to rediscover it. Concretely, per fold:

  1. Fit the incumbent (`RidgeCV` on `make_lags(z_cca, L)`) exactly as `lodo_subtr` does.
  2. Freeze it as `y_lin`, precomputed for every row.
  3. Train a **supervised learned projection from voxels** to predict the *residual*
     `y_target - y_lin`, gated by a ReZero `alpha` initialised at 0.
  4. Early-stop on held-out *participants* inside the training pool.

At step 0 the prediction is exactly the incumbent, so the fold score starts at the incumbent's
and the learned branch can only be adopted if it earns held-out loss. Learning the residual
rather than the whole map is what makes this affordable: the frozen branch is a fixed linear
function of a precomputed projection, so it never has to be recomputed.

This is the arm `analyze_temporal_ceiling_supervised.py` did **not** test. That gate varied the
rank of a *frozen, unsupervised* projection (`lr-cca` fitted blind to gaze) and found more of
it harmful. A projection fitted *against gaze* is a different object, and it is the only route
by which a voxel model could beat the canonical span.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepmreye.temporal_probe import (  # noqa: E402
    ALPHAS, MAX_TRAIN_ROWS, fold_median, make_lags, subject_scores,
)
from deepmreye.voxelnet import (  # noqa: E402
    build_net, cca_matrix, load_voxel_cache, make_lags_torch, mixup, shift_augment,
)


def device_for(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def zscore_targets(parts, lab, train_idx_by_ds):
    """Per-training-dataset target z-scoring, the `--standardize-targets dataset` rule."""
    stats = {}
    for ds, rows in train_idx_by_ds.items():
        y = lab[rows].reshape(len(rows), 20)
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
    p.add_argument("--lags", type=int, default=1, help="1 is the measured optimum at sub-TR.")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--encoder", default="lowrank", choices=("lowrank", "cnn"))
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps-per-epoch", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val-datasets", type=int, default=2)
    p.add_argument("--adopt-margin", type=float, default=0.005)
    p.add_argument("--val-subjects", type=int, default=8,
                   help="Participants of the held-out selection dataset to score.")
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian SD on voxels (z-units).")
    p.add_argument("--vox-dropout", type=float, default=0.0)
    p.add_argument("--shift", type=int, default=0,
                   help="Max integer-voxel translation (DeepMReye 1.0 used +-4).")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="Beta parameter; 0 disables.")
    p.add_argument("--folds", nargs="*", default=None)
    p.add_argument("--init-encoder", default=None,
                   help="Self-supervised checkpoint from pretrain_voxelnet.py.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/subtr/voxelnet.json")
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
    datasets = sorted({p_["dataset"] for p_ in parts})
    print(f"[*] {len(parts)} participants, {len(datasets)} datasets, "
          f"{meta['n_rows']} TRs, device {dev}", flush=True)

    # z_cca for every row, once: the projection is frozen, so it never changes across folds.
    zc = Path(args.voxels) / f"z_cca_k{args.k}.npy"
    if zc.exists():
        z_all = np.load(zc)
    else:
        print("[*] precomputing frozen canonical coordinates", flush=True)
        z_all = np.empty((meta["n_rows"], args.k), dtype=np.float32)
        step = 20000
        for i in range(0, meta["n_rows"], step):
            j = min(i + step, meta["n_rows"])
            z_all[i:j] = ((vox[i:j].astype(np.float32) - mu.astype(np.float32))
                          @ w_cca.astype(np.float32))
        np.save(zc, z_all)
    print(f"[*] z_cca {z_all.shape}", flush=True)

    mask_idx = np.flatnonzero(mask.reshape(-1))
    folds = args.folds or datasets
    results = {}

    for held in folds:
        t_fold = time.time()
        train_parts = [p_ for p_ in parts if p_["dataset"] != held]
        test_parts = [p_ for p_ in parts if p_["dataset"] == held]

        # ---- the incumbent, fitted exactly as lodo_subtr fits it ----------------------
        xs, ys = [], []
        for ds in sorted({p_["dataset"] for p_ in train_parts}):
            members = [p_ for p_ in train_parts if p_["dataset"] == ds]
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
            pred = ridge.predict(make_lags(z_all[sl].astype(np.float64), args.lags))
            base.append(subject_scores(pred, np.asarray(lab[sl]))[0])
        base_r = fold_median(base)

        # ---- per-dataset target statistics, for training the residual -----------------
        stats = zscore_targets(train_parts, lab,
                               {ds: np.concatenate([np.arange(m["start"], m["start"] + m["n"])
                                                    for m in train_parts if m["dataset"] == ds])
                                for ds in sorted({p_["dataset"] for p_ in train_parts})})

        # ---- a held-out training DATASET for selection, scored by the real metric ------
        #
        # Two corrections to the obvious design, both of which cost accuracy when wrong:
        #
        # 1. Validate on a held-out *dataset*, not on held-out participants of the training
        #    datasets. The test is cross-dataset, so participant-level validation measures
        #    within-distribution fit and happily selects a branch that has learned
        #    dataset-specific structure. Measured with the participant split: dsL01 went
        #    0.7678 -> 0.7263 while its validation loss was still improving.
        # 2. Select on **sub-TR r**, the quantity actually reported, not on residual MSE.
        #    They are not the same objective: a branch can absorb residual variance (helping
        #    MSE) while adding noise that lowers a correlation.
        # Two selection datasets, not one: the median of a single dataset's participants is
        # a noisy signal to stop on, and whichever dataset is drawn would otherwise set the
        # scale of the decision (dsL08 sits at 0.36 where dsL02 sits at 0.92).
        tr_ds = sorted({p_["dataset"] for p_ in train_parts})
        pick = np.random.default_rng(args.seed).permutation(len(tr_ds))[:args.val_datasets]
        val_ds = sorted(tr_ds[i] for i in pick)
        val_parts = [q for d in val_ds
                     for q in [p_ for p_ in train_parts if p_["dataset"] == d][:args.val_subjects]]
        fit_parts = [p_ for p_ in train_parts if p_["dataset"] not in set(val_ds)]
        print(f"    selection datasets: {', '.join(val_ds)} "
              f"({len(val_parts)} participants)", flush=True)

        net = build_net(w_cca, mu, lags=args.lags, encoder=args.encoder, rank=args.rank,
                        width=args.width, dropout=args.dropout,
                        grid_shape=tuple(meta["mask_shape"]), mask_idx=mask_idx,
                        seed=args.seed).to(dev)
        net.mask_idx_aug = torch.as_tensor(mask_idx, dtype=torch.long, device=dev)
        if args.init_encoder:
            # Self-supervised initialisation of the learned branch. `head_nl` stays zero, so
            # the network is STILL exactly the incumbent at step 0 -- pretraining changes
            # where the branch starts searching, not what the guarantee is worth. The arch
            # is checked against the checkpoint's own record rather than the CLI, the same
            # rule the *-random controls follow.
            ck = torch.load(args.init_encoder, map_location="cpu", weights_only=False)
            want = {"encoder": args.encoder, "rank": args.rank, "width": args.width}
            got = {k: ck["arch"][k] for k in want}
            if got != want:
                raise SystemExit(f"[!] pretrained encoder is {got}, this run is {want}")
            missing = net.load_state_dict(ck["encoder"], strict=False)
            print(f"    initialised encoder from {args.init_encoder} "
                  f"({len(ck['encoder'])} tensors)", flush=True)
            del missing
        # head_lin is warm-started but never used in the residual path; the incumbent's
        # prediction enters as `y_lin`, precomputed. Keeping the branch present (and equal to
        # the incumbent) is what makes `alpha = 0` mean "the incumbent, exactly".
        with torch.no_grad():
            net.head_lin.weight.copy_(torch.as_tensor(ridge.coef_, dtype=torch.float32))
            net.head_lin.bias.copy_(torch.as_tensor(ridge.intercept_, dtype=torch.float32))

        params = [p_ for n_, p_ in net.named_parameters() if not n_.startswith("head_lin")]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        # Only participants long enough for a full chunk, so the batch stacks and mixup can
        # cross participants (and therefore datasets) within one tensor.
        long_fit = [m for m in fit_parts if m["n"] > args.chunk] or fit_parts

        def sample(parts_list, gen):
            xs_, rs_ = [], []
            for _ in range(args.batch_chunks):
                m = parts_list[int(gen.integers(len(parts_list)))]
                off = int(gen.integers(max(1, m["n"] - args.chunk)))
                n_ = min(args.chunk, m["n"] - off)
                sl = slice(m["start"] + off, m["start"] + off + n_)
                y = np.asarray(lab[sl]).reshape(n_, 20)
                mn, sd = stats.get(m["dataset"], (np.zeros(20), np.ones(20)))
                yl = ridge.predict(make_lags(z_all[sl].astype(np.float64), args.lags))
                if n_ < args.chunk:
                    continue
                xs_.append(vox[sl].astype(np.float32))
                rs_.append(((y - mn) / sd) - yl)
            if not xs_:
                return None, None
            return np.stack(xs_), np.stack(rs_)

        def run_batch(x, resid, train=True, gen=None):
            if x is None:
                return 0.0
            xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
            tgt = torch.as_tensor(resid, dtype=torch.float32, device=dev)
            if train:
                if args.shift > 0:
                    xt = shift_augment(xt, net.mask_idx_aug, tuple(meta["mask_shape"]),
                                       args.shift, gen)
                if args.mixup > 0:
                    xt, tgt = mixup(xt, tgt, gen, args.mixup)
                if args.noise > 0:
                    xt = xt + args.noise * torch.randn_like(xt)
                if args.vox_dropout > 0:
                    keep = (torch.rand_like(xt) > args.vox_dropout).float()
                    xt = xt * keep / max(1e-6, 1 - args.vox_dropout)
            h = net.drop(net.encode(xt)) if train else net.encode(xt)
            pred = net.head_nl(make_lags_torch(h, net.lags)) * net.alpha.clamp(-1, 1)
            ok = torch.isfinite(tgt).all(dim=-1)
            if ok.sum() < 5:
                return 0.0
            loss = torch.nn.functional.mse_loss(pred[ok], tgt[ok])
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            return float(loss.detach())

        def score_parts(part_list, cap=4000):
            """Median sub-TR r over a set of participants, the reported metric."""
            out = []
            with torch.no_grad():
                for m in part_list:
                    n_ = min(m["n"], cap)
                    sl = slice(m["start"], m["start"] + n_)
                    yl = ridge.predict(make_lags(z_all[sl].astype(np.float64), args.lags))
                    outs = []
                    for i in range(0, n_, 512):
                        xt = torch.as_tensor(vox[sl][i:i + 512].astype(np.float32),
                                             device=dev)[None]
                        h = net.encode(xt)
                        outs.append((net.head_nl(make_lags_torch(h, net.lags))
                                     * net.alpha.clamp(-1, 1)).cpu().numpy()[0])
                    a, _ = subject_scores(yl + np.concatenate(outs), np.asarray(lab[sl]))
                    if np.isfinite(a):
                        out.append(a)
            return fold_median(out)

        gen = np.random.default_rng(args.seed)
        history = []
        net.eval()
        best_val = score_parts(val_parts)
        best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        print(f"    ep -1 selection r {best_val:.4f} (== incumbent by construction)", flush=True)
        bad = 0
        for ep in range(args.epochs):
            net.train()
            tr_loss = np.mean([run_batch(*sample(long_fit, gen), train=True, gen=gen)
                               for _ in range(args.steps_per_epoch)])
            net.eval()
            v = score_parts(val_parts)
            history.append({"epoch": ep, "train": float(tr_loss), "val_r": float(v),
                            "alpha": net.nonlinear_share()})
            print(f"    ep{ep:>3} train {tr_loss:.4f} selection r {v:.4f} "
                  f"alpha {net.nonlinear_share():.4f}", flush=True)
            # An adoption *margin*, not any improvement at all. With `1e-5` the branch is
            # taken whenever selection wobbles upward, and selection is itself a median over a
            # handful of participants: on the dsL08 fold that adopted a branch which cost the
            # test fold 0.13. The branch has to clear real ground on held-out datasets before
            # it displaces a warm start that is known-good.
            if v > best_val + args.adopt_margin:
                best_val, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        # Restores the best-selection state, which at worst is the initial one -- i.e. the
        # incumbent exactly. This is what makes "at least as good" structural rather than hoped
        # for: the branch is adopted only if it improved the metric on a held-out dataset.
        net.load_state_dict(best_state)

        # ---- score the fold ----------------------------------------------------------
        net.eval()
        r_sub, r_1tr = [], []
        with torch.no_grad():
            for m in test_parts:
                sl = slice(m["start"], m["start"] + m["n"])
                yl = ridge.predict(make_lags(z_all[sl].astype(np.float64), args.lags))
                outs = []
                for i in range(0, m["n"], 512):
                    xt = torch.as_tensor(vox[sl][i:i + 512].astype(np.float32), device=dev)[None]
                    h = net.encode(xt)
                    outs.append((net.head_nl(make_lags_torch(h, net.lags))
                                 * net.alpha.clamp(-1, 1)).cpu().numpy()[0])
                pred = yl + np.concatenate(outs)
                a, b = subject_scores(pred, np.asarray(lab[sl]))
                if np.isfinite(a):
                    r_sub.append(a)
                if np.isfinite(b):
                    r_1tr.append(b)

        got = fold_median(r_sub)
        results[held] = {"incumbent": base_r, "net": got, "net_1tr": fold_median(r_1tr),
                         "alpha": net.nonlinear_share(), "best_val": float(best_val),
                         "history": history, "n_test": len(test_parts)}
        print(f"[{held}] incumbent {base_r:.4f} -> net {got:.4f} "
              f"({got - base_r:+.4f})  alpha {net.nonlinear_share():.4f}  "
              f"({time.time() - t_fold:.0f}s)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "results": results}, indent=1))

    inc = fold_median([v["incumbent"] for v in results.values()])
    net_r = fold_median([v["net"] for v in results.values()])
    won = sum(1 for v in results.values() if v["net"] > v["incumbent"])
    print(f"\n{'=' * 72}\nincumbent {inc:.4f}   net {net_r:.4f}   "
          f"{net_r - inc:+.4f}   folds won {won}/{len(results)}\n{'=' * 72}")
    print(f"[+] {args.out}")


if __name__ == "__main__":
    main()
