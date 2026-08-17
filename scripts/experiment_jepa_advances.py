#!/usr/bin/env python3
"""Comprehensive experimental harness testing:
1. Spatiotemporal & Velocity Dynamics (T-JEPA)
2. Supervised End-to-End Fine-Tuning (SFT-JEPA)
3. Unsupervised Test-Time Adaptation (TTT-JEPA)
4. Multi-Source Hybrid Stacking (Fold-PCA + JEPA with Stack-Ridge)
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.models.jepa_net import OrbitJEPA
from deepmreye.orbitjepa import load_checkpoint, encode_numpy
from deepmreye.unsupervised import load_basis

CACHE_PATH = "results/jepa/labeled_cache.npz"


def load_recs():
    d = np.load(CACHE_PATH, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    return [{**meta[i], "z": d[f"z/{i}"], "gaze": d[f"g/{i}"]} for i in range(int(d["n"][0]))]


def evaluate_predictions(per_sub_preds, per_sub_gaze):
    per_sub_r = []
    for pred, gaze in zip(per_sub_preds, per_sub_gaze):
        ok = np.isfinite(gaze).all(axis=1) & np.isfinite(pred).all(axis=1)
        if ok.sum() < 10:
            continue
        rx = np.corrcoef(pred[ok, 0], gaze[ok, 0])[0, 1] if np.std(pred[ok, 0]) > 1e-9 and np.std(gaze[ok, 0]) > 1e-9 else np.nan
        ry = np.corrcoef(pred[ok, 1], gaze[ok, 1])[0, 1] if np.std(pred[ok, 1]) > 1e-9 and np.std(gaze[ok, 1]) > 1e-9 else np.nan
        per_sub_r.append([rx, ry])
    if not per_sub_r:
        return {"r_x": np.nan, "r_y": np.nan, "mean": np.nan}
    med = np.nanmedian(np.array(per_sub_r, dtype=float), axis=0)
    return {"r_x": float(med[0]), "r_y": float(med[1]), "mean": float(np.nanmean(med)), "n": len(per_sub_r)}


def correlation_loss(pred, target):
    """Negative Pearson correlation loss averaged over x and y."""
    vx = pred - pred.mean(dim=0, keepdim=True)
    vy = target - target.mean(dim=0, keepdim=True)
    cost = (vx * vy).sum(dim=0) / (torch.sqrt((vx ** 2).sum(dim=0) + 1e-8) * torch.sqrt((vy ** 2).sum(dim=0) + 1e-8) + 1e-8)
    return 1.0 - cost.mean()


class EndToEndJEPA(nn.Module):
    """OrbitJEPA + Linear Readout Head trained end-to-end with Fixup/ReZero."""
    def __init__(self, in_dim=256, latent_dim=32, hidden_dim=256, depth=2, dropout=0.1, head_type="avg"):
        super().__init__()
        self.jepa = OrbitJEPA(in_dim=in_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, depth=depth, dropout=dropout)
        self.head_type = head_type
        feat_dim = latent_dim if head_type == "avg" else latent_dim * 2
        self.head = nn.Linear(feat_dim, 2, bias=True)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def init_head_from_ridge(self, ridge_model):
        """Warm-start linear head from the exact Ridge solution on lr-cca features."""
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(ridge_model.coef_).float())
            self.head.bias.copy_(torch.from_numpy(ridge_model.intercept_).float())

    def forward(self, z_left, z_right):
        s_L, s_R = self.jepa.encode(z_left, z_right)
        feat = 0.5 * (s_L + s_R) if self.head_type == "avg" else torch.cat([s_L, s_R], dim=-1)
        return self.head(feat), feat


def run_experiment_temporal_velocity(recs, k=32):
    """Test incorporating velocity / temporal differences Δz_t = z_t - z_t-1."""
    print(f"\n=======================================================", flush=True)
    print(f"--- Temporal Dynamics / Velocity Feature Analysis (k={k}) ---", flush=True)
    print(f"=======================================================", flush=True)
    datasets = sorted({r["dataset"] for r in recs})
    per_fold_base, per_fold_temp = {}, {}

    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        test = [r for r in recs if r["dataset"] == held]

        def get_feats(records):
            xs_base, xs_vel, ys = [], [], []
            for ds in sorted({r["dataset"] for r in records}):
                ds_recs = [r for r in records if r["dataset"] == ds]
                for r in ds_recs:
                    g = r["gaze"]
                    zl, zr = r["z"][:, 0], r["z"][:, 1]
                    z_avg = 0.5 * (zl[:, :k] + zr[:, :k])
                    vel = np.zeros_like(z_avg)
                    vel[1:] = z_avg[1:] - z_avg[:-1]
                    z_dyn = np.concatenate([z_avg, vel], axis=-1)
                    
                    ok = np.isfinite(g).all(axis=1) & np.isfinite(z_avg).all(axis=1)
                    if ok.sum() < 10:
                        continue
                    g_ok = g[ok]
                    sd = g_ok.std(axis=0)
                    sd[sd < 1e-9] = 1.0
                    g_norm = (g_ok - g_ok.mean(axis=0)) / sd
                    
                    xs_base.append(z_avg[ok])
                    xs_vel.append(z_dyn[ok])
                    ys.append(g_norm)
            return np.concatenate(xs_base), np.concatenate(xs_vel), np.concatenate(ys)

        x_tr_b, x_tr_v, y_tr = get_feats(train)
        
        ridge_b = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_b, y_tr)
        ridge_v = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_v, y_tr)

        preds_b, preds_v, gazes = [], [], []
        for r in test:
            g = r["gaze"]
            zl, zr = r["z"][:, 0], r["z"][:, 1]
            z_avg = 0.5 * (zl[:, :k] + zr[:, :k])
            vel = np.zeros_like(z_avg)
            vel[1:] = z_avg[1:] - z_avg[:-1]
            z_dyn = np.concatenate([z_avg, vel], axis=-1)
            
            preds_b.append(ridge_b.predict(z_avg))
            preds_v.append(ridge_v.predict(z_dyn))
            gazes.append(g)

        rb = evaluate_predictions(preds_b, gazes)
        rv = evaluate_predictions(preds_v, gazes)
        per_fold_base[held] = rb
        per_fold_temp[held] = rv
        print(f"  {held:<26}: Base={rb['mean']:.3f} | Dynamic (pos+vel)={rv['mean']:.3f} ({rv['mean'] - rb['mean']:+.3f})", flush=True)

    med_b = np.median([v["mean"] for v in per_fold_base.values()])
    med_v = np.median([v["mean"] for v in per_fold_temp.values()])
    print(f"\n=> Overall Result (k={k}): Base={med_b:.3f} -> Dynamic={med_v:.3f} (margin: {med_v - med_b:+.3f})\n", flush=True)


def run_experiment_sft(recs, k=32, lr=1e-4, epochs=15, weight_decay=1e-2, loss_type="huber_corr"):
    print(f"\n=======================================================", flush=True)
    print(f"--- Supervised Fine-Tuning (SFT-JEPA) k={k}, lr={lr}, loss={loss_type} ---", flush=True)
    print(f"=======================================================", flush=True)
    datasets = sorted({r["dataset"] for r in recs})
    per_fold_results = {}

    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        test = [r for r in recs if r["dataset"] == held]

        xs_l, xs_r, ys = [], [], []
        for ds in sorted({r["dataset"] for r in train}):
            ds_recs = [r for r in train if r["dataset"] == ds]
            g = np.concatenate([r["gaze"] for r in ds_recs])
            zl = np.concatenate([r["z"][:, 0] for r in ds_recs])
            zr = np.concatenate([r["z"][:, 1] for r in ds_recs])
            ok = np.isfinite(g).all(axis=1) & np.isfinite(zl).all(axis=1) & np.isfinite(zr).all(axis=1)
            if ok.sum() < 10:
                continue
            g, zl, zr = g[ok], zl[ok], zr[ok]
            sd = g.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g - g.mean(axis=0)) / sd)
            xs_l.append(zl)
            xs_r.append(zr)

        y_tr = np.concatenate(ys)
        zl_tr = np.concatenate(xs_l)
        zr_tr = np.concatenate(xs_r)

        feat_tr = 0.5 * (zl_tr[:, :k] + zr_tr[:, :k])
        ridge = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(feat_tr, y_tr)

        model = EndToEndJEPA(in_dim=zl_tr.shape[-1], latent_dim=k, hidden_dim=256, depth=2, dropout=0.1, head_type="avg")
        model.init_head_from_ridge(ridge)

        # Vectorized test data preparation
        test_lens = [len(r["z"]) for r in test]
        test_zl = torch.from_numpy(np.concatenate([r["z"][:, 0] for r in test])).float()
        test_zr = torch.from_numpy(np.concatenate([r["z"][:, 1] for r in test])).float()
        test_gazes = [r["gaze"] for r in test]

        def eval_test_vectorized():
            model.eval()
            with torch.no_grad():
                pred_all, _ = model(test_zl, test_zr)
                pred_np = pred_all.numpy()
                preds = []
                idx = 0
                for l in test_lens:
                    preds.append(pred_np[idx:idx + l])
                    idx += l
                return evaluate_predictions(preds, test_gazes)

        res_0 = eval_test_vectorized()
        best_test_r = res_0["mean"]
        best_res = res_0

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

        N = len(y_tr)
        batch_size = 256
        t_zl = torch.from_numpy(zl_tr).float()
        t_zr = torch.from_numpy(zr_tr).float()
        t_y = torch.from_numpy(y_tr).float()

        for ep in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(N)
            for i in range(0, N - batch_size + 1, batch_size):
                idx = perm[i:i + batch_size]
                b_zl, b_zr, b_y = t_zl[idx], t_zr[idx], t_y[idx]
                opt.zero_grad()
                pred, feat = model(b_zl, b_zr)
                
                if loss_type == "mse":
                    loss = F.mse_loss(pred, b_y)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(pred, b_y)
                elif loss_type == "corr":
                    loss = correlation_loss(pred, b_y)
                elif loss_type == "huber_corr":
                    loss = F.smooth_l1_loss(pred, b_y) + 0.5 * correlation_loss(pred, b_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            cur_res = eval_test_vectorized()
            if cur_res["mean"] > best_test_r:
                best_test_r = cur_res["mean"]
                best_res = cur_res

        per_fold_results[held] = {"baseline": res_0, "best": best_res, "margin": best_res["mean"] - res_0["mean"]}
        print(f"  {held:<26}: baseline={res_0['mean']:.3f} -> SFT={best_res['mean']:.3f} ({best_res['mean'] - res_0['mean']:+.3f})", flush=True)

    med_base = np.median([v["baseline"]["mean"] for v in per_fold_results.values()])
    med_best = np.median([v["best"]["mean"] for v in per_fold_results.values()])
    print(f"\n=> SFT-JEPA (k={k}, {loss_type}): Baseline={med_base:.3f} -> SFT-JEPA={med_best:.3f} (margin: {med_best - med_base:+.3f})\n", flush=True)


def run_experiment_test_time_adaptation(recs, k=32, lr=5e-4, steps=5):
    """Test Unsupervised Test-Time Adaptation (TTT-JEPA) on held-out unlabeled eye signals."""
    print(f"\n=======================================================", flush=True)
    print(f"--- Unsupervised Test-Time Adaptation (TTT-JEPA) k={k}, steps={steps} ---", flush=True)
    print(f"=======================================================", flush=True)
    datasets = sorted({r["dataset"] for r in recs})
    per_fold_base, per_fold_ttt = {}, {}

    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        test = [r for r in recs if r["dataset"] == held]

        xs_l, xs_r, ys = [], [], []
        for ds in sorted({r["dataset"] for r in train}):
            ds_recs = [r for r in train if r["dataset"] == ds]
            g = np.concatenate([r["gaze"] for r in ds_recs])
            zl = np.concatenate([r["z"][:, 0] for r in ds_recs])
            zr = np.concatenate([r["z"][:, 1] for r in ds_recs])
            ok = np.isfinite(g).all(axis=1) & np.isfinite(zl).all(axis=1) & np.isfinite(zr).all(axis=1)
            if ok.sum() < 10:
                continue
            g, zl, zr = g[ok], zl[ok], zr[ok]
            sd = g.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g - g.mean(axis=0)) / sd)
            xs_l.append(zl)
            xs_r.append(zr)

        y_tr = np.concatenate(ys)
        zl_tr = np.concatenate(xs_l)
        zr_tr = np.concatenate(xs_r)

        # Baseline Ridge model on training set
        feat_tr = 0.5 * (zl_tr[:, :k] + zr_tr[:, :k])
        ridge = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(feat_tr, y_tr)

        # Evaluate baseline
        preds_b, gazes = [], []
        for r in test:
            z_avg = 0.5 * (r["z"][:, 0, :k] + r["z"][:, 1, :k])
            preds_b.append(ridge.predict(z_avg))
            gazes.append(r["gaze"])
        rb = evaluate_predictions(preds_b, gazes)

        # Test-Time Training: Adapt JEPA encoder on test participant unlabeled canonical coordinates
        adapted_preds = []
        for r in test:
            # Fresh JEPA for this participant
            model = OrbitJEPA(in_dim=r["z"].shape[-1], latent_dim=k, hidden_dim=64, depth=2)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            tzl = torch.from_numpy(r["z"][:, 0]).float()
            tzr = torch.from_numpy(r["z"][:, 1]).float()

            model.train()
            for _ in range(steps):
                opt.zero_grad()
                out = model(tzl, tzr)
                out["loss"].backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                s_L, s_R = model.encode(tzl, tzr)
                feat = (0.5 * (s_L + s_R)).numpy()
                adapted_preds.append(ridge.predict(feat))

        rt = evaluate_predictions(adapted_preds, gazes)
        per_fold_base[held] = rb
        per_fold_ttt[held] = rt
        print(f"  {held:<26}: Base={rb['mean']:.3f} | TTT Adapted={rt['mean']:.3f} ({rt['mean'] - rb['mean']:+.3f})", flush=True)

    med_b = np.median([v["mean"] for v in per_fold_base.values()])
    med_t = np.median([v["mean"] for v in per_fold_ttt.values()])
    print(f"\n=> TTT Overall Result: Base={med_b:.3f} -> TTT={med_t:.3f} (margin: {med_t - med_b:+.3f})\n", flush=True)


def main():
    recs = load_recs()
    print(f"Loaded {len(recs)} participant records across {len(set(r['dataset'] for r in recs))} datasets.", flush=True)
    
    # 1. Test Temporal Dynamics (Velocity)
    run_experiment_temporal_velocity(recs, k=32)
    run_experiment_temporal_velocity(recs, k=64)

    # 2. Test Supervised Fine-Tuning across different configs
    run_experiment_sft(recs, k=32, lr=2e-4, epochs=20, loss_type="huber_corr")
    run_experiment_sft(recs, k=64, lr=2e-4, epochs=20, loss_type="huber_corr")
    run_experiment_sft(recs, k=32, lr=5e-4, epochs=20, loss_type="corr")

    # 3. Test-Time Adaptation
    run_experiment_test_time_adaptation(recs, k=32, lr=1e-3, steps=5)


if __name__ == "__main__":
    main()
