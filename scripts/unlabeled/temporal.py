"""After the readout, in time: does a model of gaze dynamics beat the readout's
learned +-1 TR kernel?

The shipped readout turns three volumes (t-1, t, t+1) into the ten gaze samples
of TR t with weights learned on the training datasets -- a temporal kernel
shaped mostly by the pursuit sets, which supply most of the rows. The
alternative is to read each volume out on its own (lag 0, one value per TR: the
mean of its ten outputs) and let a temporal stage produce the ten samples:
repeat the value, interpolate between TR centres, or take the posterior mean of
a Gaussian process whose observation at TR i is the *average* of the latent
gaze over that TR plus white noise. Every stage is applied per participant and
axis to the decoded trace, so it is scored by ``probe.score_predictions`` -- the
same code as everything else -- and the GP's hyperparameters (length scale,
signal and noise variance) are fitted without labels:

- ``ml``           by marginal likelihood on the held-out dataset's decoded traces
                   (per study, or per participant);
- ``bilateral``    from the two orbits' decodings: their cross-covariance is the
                   signal's (orbit noise is independent), the rest is noise;
- ``train-gaze``   length scale (in seconds) from the *other* datasets' measured
                   gaze -- a dynamics prior from labelled training data -- with the
                   variances by marginal likelihood;
- ``oracle``       length scale from the held-out dataset's own measured gaze: the
                   ceiling for this family, not a method.

The kernel is chosen for each held-out dataset by its score on the other eight
(``nested``), never on the dataset itself. The same stages are applied to the
supervised voxel ridge, to see whether a gain belongs to the decoder or to the
stage.
"""
import json
import re
import time
from pathlib import Path

import numpy as np

from deepmreye import probe
from unlabeled.common import log, orbit_views, ridge_auto, slicer, voxel_block

KERNELS = ("matern12", "matern32", "matern52", "se")
SUB = 10                                   # gaze samples per TR
OFFSETS = (np.arange(SUB)[:, None] - np.arange(SUB)[None, :]).ravel() / SUB   # within-TR time differences
BLOCK = 32                                 # TRs per block of the marginal likelihood


# --------------------------------------------------------------------------- #
# The Gaussian process, with the TR average as its observation
# --------------------------------------------------------------------------- #

def kernel(name, tau, ell):
    r = np.abs(tau) / ell
    if name == "matern12":
        return np.exp(-r)
    if name == "matern32":
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    if name == "matern52":
        return (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
    if name == "se":
        return np.exp(-0.5 * r ** 2)
    raise ValueError(name)


def obs_cov(name, ell, s2, n):
    """Covariance of two TR averages ``d`` TRs apart, d = 0..n-1 (time in TRs)."""
    d = np.arange(n, dtype=float)
    return s2 * kernel(name, d[:, None] + OFFSETS[None, :], ell).mean(1)


def gp_filter(name, ell, s2, sig2):
    """``W [10, 2H+1]``: the posterior mean of the ten samples of TR i given the
    centred TR averages i-H..i+H. Stationary kernel and evenly spaced TRs make
    it one filter for every interior TR."""
    from scipy.linalg import solve, toeplitz

    H = int(min(60, max(12, np.ceil(5 * ell))))
    col = obs_cov(name, ell, s2, 2 * H + 1)
    M = toeplitz(col) + sig2 * np.eye(2 * H + 1)
    d = np.arange(-H, H + 1, dtype=float)
    # sample j of TR 0 is at (j + .5)/10; sample l of TR d at d + (l + .5)/10
    lj = (np.arange(SUB)[None, :] - np.arange(SUB)[:, None]) / SUB            # [j, l] = (l - j)/10
    C = s2 * kernel(name, d[None, :, None] + lj[:, None, :], ell).mean(2)      # [10, 2H+1]
    return solve(M, C.T, assume_a="pos").T, H


def apply_filter(y, W, H):
    """Centred per-TR trace ``[T]`` -> ``[T, 10]``; reflected at the run's ends."""
    pad = np.pad(y, H, mode="reflect") if len(y) > H else np.pad(y, H, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * H + 1)[:len(y)]
    return win @ W.T


def _blocks(traces):
    rows = [t[i:i + BLOCK] for t in traces for i in range(0, len(t) - BLOCK + 1, BLOCK)]
    return np.stack(rows) if rows else np.zeros((0, BLOCK))


def nll(name, ell, s2, sig2, blocks):
    from scipy.linalg import cho_factor, cho_solve, toeplitz

    M = toeplitz(obs_cov(name, ell, s2, BLOCK)) + sig2 * np.eye(BLOCK)
    try:
        c = cho_factor(M)
    except np.linalg.LinAlgError:
        return np.inf
    quad = float(np.sum(blocks * cho_solve(c, blocks.T).T))
    return 0.5 * quad + len(blocks) * float(np.log(np.diag(c[0])).sum())


def fit_ml(name, traces, ell=None, ell_max=30.0):
    """Length scale (TRs), signal and noise variance by marginal likelihood on
    centred per-TR traces; ``ell`` fixes the length scale, ``ell_max`` bounds it."""
    from scipy.optimize import minimize

    blocks = _blocks(traces)
    if not len(blocks):
        return None
    v = float(blocks.var())
    if ell is None:
        f = lambda p: nll(name, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), blocks)  # noqa: E731
        x0 = [0.0, np.log(0.7 * v), np.log(0.3 * v)]
        x0[0] = min(0.0, np.log(ell_max) - 0.1)
        bounds = [(np.log(0.05), np.log(ell_max))] + [(np.log(1e-4 * v), np.log(10 * v))] * 2
        p = minimize(f, x0, method="L-BFGS-B", bounds=bounds).x
        return float(np.exp(p[0])), float(np.exp(p[1])), float(np.exp(p[2]))
    f = lambda p: nll(name, ell, np.exp(p[0]), np.exp(p[1]), blocks)  # noqa: E731
    p = minimize(f, [np.log(0.7 * v), np.log(0.3 * v)], method="L-BFGS-B",
                 bounds=[(np.log(1e-4 * v), np.log(10 * v))] * 2).x
    return float(ell), float(np.exp(p[0])), float(np.exp(p[1]))


def fit_bilateral(name, lefts, rights, max_lag=8):
    """Signal kernel from the two orbits' cross-covariance at TR lags 0..8
    (least squares), noise as the rest of the averaged trace's variance."""
    from scipy.optimize import minimize

    num, den = np.zeros(max_lag + 1), np.zeros(max_lag + 1)
    var_avg = []
    for a, b in zip(lefts, rights):
        for d in range(max_lag + 1):
            if len(a) > d:
                num[d] += 0.5 * (np.dot(a[:len(a) - d], b[d:]) + np.dot(b[:len(b) - d], a[d:]))
                den[d] += len(a) - d
        var_avg.append(((a + b) / 2) ** 2)
    c = num / np.maximum(den, 1)
    v = float(np.concatenate(var_avg).mean())
    if c[0] <= 0:
        return None
    f = lambda p: float(((obs_cov(name, np.exp(p[0]), np.exp(p[1]), max_lag + 1) - c) ** 2).sum())  # noqa: E731
    p = minimize(f, [0.0, np.log(c[0])], method="L-BFGS-B",
                 bounds=[(np.log(0.05), np.log(30)), (np.log(1e-4 * v), np.log(10 * v))]).x
    ell, s2 = float(np.exp(p[0])), float(np.exp(p[1]))
    return ell, s2, max(v - float(obs_cov(name, ell, s2, 1)[0]), 0.02 * v)


def gaze_length_scale(name, traces_by_dataset, max_lag_s=6.0):
    """Length scale in *seconds* fitted to the measured gaze's autocorrelation
    (sub-TR samples, each dataset weighted equally): ``[(trace [n], dt), ...]``."""
    from scipy.optimize import minimize

    lags = np.linspace(0, max_lag_s, 61)
    acfs = []
    for traces in traces_by_dataset:
        acc = []
        for x, dt in traces:
            x = x[np.isfinite(x)]
            if len(x) < 50 or x.std() < 1e-9:
                continue
            x = (x - x.mean()) / x.std()
            ks = np.round(lags / dt).astype(int)
            acc.append([np.mean(x[:len(x) - k] * x[k:]) if k < len(x) - 10 else np.nan for k in ks])
        if acc:
            acfs.append(np.nanmean(acc, axis=0))
    acf = np.nanmean(acfs, axis=0)
    ok = np.isfinite(acf)
    f = lambda p: float(((acf[ok] - kernel(name, lags[ok], np.exp(p[0]))) ** 2).sum())  # noqa: E731
    return float(np.exp(minimize(f, [0.0], method="L-BFGS-B", bounds=[(np.log(0.02), np.log(30))]).x[0]))


# --------------------------------------------------------------------------- #
# Stages: per-TR trace -> ten samples per TR
# --------------------------------------------------------------------------- #

def per_tr(pred):
    """``[T, 20]`` -> ``[T, 2]``, the mean of each TR's ten samples per axis."""
    return np.asarray(pred).reshape(len(pred), SUB, 2).mean(1)


def to_pred(samples):
    """``[T, 10, 2]`` -> the ``[T, 20]`` layout every prediction is kept in."""
    return np.asarray(samples).reshape(len(samples), 2 * SUB)


def stage_repeat(y):
    return np.repeat(y[:, None, :], SUB, axis=1)


def stage_interp(y, kind="linear"):
    """Between TR centres, the decoded value placed at each TR's midpoint."""
    from scipy.interpolate import CubicSpline

    t_obs = np.arange(len(y)) + 0.5
    t = (np.arange(len(y))[:, None] + (np.arange(SUB)[None, :] + 0.5) / SUB).ravel()
    out = np.empty((len(t), 2))
    for a in range(2):
        out[:, a] = (np.interp(t, t_obs, y[:, a]) if kind == "linear"
                     else CubicSpline(t_obs, y[:, a], bc_type="natural")(np.clip(t, t_obs[0], t_obs[-1])))
    return out.reshape(len(y), SUB, 2)


def stage_gp(y, params, name):
    """``params[axis] = (ell, s2, sig2)``: the GP posterior mean, per axis."""
    out = np.empty((len(y), SUB, 2))
    for a in range(2):
        mu = y[:, a].mean()
        W, H = gp_filter(name, *params[a])
        out[:, :, a] = apply_filter(y[:, a] - mu, W, H) + mu
    return out


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #

def _tr_of(rec, trs):
    """Acquisition TR in seconds; ``dsL06`` carries it in the subject name."""
    m = re.search(r"TR(\d+)", rec["subject"])
    return float(m.group(1)) / 1000 if m and rec["dataset"].startswith("dsL06") else trs[rec["dataset"]]


def _rows(res):
    return {(q["dataset"], q["subject"]): (q["subtr"]["r"], q["1tr"]["r"]) for q in res["participants"]}


def _summ(res):
    return {"subtr": res["summary"]["subtr"]["median_r"], "1tr": res["summary"]["1tr"]["median_r"],
            "subtr_mean": res["summary"]["subtr"]["mean_r"], "1tr_mean": res["summary"]["1tr"]["mean_r"],
            "folds": {d: {"n": f["n"], "subtr": f["subtr"]["r"], "1tr": f["1tr"]["r"],
                          "subtr_x": f["subtr"]["r_x"], "subtr_y": f["subtr"]["r_y"]}
                      for d, f in res["folds"].items()}}


def paired(rows_a, rows_b, res=0, n_boot=2000, seed=0):
    """Mean over datasets of (a - b) per-dataset medians; participants resampled
    within datasets, the same ones for both; and datasets up / down by 0.005."""
    rng = np.random.default_rng(seed)
    by = {}
    for key in rows_a:
        if key in rows_b and np.isfinite(rows_a[key][res]) and np.isfinite(rows_b[key][res]):
            by.setdefault(key[0], []).append((rows_a[key][res], rows_b[key][res]))
    pairs = {d: np.array(v) for d, v in by.items()}
    stat = lambda s: float(np.mean([np.median(p[:, 0]) - np.median(p[:, 1]) for p in s.values()]))  # noqa: E731
    boots = [stat({d: p[rng.integers(0, len(p), len(p))] for d, p in pairs.items()}) for _ in range(n_boot)]
    per = {d: float(np.median(p[:, 0]) - np.median(p[:, 1])) for d, p in pairs.items()}
    return {"delta_mean": stat(pairs), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "up": sum(v > 0.005 for v in per.values()), "down": sum(v < -0.005 for v in per.values()),
            "per_dataset": per}


def temporal(ctx, out):
    """Every temporal stage on the lag-0 readout of the frozen basis (and of the
    supervised voxel ridge), against the shipped +-1 TR readout; see the module
    docstring. Writes the summaries, fitted hyperparameters, nested choices and
    paired comparisons, and every participant's r per arm."""
    n = ctx.n
    trs = {d["dataset"]: d["tr"] for d in json.load(open(Path(out).parent / "corpus_stats.json"))["labelled"]}
    recs, index = ctx.records([n], extra=[voxel_block(ctx.mask, 3), voxel_block(ctx.mask, 2)])
    name = f"lrcca_n{n}"
    t0 = time.time()
    shipped = probe.lodo(recs, slicer(index, name, k=32, lags=1), keep_predictions=True)
    lag0 = probe.lodo(recs, slicer(index, name, k=32, lags=0), keep_predictions=True,
                      views=orbit_views(index, name, k=32, lags=0))
    sup1 = probe.lodo(recs, slicer(index, "vox3", lags=1), readout=ridge_auto, keep_predictions=True)
    sup0 = probe.lodo(recs, slicer(index, "vox3", lags=0), readout=ridge_auto, keep_predictions=True)
    # 2x2x2 at lag 0 only: its +-1 TR design is a ~20 GB matrix, and its protocol
    # scores are already per participant in scheme_participants.json.
    sup2 = probe.lodo(recs, slicer(index, "vox2", lags=0), readout=ridge_auto, keep_predictions=True)
    log(f"readouts: shipped {shipped['median_subtr']:.4f}, lag 0 {lag0['median_subtr']:.4f}, "
        f"supervised 3x3x3 +-1 {sup1['median_subtr']:.4f}, lag 0 {sup0['median_subtr']:.4f}, "
        f"2x2x2 lag 0 {sup2['median_subtr']:.4f}", t0)

    datasets = sorted({r["dataset"] for r in recs})
    by_ds = {d: [r for r in recs if r["dataset"] == d] for d in datasets}
    key = lambda r: (r["dataset"], r["subject"])  # noqa: E731

    def centred(p):
        y = per_tr(p)
        return y - y.mean(0)

    arms, params = {}, {}

    def score(arm, preds):
        res = probe.score_predictions(recs, preds)
        arms[arm] = {"summary": _summ(res), "rows": _rows(res)}
        s = arms[arm]["summary"]
        log(f"{arm:<44} sub-TR {s['subtr']:.4f} / mean {s['subtr_mean']:.4f}   "
            f"1-TR {s['1tr']:.4f} / {s['1tr_mean']:.4f}", t0)

    def per_participant_stage(source, fn):
        return {key(r): to_pred(fn(per_tr(source[key(r)]), r)) for r in recs if key(r) in source}

    for label, res in (("ours, shipped (+-1 TR readout)", shipped), ("ours, lag-0 readout", lag0),
                       ("supervised, +-1 TR readout", sup1), ("supervised, lag-0 readout", sup0),
                       ("supervised 2x2x2, lag-0 readout", sup2)):
        score(label, res["predictions"])
    schemes = json.load(open(Path(out).parent / "scheme_participants.json"))
    arms["supervised 2x2x2, +-1 TR readout"] = {"summary": None, "rows": {
        (q["dataset"], q["subject"]): (q["subtr"], q["1tr"]) for q in schemes
        if q["scheme"] == "other datasets, all eight" and q["arm"] == "voxels:2+lags1"}}
    median_tr = {d: float(np.median([_tr_of(r, trs) for r in by_ds[d]])) for d in datasets}

    sources = (("ours", lag0["predictions"]), ("supervised", sup0["predictions"]),
               ("supervised 2x2x2", sup2["predictions"]))
    for src_label, src in sources:
        score(f"{src_label}, lag 0 + repeat", per_participant_stage(src, lambda y, r: stage_repeat(y)))
        score(f"{src_label}, lag 0 + linear interpolation", per_participant_stage(src, lambda y, r: stage_interp(y)))
        score(f"{src_label}, lag 0 + cubic spline", per_participant_stage(src, lambda y, r: stage_interp(y, "cubic")))
        # GP, marginal likelihood per study (the held-out dataset's decoded traces, no labels)
        for kname in KERNELS:
            fitted = {d: [fit_ml(kname, [centred(src[key(r)])[:, a] for r in by_ds[d] if key(r) in src])
                          for a in range(2)] for d in datasets}
            params[f"{src_label}|ml|{kname}"] = fitted
            score(f"{src_label}, lag 0 + GP {kname}, per study",
                  per_participant_stage(src, lambda y, r, f=fitted, k=kname: stage_gp(y, f[r["dataset"]], k)))
        # the same with the length scale bounded at 2 s, about twice measured gaze's
        fitted = {d: [fit_ml("matern32", [centred(src[key(r)])[:, a] for r in by_ds[d] if key(r) in src],
                             ell_max=2.0 / median_tr[d]) for a in range(2)] for d in datasets}
        params[f"{src_label}|ml-bounded|matern32"] = fitted
        score(f"{src_label}, lag 0 + GP matern32, per study, length scale <= 2 s",
              per_participant_stage(src, lambda y, r, f=fitted: stage_gp(y, f[r["dataset"]], "matern32")))

    # ---- ours only: the label-free variants that need the orbits or per-participant fits ----
    src = lag0["predictions"]
    for kname in KERNELS:
        fitted = {}
        for d in datasets:
            ks = [key(r) for r in by_ds[d] if key(r) in src]
            fitted[d] = [fit_bilateral(kname, [centred(lag0["views"]["left"][k_])[:, a] for k_ in ks],
                                       [centred(lag0["views"]["right"][k_])[:, a] for k_ in ks]) for a in range(2)]
        params[f"ours|bilateral|{kname}"] = fitted
        score(f"ours, lag 0 + bilateral GP {kname}",
              per_participant_stage(src, lambda y, r, f=fitted, k=kname: stage_gp(y, f[r["dataset"]], k)))
    pp = {key(r): [fit_ml("matern32", [centred(src[key(r)])[:, a]]) for a in range(2)] for r in recs if key(r) in src}
    params["ours|ml-participant|matern32"] = {f"{k_[0]}/{k_[1]}": v for k_, v in pp.items()}
    score("ours, lag 0 + GP matern32, per participant",
          per_participant_stage(src, lambda y, r: stage_gp(y, pp[key(r)], "matern32")))

    # ---- length scale from measured gaze: the other datasets' (honest) or the dataset's own (oracle) ----
    gaze = {d: [[(r["labels"][:, :, a].reshape(-1), _tr_of(r, trs) / SUB) for r in by_ds[d]] for a in range(2)]
            for d in datasets}
    for variant in ("train-gaze", "oracle"):
        fitted, ells = {}, {}
        for d in datasets:
            pools = [gaze[o] for o in datasets if o != d] if variant == "train-gaze" else [gaze[d]]
            ell_s = [gaze_length_scale("matern32", [p[a] for p in pools]) for a in range(2)]
            ells[d] = ell_s
            for r in by_ds[d]:
                if key(r) not in src:
                    continue
                tr = _tr_of(r, trs)
                fitted[key(r)] = [fit_ml("matern32", [centred(src[key(r)])[:, a]], ell=ell_s[a] / tr)
                                  for a in range(2)]
        params[f"ours|{variant}|matern32"] = {"length_scale_s": ells}
        score(f"ours, lag 0 + GP matern32, {variant} length scale",
              per_participant_stage(src, lambda y, r, f=fitted: stage_gp(y, f[key(r)], "matern32")))

    # ---- together with the shipped readout ----
    fitted = {d: [fit_ml("matern32", [centred(shipped["predictions"][key(r)])[:, a] for r in by_ds[d]])
                  for a in range(2)] for d in datasets}
    params["ours-shipped|ml|matern32"] = fitted
    score("ours, shipped + GP matern32 on its TR means",
          per_participant_stage(shipped["predictions"], lambda y, r: stage_gp(y, fitted[r["dataset"]], "matern32")))
    gp0 = {key(r): to_pred(stage_gp(per_tr(src[key(r)]), params["ours|ml|matern32"][r["dataset"]], "matern32"))
           for r in recs if key(r) in src}
    score("ours, average of shipped and lag 0 + GP matern32",
          {k_: 0.5 * (shipped["predictions"][k_] + gp0[k_]) for k_ in gp0})

    # ---- nested: the label-free stage chosen on the other eight datasets ----
    for src_label in ("ours", "supervised", "supervised 2x2x2"):
        # every stage that uses no label of the held-out dataset (the oracle does)
        pool = [a for a in arms if a.startswith(f"{src_label}, lag 0 +") and "oracle" not in a
                and "repeat" not in a]
        choice = {}
        for d in datasets:
            others = [o for o in datasets if o != d]
            best = max(pool, key=lambda a: np.mean([arms[a]["summary"]["folds"][o]["subtr"] for o in others]))
            choice[d] = best
        params[f"{src_label}|nested"] = choice
        # rebuild the chosen predictions per dataset from the stored rows' arms
        rows = {k_: v for d in datasets for k_, v in arms[choice[d]]["rows"].items() if k_[0] == d}
        folds = {}
        for d in datasets:
            f = arms[choice[d]]["summary"]["folds"][d]
            folds[d] = f
        sub = [folds[d]["subtr"] for d in datasets]
        one = [folds[d]["1tr"] for d in datasets]
        arms[f"{src_label}, lag 0 + stage chosen on the other datasets"] = {
            "summary": {"subtr": float(np.median(sub)), "1tr": float(np.median(one)),
                        "subtr_mean": float(np.mean(sub)), "1tr_mean": float(np.mean(one)), "folds": folds},
            "rows": rows}
        s = arms[f"{src_label}, lag 0 + stage chosen on the other datasets"]["summary"]
        log(f"{src_label + ', nested choice':<44} sub-TR {s['subtr']:.4f} / mean {s['subtr_mean']:.4f}   "
            f"1-TR {s['1tr']:.4f} / {s['1tr_mean']:.4f}", t0)

    ref = {"ours": arms["ours, shipped (+-1 TR readout)"]["rows"],
           "supervised": arms["supervised, +-1 TR readout"]["rows"],
           "supervised 2x2x2": arms["supervised 2x2x2, +-1 TR readout"]["rows"]}
    comparisons = {a: {"vs": "shipped +-1 TR readout of the same decoder",
                       "subtr": paired(v["rows"], ref[a.split(",")[0]], 0),
                       "1tr": paired(v["rows"], ref[a.split(",")[0]], 1)} for a, v in arms.items()}
    nested = lambda label: arms[f"{label}, lag 0 + stage chosen on the other datasets"]["rows"]  # noqa: E731
    for label in ("supervised", "supervised 2x2x2"):
        comparisons[f"ours nested vs {label} nested"] = {res: paired(nested("ours"), nested(label), i)
                                                         for i, res in enumerate(("subtr", "1tr"))}
    comparisons["ours shipped vs supervised 2x2x2 shipped"] = {
        res: paired(ref["ours"], ref["supervised 2x2x2"], i) for i, res in enumerate(("subtr", "1tr"))}
    json.dump({"arms": {a: {"summary": v["summary"],
                            "participants": [{"dataset": k_[0], "subject": k_[1], "subtr": r_[0], "1tr": r_[1]}
                                             for k_, r_ in v["rows"].items()]} for a, v in arms.items()},
               "params": params, "comparisons": comparisons},
              open(out, "w"), indent=1, default=float)
