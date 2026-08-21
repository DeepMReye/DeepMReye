"""The sub-TR gaze protocol, written down once.

`eval_probe.py` cannot score sub-TR gaze and never could: `evaluate.probe.temporal_targets`
nanmeans `[B, W, 10, 2] -> [B, n_t, 2]`, so every number it has produced is 1-TR mean gaze at
5-TR bins. The sub-TR headline (`lr-cca:32 + lags+-2` at 0.759) exists only inside
`scripts/benchmark_all_11_datasets.py`, which additionally uses a different basis
(`basis_n2000` against `basis_n1039` elsewhere), a different alpha grid and a different NaN
rule. Two implementations of "the number" is how a 0.221 came to be compared against a 0.847.

This module is the single audited implementation. `lodo_subtr` reproduces the benchmark's
scoring exactly and returns **both** resolutions, so no arm can quote one and imply the other.

Three deliberate differences from `sweep_orbitjepa.build_labeled_cache`, which answers a
different question and must not be reused for this one:

- **No `bin_reduce`.** That cache averages 5 TRs, so its "lag" unit is 5 TRs.
- **Labels keep their sub-TR axis.** That cache does `nanmean(labels, axis=1)` at build time,
  destroying the quantity this module exists to measure.
- **No `EXCLUDE`.** Its `EXCLUDE = ("dsL11_backtothefuture",)` predates the ingest; dsL11 is
  37 participants and part of the 9-fold headline.

And one addition: a **corpus fingerprint** in the cache guard. The existing labeled cache
validates only `(m, regress_motion, basis)`, so the on-disk copy -- 285 participants, 7
datasets, pre-vertical-convention-repair labels -- loads without complaint and silently
answers a retired question.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye.orbitjepa import orbit_projections  # noqa: E402

CACHE_VERSION = 1
# The label convention this cache was built under. `dsL08`/`dsL12` shipped with the vertical
# axis negated and `dsL12` with x/y transposed; a cache built before that repair scores a
# different corpus under the same name. Bump this string if the convention changes again.
LABEL_CONVENTION = "y-down-2026-08-20"

MIN_TRS = 60
MAX_TRAIN_ROWS = 20000
ALPHAS = np.logspace(-2, 4, 13)

# The headline numbers this module must reproduce before any ordering it produces is
# quotable. 9 folds, `basis_n2000`, 337 participants.
CALIBRATION = {"lr-cca:32": 0.742, "lr-cca:32+lags2": 0.759}


def calc_r(p, t):
    """Pearson r over finite pairs, NaN when either side is degenerate."""
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 10 or np.std(t[ok]) < 1e-6 or np.std(p[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


def make_lags(z, lags=0):
    """Stack `z` at offsets -lags..+lags along the feature axis.

    Moved here from `scripts/benchmark_all_11_datasets.py` so there is one definition; the
    benchmark imports it back. Edge padding, not zero padding -- worth stating because
    `nn.Conv1d` defaults to zeros, so an identity-initialised conv built to imitate this
    would differ from it only on the first and last `lags` rows of every participant, which
    is invisible in every downstream number.
    """
    if lags == 0:
        return z
    parts = []
    for lag in range(-lags, lags + 1):
        if lag < 0:
            p = np.pad(z[:lag], ((-lag, 0), (0, 0)), mode="edge")
        elif lag > 0:
            p = np.pad(z[lag:], ((0, lag), (0, 0)), mode="edge")
        else:
            p = z
        parts.append(p)
    return np.concatenate(parts, axis=1)


def corpus_fingerprint(recs):
    """A hash of which participants, at what length, under which label convention."""
    items = sorted((r["dataset"], r["subject"], int(r["labels"].shape[0])) for r in recs)
    payload = json.dumps([LABEL_CONVENTION, items], separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf8")).hexdigest()


def build_subtr_cache(root, mask, basis, m=256, regress_motion=False, verbose=True):
    """Every gaze-labeled participant as per-TR canonical coords + `[T, 10, 2]` labels."""
    import h5py

    flat_mask = mask.reshape(-1)
    recs = []
    for ds_dir in sorted(p for p in Path(root).glob("dsL*") if p.is_dir()):
        for path in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(path, "r") as f:
                    if "labels" not in f:
                        continue
                    block = f["eye_block"][:]
                    labels = f["labels"][:]
            except Exception:
                continue
            t = block.shape[-1]
            if t < MIN_TRS or not np.isfinite(labels).any():
                continue

            rows = block.reshape(-1, t).T[:, flat_mask].astype(np.float64)
            zl, zr = orbit_projections(rows, basis, m=m, regress_motion=regress_motion)
            n = min(len(zl), len(labels))
            recs.append({
                "dataset": ds_dir.name,
                "subject": path.stem,
                # Both orbits kept: whether to average or concatenate them is an open
                # question, and a cache that has already averaged cannot answer it.
                "z": np.stack([zl[:n], zr[:n]], axis=1).astype(np.float32),  # [T, 2, m]
                "labels": labels[:n].astype(np.float32),                      # [T, 10, 2]
            })
        if verbose:
            got = sum(1 for r in recs if r["dataset"] == ds_dir.name)
            print(f"    {ds_dir.name:<28}{got:>4} participants", flush=True)
    if not recs:
        raise SystemExit("[!] no labeled participants found")
    return recs


def save_subtr_cache(path, recs, basis_path, m, regress_motion):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"z/{i}": r["z"] for i, r in enumerate(recs)}
    flat.update({f"y/{i}": r["labels"] for i, r in enumerate(recs)})
    np.savez(path,
             version=np.array([CACHE_VERSION]),
             n=np.array([len(recs)]),
             m=np.array([m]),
             regress_motion=np.array([int(regress_motion)]),
             basis=np.array(str(basis_path)),
             fingerprint=np.array(corpus_fingerprint(recs)),
             meta=np.array(json.dumps([{"dataset": r["dataset"], "subject": r["subject"]}
                                       for r in recs])),
             **flat)


def load_subtr_cache(path, basis_path, m, regress_motion, fingerprint=None):
    """Load, refusing anything built for a different corpus or geometry.

    `fingerprint=None` skips only the corpus check (there is nothing to compare against yet);
    version, rank, motion handling and basis path are always enforced.
    """
    d = np.load(path, allow_pickle=False)
    got = (int(d["version"][0]), int(d["m"][0]), bool(d["regress_motion"][0]), str(d["basis"]))
    want = (CACHE_VERSION, int(m), bool(regress_motion), str(basis_path))
    if got != want:
        raise SystemExit(f"[!] sub-TR cache built for {got}, requested {want}; "
                         f"rebuild with --build-cache")
    if fingerprint is not None and str(d["fingerprint"]) != fingerprint:
        raise SystemExit("[!] sub-TR cache was built on a different corpus "
                         f"({str(d['fingerprint'])[:12]} != {fingerprint[:12]}); "
                         "rebuild with --build-cache")
    meta = json.loads(str(d["meta"]))
    return [{**meta[i], "z": d[f"z/{i}"], "labels": d[f"y/{i}"]}
            for i in range(int(d["n"][0]))]


def cca_avg(rec, k=32):
    """The incumbent's feature: L/R average of the leading `k` canonical directions."""
    z = rec["z"]
    return 0.5 * (z[:, 0, :k] + z[:, 1, :k]).astype(np.float64)


def subject_scores(pred_20, labels):
    """One participant's `(sub-TR r, 1-TR r)` from `[T, 20]` predictions and `[T, 10, 2]` gaze.

    Factored out so a network and the linear incumbent are scored by the *same* code rather
    than by two implementations that agree until they do not. `ok` requires all 20 target
    components finite, which is the incumbent's row rule -- a model scored on a different row
    set has a different denominator in every r.
    """
    t_n = len(pred_20)
    y_flat = labels[:t_n].reshape(t_n, 20)
    ok = np.isfinite(y_flat).all(axis=1) & np.isfinite(pred_20).all(axis=1)
    if ok.sum() < 10:
        return float("nan"), float("nan")
    pred = np.asarray(pred_20)[:t_n].reshape(t_n, 10, 2)
    lab = np.asarray(labels)[:t_n]

    p_s, t_s = pred[ok].reshape(-1, 2), lab[ok].reshape(-1, 2)
    rx, ry = calc_r(p_s[:, 0], t_s[:, 0]), calc_r(p_s[:, 1], t_s[:, 1])
    r_sub = (rx + ry) / 2.0 if np.isfinite(rx) and np.isfinite(ry) else float("nan")

    p_1, t_1 = np.nanmean(pred[ok], axis=1), np.nanmean(lab[ok], axis=1)
    rx, ry = calc_r(p_1[:, 0], t_1[:, 0]), calc_r(p_1[:, 1], t_1[:, 1])
    r_1 = (rx + ry) / 2.0 if np.isfinite(rx) and np.isfinite(ry) else float("nan")
    return r_sub, r_1


def fold_median(scores):
    """Median over participants, NaN-safe, matching the incumbent's aggregation."""
    vals = [v for v in scores if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def lodo_subtr(recs, feature_fn, seed=0, alphas=ALPHAS, max_train_rows=MAX_TRAIN_ROWS,
               readout=None):
    """Leave-one-dataset-out, scored at sub-TR **and** 1-TR resolution.

    Reproduces `scripts/benchmark_all_11_datasets.py:115-200` exactly: targets z-scored per
    *training* dataset, `sorted()` dataset iteration, the same row-validity rule
    (all 20 target components finite), the same subsample cap, per-subject r, median over
    subjects within a fold, median over folds.

    `sorted`, not a bare set: set iteration order over strings varies with PYTHONHASHSEED
    between processes, which changes the concatenation order and therefore which rows the
    subsample keeps -- ~0.01 of avoidable noise in a comparison meant to resolve 0.02.

    `readout` is a zero-argument factory returning a fresh unfitted estimator, so a
    non-linear readout is scored through *this* function rather than a second copy of the
    protocol. Default is the incumbent's `RidgeCV(alphas)`.
    """
    from sklearn.linear_model import RidgeCV

    if readout is None:
        def readout():
            return RidgeCV(alphas=alphas)

    feats = {id(r): np.asarray(feature_fn(r), dtype=np.float64) for r in recs}
    datasets = sorted({r["dataset"] for r in recs})
    per_fold = {"subtr": {}, "1tr": {}}

    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        test = [r for r in recs if r["dataset"] == held]

        xs, ys = [], []
        for ds in sorted({r["dataset"] for r in train}):
            members = [r for r in train if r["dataset"] == ds]
            x = np.concatenate([feats[id(r)] for r in members])
            y = np.concatenate([r["labels"].reshape(len(r["labels"]), 20) for r in members])
            ok = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            x, y = x[ok], y[ok]
            sd = y.std(axis=0)
            sd[sd < 1e-9] = 1.0
            xs.append(x)
            ys.append((y - y.mean(axis=0)) / sd)
        if not xs:
            continue

        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > max_train_rows:
            idx = np.random.default_rng(seed).choice(len(x_tr), max_train_rows, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]
        model = readout().fit(x_tr, y_tr)

        r_sub, r_1tr = [], []
        for s in test:
            a, b = subject_scores(model.predict(feats[id(s)]), s["labels"])
            if np.isfinite(a):
                r_sub.append(a)
            if np.isfinite(b):
                r_1tr.append(b)

        per_fold["subtr"][held] = fold_median(r_sub)
        per_fold["1tr"][held] = fold_median(r_1tr)

    out = dict(per_fold)
    for key in ("subtr", "1tr"):
        vals = [v for v in per_fold[key].values() if np.isfinite(v)]
        out[f"median_{key}"] = float(np.median(vals)) if vals else float("nan")
    return out


def calibrate(recs, tol=0.01, verbose=True):
    """Reproduce the known headline numbers, or refuse to be trusted.

    `fold-pca:64` (0.746) is deliberately **not** here: it refits a PCA on the held-out
    fold's training *voxels*, which this cache does not store. It is covered by re-running
    `scripts/benchmark_all_11_datasets.py` end to end, which is the check that the two
    implementations still agree.
    """
    arms = {
        "lr-cca:32": lambda r: cca_avg(r, 32),
        "lr-cca:32+lags2": lambda r: make_lags(cca_avg(r, 32), 2),
    }
    ok = True
    for name, fn in arms.items():
        got = lodo_subtr(recs, fn)["median_subtr"]
        want = CALIBRATION[name]
        hit = abs(got - want) <= tol
        ok &= hit
        if verbose:
            print(f"  {name:<20} sub-TR {got:.4f}  expected {want:.4f}  "
                  f"{'OK' if hit else 'MISMATCH'}", flush=True)
    return ok


def _main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    args = p.parse_args()

    from deepmreye.datasource import resolve
    from deepmreye.unsupervised import corpus_mask, load_basis

    data_dir = Path(args.data_dir) if args.data_dir else resolve(None, download=False, quiet=True)
    basis_path = Path(args.basis)

    if args.build_cache or not Path(args.cache).exists():
        print(f"[*] building sub-TR cache from {data_dir}", flush=True)
        mask = corpus_mask(data_dir)
        _m, bases, _meta = load_basis(basis_path)
        recs = build_subtr_cache(data_dir, mask, bases["lr-cca"], m=args.m)
        save_subtr_cache(args.cache, recs, basis_path, args.m, False)
        print(f"[+] {len(recs)} participants -> {args.cache}")
    else:
        recs = load_subtr_cache(args.cache, basis_path, args.m, False)
        print(f"[*] loaded {len(recs)} participants from {args.cache}")

    print(f"[*] {len(recs)} participants, "
          f"{len(sorted({r['dataset'] for r in recs}))} datasets, "
          f"fingerprint {corpus_fingerprint(recs)[:12]}")

    if args.calibrate:
        print("[*] calibrating against the known headline numbers:")
        if not calibrate(recs):
            raise SystemExit("[!] calibration failed -- do not trust any ordering from this "
                             "module until it reproduces")
        print("[+] calibrated")


if __name__ == "__main__":
    _main()
