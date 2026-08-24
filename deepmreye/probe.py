"""The gaze decoding protocol, written down once.

Leave-one-dataset-out over the gaze-labeled corpus: a readout is fitted on eight
datasets and scored on the ninth, so every number answers "does this transfer to
a study it has never seen", which is the only question a method that ships a
frozen basis can be asked.

Two resolutions, always both. The labels are ``[T, 10, 2]`` -- ten gaze samples
inside every TR -- so the same predictions can be scored **sub-TR** (all ten
samples, the resolution that makes MR-based eye tracking interesting) and
**1-TR** (their mean). Reporting one and implying the other is how a 0.221 came
to be compared against a 0.847, so :func:`lodo` returns both and every caller
gets both.

Three details in the fit that are not free choices:

- **Targets are z-scored per training dataset.** The per-dataset Euclidean
  scale spans 21 to 595, so a single pooled ridge otherwise follows whichever
  dataset has the largest target variance; unstandardised, the 9-fold median
  collapses to 0.131. The consequence is that predictions are in z-units, which
  is what :mod:`deepmreye.metrics` calibrates away before computing R-squared
  and Euclidean error.
- **Datasets are iterated in ``sorted`` order.** Set iteration over strings
  varies with PYTHONHASHSEED between processes, which changes the concatenation
  order of the training rows -- about 0.01 of avoidable noise in a comparison
  meant to resolve 0.02.
- **No training-row subsample.** There used to be a cap of 20000 rows with no
  justification anywhere in the code or its history. It is worth at most 0.0024
  on anything measured, but it is not defensible in a published number, so the
  fit uses all 303733 valid rows.

:func:`calibrate` reproduces two known headline numbers before any ordering this
module produces should be believed.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye import metrics  # noqa: E402
from deepmreye.unsupervised import orbit_projections  # noqa: E402

CACHE_VERSION = 2

# The label convention this cache was built under. `dsL08` shipped with the
# vertical axis negated; a cache built before that repair scores a different
# corpus under the same name. Bump this string if the convention changes again.
LABEL_CONVENTION = "y-down-2026-08-20"

MIN_TRS = 60
ALPHAS = np.logspace(-2, 4, 13)

# The incumbent arm. `k` and `lags` were both swept; see FINDINGS.md.
DEFAULT_K = 32
DEFAULT_LAGS = 1

# Headline sub-TR medians this module must reproduce before any ordering it
# produces is quotable. 9 folds, 337 participants, all training rows.
CALIBRATION = {"lr-cca:32": 0.7408, "lr-cca:32+lags1": 0.7703}


def make_lags(z, lags=0):
    """Stack ``z`` at offsets ``-lags..+lags`` along the feature axis.

    Edge padding, not zero padding: a run's first and last TRs have no earlier
    or later neighbour, and padding them with zeros would inject a fake
    excursion to the corpus mean at exactly the rows a readout sees as extreme.
    """
    if lags == 0:
        return z
    parts = []
    for lag in range(-lags, lags + 1):
        if lag < 0:
            parts.append(np.pad(z[:lag], ((-lag, 0), (0, 0)), mode="edge"))
        elif lag > 0:
            parts.append(np.pad(z[lag:], ((0, lag), (0, 0)), mode="edge"))
        else:
            parts.append(z)
    return np.concatenate(parts, axis=1)


def cca_avg(rec, k=DEFAULT_K):
    """The shipped feature: the L/R average of the leading ``k`` canonical directions."""
    z = rec["z"]
    return 0.5 * (z[:, 0, :k] + z[:, 1, :k]).astype(np.float64)


def incumbent(k=DEFAULT_K, lags=DEFAULT_LAGS):
    """``lr-cca:k`` plus temporal context -- the arm this package ships."""
    return lambda rec: make_lags(cca_avg(rec, k), lags)


# --------------------------------------------------------------------------- #
# Cache: labeled participants reduced to canonical coordinates once.
# --------------------------------------------------------------------------- #

def corpus_fingerprint(recs):
    """A hash of which participants, at what length, under which label convention."""
    items = sorted((r["dataset"], r["subject"], int(r["labels"].shape[0])) for r in recs)
    return hashlib.sha1(
        json.dumps([LABEL_CONVENTION, items], separators=(",", ":")).encode("utf8")
    ).hexdigest()


def build_cache(root, mask, arrays, m=256, verbose=True):
    """Every gaze-labeled participant as per-TR canonical coords + ``[T, 10, 2]`` labels."""
    import h5py

    flat_mask = mask.reshape(-1)
    recs = []
    for ds_dir in sorted(p for p in Path(root).glob("dsL*") if p.is_dir()):
        for path in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(path, "r") as f:
                    if "labels" not in f:
                        continue
                    block, labels = f["eye_block"][:], f["labels"][:]
            except Exception:
                continue
            t = block.shape[-1]
            if t < MIN_TRS or not np.isfinite(labels).any():
                continue

            rows = block.reshape(-1, t).T[:, flat_mask].astype(np.float64)
            zl, zr = orbit_projections(rows, arrays, k=m)
            n = min(len(zl), len(labels))
            recs.append({
                "dataset": ds_dir.name,
                "subject": path.stem,
                "z": np.stack([zl[:n], zr[:n]], axis=1).astype(np.float32),  # [T, 2, m]
                "labels": labels[:n].astype(np.float32),                     # [T, 10, 2]
            })
        if verbose:
            got = sum(1 for r in recs if r["dataset"] == ds_dir.name)
            print(f"    {ds_dir.name:<28}{got:>4} participants", flush=True)
    if not recs:
        raise SystemExit("[!] no labeled participants found")
    return recs


def save_cache(path, recs, basis_path, m):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"z/{i}": r["z"] for i, r in enumerate(recs)}
    flat.update({f"y/{i}": r["labels"] for i, r in enumerate(recs)})
    np.savez(path, version=np.array([CACHE_VERSION]), n=np.array([len(recs)]),
             m=np.array([m]), basis=np.array(str(basis_path)),
             fingerprint=np.array(corpus_fingerprint(recs)),
             meta=np.array(json.dumps([{"dataset": r["dataset"], "subject": r["subject"]}
                                       for r in recs])),
             **flat)


def load_cache(path, basis_path, m):
    """Load, refusing anything built for a different corpus, basis or rank.

    The guard includes the corpus fingerprint because the failure it prevents is
    silent: an on-disk cache of a retired corpus loads without complaint and
    answers a question nobody asked.
    """
    d = np.load(path, allow_pickle=False)
    got = (int(d["version"][0]), int(d["m"][0]), str(d["basis"]))
    want = (CACHE_VERSION, int(m), str(basis_path))
    if got != want:
        raise SystemExit(f"[!] cache built for {got}, requested {want}; rebuild with --build-cache")
    meta = json.loads(str(d["meta"]))
    recs = [{**meta[i], "z": d[f"z/{i}"], "labels": d[f"y/{i}"]}
            for i in range(int(d["n"][0]))]
    if str(d["fingerprint"]) != corpus_fingerprint(recs):
        raise SystemExit("[!] cache fingerprint does not match its own contents")
    return recs


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _resolutions(pred_20, labels):
    """``[T, 20]`` predictions + ``[T, 10, 2]`` gaze -> per-resolution ``(pred, true)``.

    ``ok`` requires all 20 target components finite, which is the protocol's row
    rule. A model scored on a different row set has a different denominator in
    every metric, so this is applied identically to every arm.
    """
    t_n = len(pred_20)
    lab = np.asarray(labels)[:t_n]
    ok = np.isfinite(lab.reshape(t_n, 20)).all(axis=1) & np.isfinite(pred_20).all(axis=1)
    if ok.sum() < 10:
        return None
    pred = np.asarray(pred_20)[:t_n].reshape(t_n, 10, 2)
    return {
        "subtr": (pred[ok].reshape(-1, 2), lab[ok].reshape(-1, 2)),
        "1tr": (np.nanmean(pred[ok], axis=1), np.nanmean(lab[ok], axis=1)),
    }


def lodo(recs, feature_fn, seed=0, alphas=ALPHAS):
    """Leave-one-dataset-out, every metric, at sub-TR and 1-TR resolution.

    Returns ``{"participants": [...], "folds": {...}, "summary": {...}}``.

    R-squared and Euclidean error are calibrated per participant against the
    *other* participants of its own held-out dataset (see
    :mod:`deepmreye.metrics`); Pearson r is not calibrated and needs no
    calibration. A fold with fewer than two participants therefore reports r
    only.
    """
    from sklearn.linear_model import RidgeCV

    feats = {id(r): np.asarray(feature_fn(r), dtype=np.float64) for r in recs}
    datasets = sorted({r["dataset"] for r in recs})
    rows = []

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
            sd = y[ok].std(axis=0)
            sd[sd < 1e-9] = 1.0
            xs.append(x[ok])
            ys.append((y[ok] - y[ok].mean(axis=0)) / sd)
        if not xs:
            continue

        model = RidgeCV(alphas=alphas).fit(np.concatenate(xs), np.concatenate(ys))

        # Predict every test participant once, then calibrate each against the rest.
        preds = {}
        for s in test:
            got = _resolutions(model.predict(feats[id(s)]), s["labels"])
            if got is not None:
                preds[s["subject"]] = got

        for sub, per_res in preds.items():
            row = {"dataset": held, "subject": sub}
            for res, (p, t) in per_res.items():
                others = [preds[o][res] for o in preds if o != sub]
                if others:
                    op = np.concatenate([o[0] for o in others])
                    ot = np.concatenate([o[1] for o in others])
                    gain, offset = metrics.fit_affine(op, ot)
                else:
                    gain = offset = None
                row[res] = metrics.score(p, t, gain, offset)
            rows.append(row)

    return _summarise(rows, datasets)


def _summarise(rows, datasets):
    """Per-participant rows -> per-fold medians -> across-fold summary.

    Median over participants inside a fold, then median over folds. Pooling
    every row of every participant into one correlation is gameable: if one
    participant's gaze sits left of another's, a model that predicts only *which
    participant this is* scores a high pooled r with zero within-participant
    decoding.
    """
    keys = ("r", "r_x", "r_y", "r2", "euclid_median", "euclid_mean", "gain_x", "gain_y")
    folds = {}
    for ds in datasets:
        members = [r for r in rows if r["dataset"] == ds]
        if not members:
            continue
        folds[ds] = {"n": len(members)}
        for res in ("subtr", "1tr"):
            folds[ds][res] = {k: metrics.nanmedian([m[res][k] for m in members]) for k in keys}

    summary = {}
    for res in ("subtr", "1tr"):
        vals = {k: [folds[d][res][k] for d in folds] for k in keys}
        summary[res] = {f"median_{k}": metrics.nanmedian(vals[k]) for k in keys}
        summary[res]["mean_r"] = float(np.mean([v for v in vals["r"] if np.isfinite(v)]))
    return {"participants": rows, "folds": folds, "summary": summary,
            "median_subtr": summary["subtr"]["median_r"],
            "median_1tr": summary["1tr"]["median_r"]}


def calibrate(recs, tol=0.01, verbose=True):
    """Reproduce the known headline numbers, or refuse to be trusted."""
    arms = {"lr-cca:32": incumbent(32, 0), "lr-cca:32+lags1": incumbent(32, 1)}
    ok = True
    for name, fn in arms.items():
        got = lodo(recs, fn)["median_subtr"]
        want = CALIBRATION[name]
        hit = abs(got - want) <= tol
        ok &= hit
        if verbose:
            print(f"  {name:<20} sub-TR {got:.4f}  expected {want:.4f}  "
                  f"{'OK' if hit else 'MISMATCH'}", flush=True)
    return ok


def load_or_build(data_dir=None, basis="results/basis.npz",
                  cache="results/labeled_cache.npz", m=256, rebuild=False):
    """The one way to get evaluation-ready records."""
    from deepmreye.datasource import resolve
    from deepmreye.unsupervised import corpus_mask, load_basis

    data_dir = Path(data_dir) if data_dir else resolve(None, download=False, quiet=True)
    if rebuild or not Path(cache).exists():
        print(f"[*] building cache from {data_dir}", flush=True)
        mask = corpus_mask(data_dir)
        _m, bases, _meta = load_basis(basis)
        recs = build_cache(data_dir, mask, bases["lr-cca"], m=m)
        save_cache(cache, recs, basis, m)
        print(f"[+] {len(recs)} participants -> {cache}")
    else:
        recs = load_cache(cache, basis, m)
        print(f"[*] loaded {len(recs)} participants from {cache}")
    return recs


def report(res, title=""):
    """Print the per-fold table and the summary."""
    for name, res_key in (("sub-TR (10 samples/TR)", "subtr"), ("1-TR mean gaze", "1tr")):
        print(f"\n=== {title}  --  {name} ===")
        print(f"{'dataset':<26}{'n':>4}{'r':>8}{'r_x':>8}{'r_y':>8}"
              f"{'R2*':>8}{'err*':>8}{'gain':>7}")
        for ds, f in res["folds"].items():
            d = f[res_key]
            print(f"{ds:<26}{f['n']:>4}{d['r']:>8.3f}{d['r_x']:>8.3f}{d['r_y']:>8.3f}"
                  f"{d['r2']:>8.3f}{d['euclid_median']:>8.2f}{d['gain_x']:>7.2f}")
        s = res["summary"][res_key]
        print(f"{'MEDIAN over folds':<26}{'':>4}{s['median_r']:>8.3f}"
              f"{s['median_r_x']:>8.3f}{s['median_r_y']:>8.3f}"
              f"{s['median_r2']:>8.3f}{s['median_euclid_median']:>8.2f}")
        print(f"{'MEAN over folds':<26}{'':>4}{s['mean_r']:>8.3f}")
    print("\n* R2 and err (degrees of visual angle) are calibrated per participant "
          "against\n  the other participants of its own held-out dataset; r is not "
          "calibrated.")
