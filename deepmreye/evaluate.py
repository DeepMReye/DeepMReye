"""The evaluation protocol, written down once.

Leave-one-dataset-out over the gaze-labelled datasets: the readout is fitted on
all datasets but one and scored on every participant of the one held out, so
every number answers "does this transfer to a study it has never seen".

Two resolutions, always both. The labels are ``[T, 10, 2]`` -- ten gaze samples
inside every volume -- so the same predictions are scored **sub-TR** (all ten
samples) and **per TR** (their mean). Each participant gets a Pearson r per axis
(averaged over x and y), R-squared and the median Euclidean error; a dataset is
the median over its participants, and the summary is the median and the mean
over datasets.

R-squared and error need a calibration, because predictions are in the training
datasets' standard units: one gain and offset per axis is fitted on the *other*
participants of the held-out dataset (:mod:`deepmreye.metrics`). Pearson r needs
none.

On the published corpus (nine datasets, 337 participants) and the published
basis this reproduces a sub-TR median of 0.771 (0.835 per TR, mean over datasets
0.721); :func:`calibrate` checks that before a run is trusted.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from deepmreye import metrics
from deepmreye.basis import orbit_projections
from deepmreye.model import ALPHAS, DEFAULT_K, DEFAULT_LAGS, fit_ridge, make_lags

CACHE_VERSION = 3

# The label convention the cache was built under (screen coordinates, y grows
# downwards). Bump if it ever changes, so that an old cache cannot be read as new.
LABEL_CONVENTION = "y-down-2026-08-20"

MIN_TRS = 60

# Sub-TR medians this module must reproduce on the published corpus and basis
# before anything it reports is quotable.
CALIBRATION = {"lr-cca:32": 0.7412, "lr-cca:32+lags1": 0.7712}


def bilateral(rec, k=DEFAULT_K):
    """The decoder's feature from a cached record: the two orbits' average."""
    z = rec["z"]
    return 0.5 * (z[:, 0, :k] + z[:, 1, :k]).astype(np.float64)


def features(k=DEFAULT_K, lags=DEFAULT_LAGS):
    """The shipped feature function: ``k`` canonical pairs with ``lags`` of context."""
    return lambda rec: make_lags(bilateral(rec, k), lags)


def orbit_views(k=DEFAULT_K, lags=DEFAULT_LAGS):
    """Each orbit alone, for :func:`lodo`'s ``views=`` (the label-free confidence)."""
    return {name: (lambda rec, i=i: make_lags(rec["z"][:, i, :k].astype(np.float64), lags))
            for i, name in enumerate(("left", "right"))}


# --------------------------------------------------------------------------- #
# Cache: labelled participants reduced to canonical coordinates once.
# --------------------------------------------------------------------------- #

def corpus_fingerprint(recs):
    """A hash of which participants, at what length, under which label convention."""
    items = sorted((r["dataset"], r["subject"], int(r["labels"].shape[0])) for r in recs)
    return hashlib.sha1(
        json.dumps([LABEL_CONVENTION, items], separators=(",", ":")).encode("utf8")
    ).hexdigest()


def build_cache(root, mask, arrays, m=256, pattern="dsL*", verbose=True):
    """Every gaze-labelled participant under ``root/<pattern>/`` as per-TR
    canonical coordinates ``z [T, 2, m]`` (left, right) plus ``labels [T, 10, 2]``."""
    import h5py

    flat_mask = mask.reshape(-1)
    recs = []
    for ds_dir in sorted(p for p in Path(root).glob(pattern) if p.is_dir()):
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
                "z": np.stack([zl[:n], zr[:n]], axis=1).astype(np.float32),
                "labels": labels[:n].astype(np.float32),
            })
        if verbose:
            got = sum(1 for r in recs if r["dataset"] == ds_dir.name)
            print(f"    {ds_dir.name:<28}{got:>4} participants", flush=True)
    if not recs:
        raise SystemExit(f"[!] no labelled participants found under {root}/{pattern}")
    return recs


def save_cache(path, recs, basis_path, m):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"z/{i}": r["z"] for i, r in enumerate(recs)}
    flat.update({f"y/{i}": r["labels"] for i, r in enumerate(recs)})
    np.savez(path, version=np.array([CACHE_VERSION]), n=np.array([len(recs)]),
             m=np.array([m]), basis=np.array(str(Path(basis_path).resolve())),
             fingerprint=np.array(corpus_fingerprint(recs)),
             meta=np.array(json.dumps([{"dataset": r["dataset"], "subject": r["subject"]}
                                       for r in recs])),
             **flat)


def load_cache(path, basis_path, m):
    """Load, refusing anything built for a different basis, rank or corpus."""
    d = np.load(path, allow_pickle=False)
    got = (int(d["version"][0]), int(d["m"][0]), str(d["basis"]))
    want = (CACHE_VERSION, int(m), str(Path(basis_path).resolve()))
    if got != want:
        raise SystemExit(f"[!] cache built for {got}, requested {want}; rebuild it")
    meta = json.loads(str(d["meta"]))
    recs = [{**meta[i], "z": d[f"z/{i}"], "labels": d[f"y/{i}"]}
            for i in range(int(d["n"][0]))]
    if str(d["fingerprint"]) != corpus_fingerprint(recs):
        raise SystemExit("[!] cache fingerprint does not match its own contents")
    return recs


def load_or_build(data_dir, basis, cache=None, m=256, rebuild=False, pattern="dsL*"):
    """Evaluation-ready records for every labelled participant under ``data_dir``.

    The basis file's own eye mask is used, so the records match what the
    decoder computes. ``cache`` (optional) keeps the projections between runs.
    """
    from deepmreye.basis import load_basis

    if cache and not rebuild and Path(cache).exists():
        recs = load_cache(cache, basis, m)
        print(f"[*] loaded {len(recs)} participants from {cache}")
        return recs
    print(f"[*] projecting labelled participants under {data_dir}", flush=True)
    mask, bases, _meta = load_basis(basis)
    recs = build_cache(data_dir, mask, bases["lr-cca"], m=m, pattern=pattern)
    if cache:
        save_cache(cache, recs, basis, m)
        print(f"[+] {len(recs)} participants -> {cache}")
    return recs


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _resolutions(pred_20, labels):
    """``[T, 20]`` predictions + ``[T, 10, 2]`` gaze -> per-resolution ``(pred, true)``.

    A row counts when all 20 targets and all predictions are finite -- the same
    rule for every decoder, so that every decoder has the same denominator.
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


def lodo(recs, feature_fn, alphas=ALPHAS, train_filter=None, readout=None,
         keep_predictions=False, views=None):
    """Leave-one-dataset-out, every metric, at sub-TR and per-TR resolution.

    Returns ``{"participants": [...], "folds": {...}, "summary": {...},
    "median_subtr", "median_1tr"}``.

    ``train_filter(rec, held) -> bool`` restricts the **training** side only
    (for example to fewer labelled datasets); every fold is still scored on its
    whole held-out dataset. ``readout(x, y, groups) -> predict`` replaces the
    ridge. ``keep_predictions`` adds ``"predictions"``, ``{(dataset, subject):
    [T, 20]}`` in the readout's z-units. ``views`` is ``{name: feature_fn}``:
    each test participant's features under another function, put through the
    same fitted readout and returned unscored as ``"views"`` -- for the two
    orbits alone, their agreement is the label-free confidence.
    """
    feats = {id(r): np.asarray(feature_fn(r), dtype=np.float64) for r in recs}
    datasets = sorted({r["dataset"] for r in recs})
    rows, kept = [], {}
    viewed = {name: {} for name in (views or {})}

    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        test = [r for r in recs if r["dataset"] == held]
        if train_filter is not None:
            train = [r for r in train if train_filter(r, held)]
        if not test:
            continue
        fitted = fit_ridge([(r["dataset"], feats[id(r)], r["labels"]) for r in train],
                           alphas, readout)
        if fitted is None:
            continue
        predict = fitted.predict if readout is None else fitted

        # Predict every test participant once, then calibrate each against the rest.
        preds = {}
        for s in test:
            pred = predict(feats[id(s)])
            if keep_predictions:
                kept[(s["dataset"], s["subject"])] = pred
            for name, fn in (views or {}).items():
                viewed[name][(s["dataset"], s["subject"])] = predict(
                    np.asarray(fn(s), dtype=np.float64))
            got = _resolutions(pred, s["labels"])
            if got is not None:
                preds[(s["dataset"], s["subject"])] = got
        rows += _score_fold(preds)

    res = _summarise(rows, datasets)
    if keep_predictions:
        res["predictions"] = kept
    if views:
        res["views"] = viewed
    return res


def _score_fold(preds):
    """``{(dataset, subject): {res: (pred, true)}}`` -> one scored row each.

    R-squared and error are calibrated per participant against the *other*
    participants of the same dataset in the same fold; r needs no calibration.
    """
    rows = []
    for (ds, sub), per_res in preds.items():
        row = {"dataset": ds, "subject": sub}
        for res, (p, t) in per_res.items():
            others = [preds[o][res] for o in preds if o != (ds, sub) and o[0] == ds]
            if others:
                op = np.concatenate([o[0] for o in others])
                ot = np.concatenate([o[1] for o in others])
                gain, offset = metrics.fit_affine(op, ot)
            else:
                gain = offset = None
            row[res] = metrics.score(p, t, gain, offset)
        rows.append(row)
    return rows


def _summarise(rows, datasets):
    """Per-participant rows -> per-dataset medians -> median and mean over datasets.

    Median over participants inside a dataset first. Pooling every participant's
    rows into one correlation is gameable: a model that only recognises *which
    participant this is* scores a high pooled r with no gaze decoding at all.
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
    """Reproduce the published sub-TR medians, or refuse to be trusted."""
    arms = {"lr-cca:32": features(32, 0), "lr-cca:32+lags1": features(32, 1)}
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


def report(res, title=""):
    """Print the per-dataset table and the summary."""
    for name, res_key in (("sub-TR (10 samples per TR)", "subtr"), ("per TR (mean gaze)", "1tr")):
        print(f"\n=== {title}  --  {name} ===")
        print(f"{'dataset':<26}{'n':>4}{'r':>8}{'r_x':>8}{'r_y':>8}"
              f"{'R2*':>8}{'err*':>8}{'gain':>7}")
        for ds, f in res["folds"].items():
            d = f[res_key]
            print(f"{ds:<26}{f['n']:>4}{d['r']:>8.3f}{d['r_x']:>8.3f}{d['r_y']:>8.3f}"
                  f"{d['r2']:>8.3f}{d['euclid_median']:>8.2f}{d['gain_x']:>7.2f}")
        s = res["summary"][res_key]
        print(f"{'MEDIAN over datasets':<26}{'':>4}{s['median_r']:>8.3f}"
              f"{s['median_r_x']:>8.3f}{s['median_r_y']:>8.3f}"
              f"{s['median_r2']:>8.3f}{s['median_euclid_median']:>8.2f}")
        print(f"{'MEAN over datasets':<26}{'':>4}{s['mean_r']:>8.3f}")
    print("\n* R2 and err (degrees of visual angle) are calibrated per participant against\n"
          "  the other participants of its own held-out dataset; r is not calibrated.")
