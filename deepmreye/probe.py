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
# produces is quotable. 9 folds, 337 participants, all training rows, on the
# basis `scripts/fit_corpus_basis.py` writes to results/basis.npz (all 2,060
# eligible unlabelled participants of 652 datasets; 6 October 2026). The basis
# before it, a draw of 2,000 that held four evaluation participants under
# their OpenNeuro names, read 0.7408 / 0.7703.
CALIBRATION = {"lr-cca:32": 0.7412, "lr-cca:32+lags1": 0.7712}


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


# Which participants a readout is fitted on. `lodo` is the protocol and the only
# split a headline number may come from. The others hold the features fixed and
# move the training set, which is how transfer is separated from content:
# `within` fits one readout per dataset on its other participants, `pooled` fits
# one readout on every dataset including the test participant's own, and
# `halves` fits one readout per participant on the first half of its own run
# and scores the second half.
SPLITS = ("lodo", "within", "pooled", "halves")
N_FOLDS = 5


def _halves(recs, train_fraction=0.5):
    """Every run cut into a training record and a test record.

    The test record is always the second half of the run; the training record
    is its first ``train_fraction`` (at most a half), so shortening the
    calibration does not move the test set. In time order, not a random split
    of rows: neighbouring TRs are nearly the same sample, so interleaved rows
    would score the autocorrelation. Each part is lagged on its own (edge
    padding), so no test feature reads a training volume.
    """
    out = []
    for r in recs:
        h, t = len(r["z"]) // 2, int(round(len(r["z"]) * train_fraction))
        out.append({**r, "z": r["z"][:t], "labels": r["labels"][:t], "half": 0})
        out.append({**r, "z": r["z"][h:], "labels": r["labels"][h:], "half": 1})
    return out


def _splits(recs, split, shots=None, seed=0):
    """Yield ``(held, train, test)`` participant lists for one split."""
    datasets = sorted({r["dataset"] for r in recs})
    if split == "halves":                 # `recs` is already the output of _halves
        for first, second in zip(recs[::2], recs[1::2]):
            yield (first["dataset"], first["subject"]), [first], [second]
        return
    if split == "lodo":
        for held in datasets:
            yield (held, [r for r in recs if r["dataset"] != held],
                   [r for r in recs if r["dataset"] == held])
        return
    members = {d: [r for r in recs if r["dataset"] == d] for d in datasets}
    for f in range(N_FOLDS):
        # Every n-th participant of every dataset, so each is tested exactly once
        # and a dataset of six still has four to train on.
        test = [r for d in datasets for i, r in enumerate(members[d]) if i % N_FOLDS == f]
        ids = {id(r) for r in test}
        if split == "pooled":
            yield f, [r for r in recs if id(r) not in ids], test
        else:
            for j, d in enumerate(datasets):
                train = [r for r in members[d] if id(r) not in ids]
                if shots is not None:
                    # Nested in `shots` for a fixed seed: the one participant of
                    # a one-shot draw is also in the two-shot draw.
                    order = np.random.default_rng([seed, f, j]).permutation(len(train))
                    train = [train[i] for i in order[:shots]]
                yield d, train, [r for r in test if r["dataset"] == d]


def lodo(recs, feature_fn, seed=0, alphas=ALPHAS, train_filter=None,
         split="lodo", readout=None, shots=None, keep_predictions=False,
         train_fraction=0.5, views=None):
    """Leave-one-dataset-out, every metric, at sub-TR and 1-TR resolution.

    Returns ``{"participants": [...], "folds": {...}, "summary": {...}}``.

    R-squared and Euclidean error are calibrated per participant against the
    *other* test participants of its own dataset (see :mod:`deepmreye.metrics`);
    Pearson r is not calibrated and needs no calibration. A participant tested
    alone therefore reports r only.

    ``train_filter(rec, held) -> bool`` restricts the **training** side only,
    which is how a labeled-budget curve is measured: the readout sees fewer
    labeled participants while every fold is still scored on its whole held-out
    dataset. Subsetting the test side instead would move the denominator of the
    metric along with the budget and make the curve unreadable. ``None`` is the
    shipped protocol and changes nothing.

    ``split`` is one of :data:`SPLITS`. Anything but ``"lodo"`` is a diagnostic:
    the same features and the same scoring with the readout trained closer to
    the test participant, so the difference between splits is the cost of
    transfer and nothing else.

    ``readout(x, y, groups) -> predict`` replaces the ridge. ``groups`` is the
    participant index of every training row, for a readout that needs to hold
    participants out. ``None`` is ``RidgeCV`` over ``alphas``, the protocol.

    ``shots`` is the few-shot form of ``split="within"``: the readout is fitted
    on at most that many participants of the test participant's own dataset,
    drawn by ``seed``. The test side is the ``within`` one, untouched, so every
    participant is still scored exactly once and the denominator does not move
    with the number of shots.

    ``split="halves"`` is the within-participant form: one readout per
    participant, fitted on the first half of its run and scored on the second.
    Each is the only test participant of its fold, so it reports r only.
    ``train_fraction`` shortens the training part to the first that fraction of
    the run (at most 0.5); the test part stays the second half.

    ``keep_predictions`` adds ``"predictions"``, ``{(dataset, subject): [T, 20]}``
    in the readout's z-units, so that one call's output can be another call's
    feature -- a readout trained elsewhere, then calibrated on a few shots.

    ``views`` is ``{name: feature_fn}``: each test participant's features under
    another function, put through the same fitted readout and returned unscored
    as ``"views"``, ``{name: {(dataset, subject): [T, 20]}}``. For ``lr-cca``
    the views are the two orbits alone; the readout is linear in their average,
    so their two decodings average to the scored one exactly, and how far
    apart they are is a confidence that needs no labels.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if train_filter is not None and split != "lodo":
        raise ValueError("train_filter is defined per held-out dataset; use split='lodo'")
    if shots is not None and split != "within":
        raise ValueError("shots are drawn from the test participant's own dataset; "
                         "use split='within'")

    if not 0.0 < train_fraction <= 0.5:
        raise ValueError(f"train_fraction must be in (0, 0.5], got {train_fraction}")
    if train_fraction != 0.5 and split != "halves":
        raise ValueError("train_fraction is the length of a participant's own "
                         "calibration; use split='halves'")
    if split == "halves":
        recs = _halves(recs, train_fraction)
    feats = {id(r): np.asarray(feature_fn(r), dtype=np.float64) for r in recs}
    datasets = sorted({r["dataset"] for r in recs})
    rows, kept = [], {}
    viewed = {name: {} for name in (views or {})}

    for held, train, test in _splits(recs, split, shots, seed):
        if train_filter is not None:
            train = [r for r in train if train_filter(r, held)]
        if not test:
            continue

        predict = _fit(train, feats, alphas, readout)
        if predict is None:
            continue

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


def _fit(train, feats, alphas=ALPHAS, readout=None):
    """The readout on every finite row of ``train``, gaze z-scored per dataset.

    ``feats`` maps ``id(rec)`` to that record's feature matrix. Returns the
    fitted ``predict``, or ``None`` when no dataset has ten usable rows.
    """
    from sklearn.linear_model import RidgeCV

    # Which rows train, then one preallocated design matrix: a voxel-sized
    # feature set does not fit three copies of itself in memory.
    parts, ys, gs = [], [], []
    for ds in sorted({r["dataset"] for r in train}):
        members = [r for r in train if r["dataset"] == ds]
        y = np.concatenate([r["labels"].reshape(len(r["labels"]), 20) for r in members])
        g = np.concatenate([np.full(len(feats[id(r)]), len(gs) * 100000 + i)
                            for i, r in enumerate(members)])
        oks = [np.isfinite(feats[id(r)]).all(axis=1) for r in members]
        ok = np.isfinite(y).all(axis=1) & np.concatenate(oks)
        if ok.sum() < 10:
            continue
        sd = y[ok].std(axis=0)
        sd[sd < 1e-9] = 1.0
        ys.append((y[ok] - y[ok].mean(axis=0)) / sd)
        gs.append(g[ok])
        bounds = np.cumsum([0] + [len(o) for o in oks])
        parts += [(r, ok[a:b]) for r, a, b in zip(members, bounds[:-1], bounds[1:])]
    if not ys:
        return None

    y_all = np.concatenate(ys)
    x_all = np.empty((len(y_all), feats[id(parts[0][0])].shape[1]))
    pos = 0
    for r, ok in parts:
        n = int(ok.sum())
        x_all[pos:pos + n] = feats[id(r)][ok]
        pos += n
    if readout is None:
        return RidgeCV(alphas=alphas).fit(x_all, y_all).predict
    return readout(x_all, y_all, np.concatenate(gs))


def fit_readout(recs, feature_fn, alphas=ALPHAS, readout=None):
    """The protocol's readout fitted on *every* labelled participant, for decoding
    data that has no labels at all. Outputs are in the training datasets' gaze
    z-units, as every prediction :func:`lodo` keeps is."""
    feats = {id(r): np.asarray(feature_fn(r), dtype=np.float64) for r in recs}
    return _fit(recs, feats, alphas, readout)


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


def score_predictions(recs, predictions):
    """Score given ``{(dataset, subject): [T, 20]}`` predictions as :func:`lodo` does.

    Each dataset is one fold, which is what leave-one-dataset-out makes it, so
    for predictions kept from a ``lodo`` call this reproduces that call's
    result exactly. It exists for arms that transform predictions after the
    readout -- a prior applied to decoded gaze -- so that they are scored by
    the same code as everything else rather than by a second implementation.
    """
    datasets = sorted({r["dataset"] for r in recs})
    rows = []
    for d in datasets:
        preds = {}
        for r in recs:
            key = (r["dataset"], r["subject"])
            if r["dataset"] != d or key not in predictions:
                continue
            got = _resolutions(predictions[key], r["labels"])
            if got is not None:
                preds[key] = got
        rows += _score_fold(preds)
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
