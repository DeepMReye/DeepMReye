"""After the readout: a confidence that needs no labels, the participants it
keeps, pseudo-labels, and viewing priors across participants."""
import json
import time

import numpy as np

from deepmreye import probe
from unlabeled.common import (
    consensus_prior, keep_most_confident, log, orbit_views, per_participant, pls_readout,
    ridge_auto, ridge_by_covariance, slicer, summary, voxel_block,
)

CONFIDENCE_FRACS = (0.9, 0.8, 0.5)
PARTICIPANT_SCORES = ("agreement", "decoded sd", "canonical agreement", "random-basis agreement", "volumes")


def random_orbits(ctx, k=32, seed=0):
    """The untrained control for orbit agreement: ``k`` random orthonormal
    directions per orbit, on the shipped basis's own orbit split and mean."""
    _m, bases, _meta = ctx.basis()
    lr = bases["lr-cca"]
    rng = np.random.default_rng(seed)
    W = np.zeros((int(ctx.mask.sum()), 2 * k), dtype=np.float32)
    for side, idx in enumerate((lr["left_index"], lr["right_index"])):
        W[idx, side * k:(side + 1) * k] = np.linalg.qr(rng.normal(size=(len(idx), k)))[0]
    return ("random", W, lr["mean"], (2, k))


def _zscore(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def _tr_table(rec, pred, left, right, za):
    """Label-free descriptors of every volume's decoding, and its 1-TR error.

    The error is in each participant's own z-units, so it ranks volumes within
    a participant -- which is all a per-volume confidence is for.
    """
    y = rec["labels"].reshape(len(rec["labels"]), 20)
    ok = np.isfinite(y).all(1)
    one_tr = lambda p: np.asarray(p).reshape(len(p), 10, 2).mean(1)  # noqa: E731
    a1 = one_tr(pred)
    step = np.sqrt((np.diff(a1, axis=0) ** 2).sum(1))
    d = {"disagreement": np.sqrt(((one_tr(left) - one_tr(right)) ** 2).sum(1)),
         "eccentricity": np.sqrt((_zscore(a1) ** 2).sum(1)),
         "velocity": np.r_[step[:1], step] + np.r_[step, step[-1:]],
         "sub-TR spread": pred.reshape(len(pred), 10, 2).std(1).sum(1),
         "feature step": np.r_[0, np.sqrt((np.diff(za, axis=0) ** 2).sum(1))],
         "feature energy": np.sqrt((za ** 2).sum(1))}
    err = np.sqrt(((_zscore(a1[ok]) - _zscore(one_tr(y)[ok])) ** 2).sum(1))
    return {n: v[ok] for n, v in d.items()}, err


def confidence(ctx, out):
    """Orbit agreement as a confidence -- the shipped readout applied to each
    orbit alone (``views=``), and how well the two decodings agree -- scored per
    participant (does it rank them, flag the worst fifth, what is left after
    dropping the least confident) and per volume, against hand-made scores, a
    random-basis control, and an error model trained on other datasets' errors
    as DeepMReye 1.0's predicted error was. Then pseudo-labels: the frozen
    decoder labels each dataset's own participants and a student readout is
    fitted on them, with and without the confidence and a consensus prior."""
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import roc_auc_score

    from deepmreye.metrics import nanmedian, orbit_agreement

    k, name = 32, f"lrcca_n{ctx.n}"
    recs, index = ctx.records([ctx.n], extra=[random_orbits(ctx, k)])
    fn, flat = slicer(index, name, k=k, lags=1), slicer(index, name, k=k, lags=0)
    views = orbit_views(index, name, k)
    res = probe.lodo(recs, fn, keep_predictions=True, views=views)
    ctl = probe.lodo(recs, slicer(index, "random", k=k, lags=1), keep_predictions=True,
                     views=orbit_views(index, "random", k))
    print(f"  protocol {res['median_subtr']:.4f}; random-basis control {ctl['median_subtr']:.4f}", flush=True)

    # ---- per participant ----
    r_of = {(q["dataset"], q["subject"]): q["subtr"]["r"] for q in res["participants"]}
    a, b, shape = index[name]
    rows = []
    for rec in recs:
        key = (rec["dataset"], rec["subject"])
        if key not in r_of:
            continue
        z = rec["z"][:, a:b].astype(np.float64).reshape(len(rec["z"]), *shape)[:, :, :k]
        rho = orbit_agreement(res["views"]["left"][key], res["views"]["right"][key])
        rows.append({
            "dataset": key[0], "subject": key[1], "r": r_of[key], "agreement": rho,
            # Spearman-Brown: what the averaged trace would correlate with the
            # orbits' common signal at. A ceiling, not a prediction of r.
            "ceiling": float(np.sqrt(2 * max(rho, 0.0) / (1 + max(rho, 0.0)))),
            "decoded sd": float(res["predictions"][key].reshape(-1, 2).std(0).mean()),
            "canonical agreement": float(np.mean([np.corrcoef(z[:, 0, j], z[:, 1, j])[0, 1] for j in range(k)])),
            "random-basis agreement": orbit_agreement(ctl["views"]["left"][key], ctl["views"]["right"][key]),
            "volumes": int(len(rec["z"]))})
    names = sorted({q["dataset"] for q in rows})
    table = {}
    for s in PARTICIPANT_SCORES + ("oracle",):
        col = "r" if s == "oracle" else s
        sp, auc = {}, {}
        for d in names:
            v = np.array([q[col] for q in rows if q["dataset"] == d])
            r = np.array([q["r"] for q in rows if q["dataset"] == d])
            sp[d] = float(spearmanr(v, r).correlation)
            worst = r <= np.quantile(r, 0.2)
            auc[d] = float(roc_auc_score(worst, -v)) if 0 < worst.sum() < len(r) else None
        table[s] = {"spearman": nanmedian(sp.values()), "auroc_worst20": nanmedian(auc.values()),
                    "spearman_per_dataset": sp, "auroc_per_dataset": auc,
                    **{f"keep{int(f * 100)}": keep_most_confident(rows, col, f) for f in (1.0,) + CONFIDENCE_FRACS}}
    rng, draws = np.random.default_rng(0), []
    for _ in range(200):
        for q in rows:
            q["_draw"] = rng.random()
        draws.append([keep_most_confident(rows, "_draw", f) for f in CONFIDENCE_FRACS])
    for q in rows:
        del q["_draw"]
    table["random"] = {f"keep{int(f * 100)}": np.mean([d_[i] for d_ in draws], axis=0).tolist()
                       for i, f in enumerate(CONFIDENCE_FRACS)}
    table["ceiling_by_dataset"] = {
        d: {"ceiling": float(np.median([q["ceiling"] for q in rows if q["dataset"] == d])),
            "r": float(np.median([q["r"] for q in rows if q["dataset"] == d]))} for d in names}
    record = {"participants": rows, "participant_summary": table}
    json.dump(record, open(out, "w"), indent=1)
    for s in PARTICIPANT_SCORES + ("oracle", "random"):
        x = table[s]
        print(f"  {s:<24} spearman {x.get('spearman', float('nan')):5.2f}  auroc "
              f"{x.get('auroc_worst20', float('nan')):5.2f}  keep80 {x['keep80'][0]:.3f}  "
              f"keep50 {x['keep50'][0]:.3f}", flush=True)

    # ---- per volume: hand-made scores, and a learned error model (DeepMReye 1.0's
    # predicted error, in kind). Nested: the error model for a held dataset is fitted
    # on the other datasets' errors, each decoded by a readout that saw neither.
    per_volume = {}
    for held in names:
        inner = probe.lodo(recs, fn, keep_predictions=True, views=views,
                           train_filter=lambda r, h, held=held: r["dataset"] != held)
        xs, es = [], []
        for rec in recs:
            key = (rec["dataset"], rec["subject"])
            if rec["dataset"] == held or key not in inner["predictions"]:
                continue
            d, err = _tr_table(rec, inner["predictions"][key], inner["views"]["left"][key],
                               inner["views"]["right"][key], flat(rec))
            if len(err) >= 30:
                xs.append(np.stack([_zscore(v) for v in d.values()], 1))
                es.append(_zscore(np.log(err ** 2 + 1e-3)))
        x_all, e_all = np.concatenate(xs), np.concatenate(es)
        learned = {"learned (ridge)": RidgeCV(alphas=probe.ALPHAS).fit(x_all, e_all),
                   "learned (boosted trees)": HistGradientBoostingRegressor(
                       max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(x_all, e_all)}
        got = {}
        for rec in recs:
            key = (rec["dataset"], rec["subject"])
            if rec["dataset"] != held or key not in res["predictions"]:
                continue
            d, err = _tr_table(rec, res["predictions"][key], res["views"]["left"][key],
                               res["views"]["right"][key], flat(rec))
            if len(err) < 30:
                continue
            x = np.stack([_zscore(v) for v in d.values()], 1)
            scores = {**d, **{m: model.predict(x) for m, model in learned.items()}, "oracle": err}
            for s, v in scores.items():
                order = np.argsort(v)
                got.setdefault(s, []).append(
                    [float(spearmanr(v, err).correlation)]
                    + [float(np.mean(err[order[:int(round(f * len(err)))]] ** 2) / np.mean(err ** 2))
                       for f in CONFIDENCE_FRACS])
        per_volume[held] = {s: np.median(v, axis=0).tolist() for s, v in got.items()}
        print(f"  {held:<28}" + "".join(f"{s} {v[0]:.2f}  " for s, v in per_volume[held].items()), flush=True)
    record["per_volume"] = {
        "columns": ["spearman"] + [f"sq_error_keep{int(f * 100)}" for f in CONFIDENCE_FRACS],
        "by_dataset": per_volume,
        "median": {s: np.median([per_volume[d][s] for d in names], axis=0).tolist()
                   for s in per_volume[names[0]]}}
    json.dump(record, open(out, "w"), indent=1)

    # ---- pseudo-labels ----
    teacher = res["predictions"]
    disagreement = {key: np.sqrt(((res["views"]["left"][key] - res["views"]["right"][key]) ** 2).mean(1))
                    for key in teacher}

    def student(targets, keep=None):
        """A readout fitted inside each dataset on pseudo-labels (5-fold, the
        ``within`` split); returns its predictions, to be scored on gaze."""
        pseudo = []
        for rec in recs:
            key = (rec["dataset"], rec["subject"])
            y = np.array(targets[key], dtype=np.float32)
            if keep is not None:
                y[~keep[key]] = np.nan
            pseudo.append({**rec, "labels": y.reshape(len(y), 10, 2)})
        return probe.lodo(pseudo, fn, split="within", keep_predictions=True)["predictions"]

    confident = {key: v <= np.median(v) for key, v in disagreement.items()}
    consensus = consensus_prior(recs, teacher, 1.0, 1.0)
    st_cons = student(consensus)
    arms = {
        "frozen decoder (teacher)": teacher,
        "student on pseudo-labels": student(teacher),
        "student on pseudo-labels, most-agreeing half of volumes": student(teacher, confident),
        "teacher + consensus prior": consensus,
        "student on consensus pseudo-labels": st_cons,
        "student on consensus pseudo-labels + consensus prior": consensus_prior(recs, st_cons, 1.0, 1.0),
    }
    cut = {d: np.quantile([q["agreement"] for q in rows if q["dataset"] == d], 0.2) for d in names}
    agree = {(q["dataset"], q["subject"]): q["agreement"] for q in rows}
    filters = {"readout without the 20% least-agreeing labelled participants":
               lambda r, h: agree.get((r["dataset"], r["subject"]), 1.0) >= cut.get(r["dataset"], -1.0)}
    for seed in range(3):
        drop = {key: np.random.default_rng([seed, i]).random() < 0.2 for i, key in enumerate(sorted(agree))}
        filters[f"readout without a random 20% (seed {seed})"] = \
            lambda r, h, drop=drop: not drop.get((r["dataset"], r["subject"]), False)
    pseudo_rows = [summary(probe.score_predictions(recs, preds), {"arm": arm}) for arm, preds in arms.items()]
    pseudo_rows += [summary(probe.lodo(recs, fn, train_filter=keep), {"arm": arm}) for arm, keep in filters.items()]
    for row in pseudo_rows:
        print(f"  {row['arm']:<64} {row['subtr']:.4f} {row['1tr']:.4f} mean {row['subtr_mean']:.4f}", flush=True)
    record["pseudo_labels"] = pseudo_rows
    json.dump(record, open(out, "w"), indent=1)


def reliable(ctx, out):
    """Every arm of the corpus-size panel scored per participant at every
    corpus size, with the frozen basis's orbit agreement at that size, so each
    arm can be scored on all participants and on the same most-confident
    fraction. The selection is the frozen basis's (it is the only arm with two
    orbits to compare) and is applied to every arm alike."""
    from deepmreye.metrics import orbit_agreement

    # The 2x2x2 voxel ridge is in `schemes` ("other datasets, all eight"): with
    # every corpus size projected as well it does not fit in memory here.
    recs, index = ctx.records(extra=[voxel_block(ctx.mask, 3)])
    rows, t0 = {}, time.time()

    def flat(res):
        return {f"{q['dataset']}/{q['subject']}": q["subtr"] for q in per_participant(res)}

    for n in ctx.sizes:
        name = f"lrcca_n{n}"
        res = probe.lodo(recs, slicer(index, name, k=32, lags=1), keep_predictions=True,
                         views=orbit_views(index, name))
        rows[f"lr-cca:32+lags1@{n}"] = flat(res)
        rows[f"agreement@{n}"] = {f"{d}/{s}": orbit_agreement(res["views"]["left"][(d, s)],
                                                              res["views"]["right"][(d, s)])
                                  for (d, s) in res["predictions"]}
        rows[f"corpus-pca:64+lags1@{n}"] = flat(probe.lodo(recs, slicer(index, f"cpca_n{n}", k=64, lags=1)))
        log(f"N={n:<5} lr-cca {res['median_subtr']:.4f}", t0)
        json.dump(rows, open(out, "w"))
    for arm, block, readout in (("voxels:3+lags1", "vox3", ridge_by_covariance),
                                ("voxels:3 pls:64+lags1", "vox3", pls_readout(64))):
        res = probe.lodo(recs, slicer(index, block, lags=1), readout=readout)
        rows[arm] = flat(res)
        log(f"{arm:<24} {res['median_subtr']:.4f}", t0)
        json.dump(rows, open(out, "w"))


PRIORS = {"none": None, "consensus": (0.0, 1.0), "own+consensus": (1.0, 1.0), "own+2x consensus": (1.0, 2.0)}


def priors(ctx, out):
    """A label-free viewing prior: where every participant of a dataset saw the
    same stimulus on the same clock, the other participants' decoded gaze at
    the same moment is a prior on this one's. Applied after the readout, to the
    frozen basis and to the supervised voxel decoder alike, and scored by
    ``probe.score_predictions``."""
    recs, index = ctx.records([ctx.n], extra=[voxel_block(ctx.mask, 2)])
    arms = {f"lr-cca:32+lags1@{ctx.n}": (slicer(index, f"lrcca_n{ctx.n}", k=32, lags=1), None),
            "voxels:2+lags1": (slicer(index, "vox2", lags=1), ridge_auto)}
    rows, t0 = [], time.time()
    for arm, (fn, readout) in arms.items():
        preds = probe.lodo(recs, fn, readout=readout, keep_predictions=True)["predictions"]
        for prior, weights in PRIORS.items():
            res = probe.score_predictions(recs, preds if weights is None else consensus_prior(recs, preds, *weights))
            row = summary(res, {"arm": arm, "prior": prior})
            row["participants"] = per_participant(res)
            rows.append(row)
            log(f"{arm:<24} {prior:<18} sub-TR {row['subtr']:.4f}  1-TR {row['1tr']:.4f}  "
                f"mean {row['subtr_mean']:.4f}", t0)
            json.dump(rows, open(out, "w"), indent=1)
