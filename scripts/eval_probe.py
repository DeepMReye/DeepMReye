#!/usr/bin/env python3
"""The baseline table: how well can gaze be read out of raw fMRI voxels, and by what.

A probe number on its own says nothing. This crosses two axes so that it does.

**Generalization level** (``--protocol``), in increasing strictness. This is the
structure DeepMReye 1.0 reported, plus one:

- ``within``   -- same participant, early timepoints train and late ones test.
                  The easiest setting; it answers "does this work at all". Train
                  and test share no timepoint, but we have no run boundaries
                  stored, so they are temporally adjacent.
- ``subject``  -- held-out participants, same scanner and paradigm.
- ``dataset``  -- leave one dataset out, each in turn. A scanner and a
                  population the readout has never seen.
- ``paradigm`` -- leave one *paradigm* out. dsL02/03/04 are all smooth pursuit,
                  so holding out one of them alone still trains on the same
                  task; this is the honest unseen-task number.

**Feature source** (``--features``): see ``deepmreye/evaluate/features.py``.
``raw`` is the published baseline -- stride-``--voxel-stride`` voxels, mean-pooled
per temporal patch. The rest are linear bases over the *full* eye mask:
``fold-pca`` fitted on the training fold, and ``corpus-pca`` / ``diff-pca`` /
``lr-cca`` fitted once on the 1773 unlabeled participants
(``scripts/fit_corpus_basis.py``). Read them together: ``fold-pca`` minus ``raw``
is what the full mask is worth, and ``corpus-pca`` minus ``fold-pca`` is what
the unlabeled corpus is worth.

**Readout** (``--readouts``): see ``deepmreye/evaluate/baselines.py``.

Metrics are aggregated **per participant, then median across participants**.
Pooling every row of every subject into one correlation would let a model score
well by predicting only which subject it is looking at. ``--pooled`` also prints
the pooled numbers for comparison.

Pearson r is the headline here rather than R^2, deliberately: cross-dataset gaze
is mis-calibrated in gain (measured gains 0.11-2.27 against the training scale),
which destroys R^2 while leaving the correlation intact. That is a separate
problem from whether the voxels carry gaze -- see ``scripts/analyze_calibration.py``.

    python scripts/eval_probe.py --protocol within
    python scripts/eval_probe.py --protocol dataset --readouts mean linear ridge-cv pca-ridge pls
    python scripts/eval_probe.py --protocol dataset --readouts ridge-cv svr lgbm mlp
    python scripts/eval_probe.py --protocol dataset --readouts ridge-cv \
        --features raw fold-pca corpus-pca diff-pca lr-cca
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.data.probe_dataset import ProbeDataset, dataset_folds, paradigm_folds
from deepmreye.datasource import resolve
from deepmreye.evaluate.align import ALIGN_METHODS, apply_pair
from deepmreye.evaluate.baselines import ALL_READOUTS, DEFAULT_READOUTS, fit_readout, predict
from deepmreye.evaluate.features import (
    CORPUS_KINDS,
    FEATURE_KINDS,
    HYBRID_KINDS,
    JEPA_KINDS,
    CompositeExtractor,
    FeatureExtractor,
    JepaExtractor,
    parse_spec,
    pool_time,
)
from deepmreye.evaluate.probe import (
    aggregate_by_subject,
    compute_probe_metrics,
    flatten_valid_groups,
    temporal_targets,
)


def extract(loader, n_t, extractors, desc):
    """Features for every requested source, in one pass over the loader.

    The read is the expensive part -- a window is 1 MB off disk against a
    projection that costs microseconds -- so all feature sources share a single
    traversal and a single temporal pooling, and differ only in the linear map
    applied afterwards.
    """
    feats = {name: [] for name in extractors}
    targs, dsets, subs = [], [], []
    for x, y, ds, sub, _tr in tqdm(loader, desc=desc, leave=False):
        pooled = pool_time(x, n_t)
        for name, ex in extractors.items():
            feats[name].append(ex(pooled, x, n_t, subject_ids=sub))
        targs.append(temporal_targets(y, n_t))
        dsets.extend(ds)
        subs.extend(sub)
    if not targs:
        return None
    return ({name: np.concatenate(v) for name, v in feats.items()},
            np.concatenate(targs), np.array(dsets), np.array(subs))


def fit_fold_bases(extractors, train_ds, args):
    """Fit any fold-local basis on a subsample of the training split.

    A subsample, not the whole split: a 256-component PCA over 14236 voxels is
    settled by a few thousand rows, and reading every training window twice
    (once to fit the basis, once to apply it) would double the cost of the arm
    that exists only as a control.
    """
    # A composite shares its sub-extractors' fitting, so dedupe by identity.
    pending = []
    for ex in extractors.values():
        pending.extend(p for p in ex.parts
                       if p.needs_fit and not any(p is q for q in pending))
    if not pending:
        return
    subset = cap(train_ds, args.basis_fit_windows)
    loader = DataLoader(subset, batch_size=args.batch_size,
                        num_workers=args.num_workers)
    n_t = args.window_size // args.temp_patch_size
    rows = [[] for _ in pending]
    targs = [[] for _ in pending]
    sub_lists = [[] for _ in pending]
    for x, y, _ds, sub, _tr in tqdm(loader, desc="fit fold basis", leave=False):
        pooled = pool_time(x, n_t)
        yt = temporal_targets(y, n_t)
        for i, ex in enumerate(pending):
            sel = ex.select(pooled)
            rows[i].append(sel.reshape(-1, sel.shape[-1]))
            targs[i].append(yt.reshape(-1, yt.shape[-1]))
            sub_lists[i].extend(np.repeat(sub, sel.shape[1]))
    for i, ex in enumerate(pending):
        r = np.concatenate(rows[i])
        t = np.concatenate(targs[i])
        s = np.array(sub_lists[i])
        valid = ~np.isnan(t).any(axis=1)
        ex.fit(r[valid], targets=t[valid], subject_ids=s[valid], seed=args.seed)


def build_extractors(args, mask, bases):
    """One extractor per requested feature source; ``a+b`` concatenates.

    Sub-extractors are shared across specs by kind, so ``fold-pca`` and
    ``fold-pca+lr-cca`` in the same run fit one basis, not two -- and, more
    importantly, the *same* one, so the composite is exactly its parts.
    """
    shared = {}

    def part(kind, budget):
        key = (kind, budget)
        if key not in shared:
            if kind in JEPA_KINDS:
                shared[key] = build_jepa_extractor(kind, args, mask, budget)
                return shared[key]

            # `fold-shrunk-pca` is fitted per fold but shrinks toward the corpus, so
            # it needs `corpus-pca`'s arrays under its own name.
            needs = "corpus-pca" if kind == "fold-shrunk-pca" else kind
            if needs in CORPUS_KINDS and needs not in bases:
                raise SystemExit(
                    f"[!] feature source {kind!r} needs the {needs!r} basis. "
                    f"Run: python scripts/fit_corpus_basis.py --out {args.basis}")
            shared[key] = FeatureExtractor(
                kind, mask=mask, basis=bases.get(needs),
                n_components=budget or args.n_basis, stride=args.voxel_stride,
                shrink_lambda=args.shrink_lambda)
        return shared[key]

    out = {}
    for spec in args.features:
        kinds = parse_spec(spec)
        out[spec] = (part(*kinds[0]) if len(kinds) == 1
                     else CompositeExtractor(spec, [part(*k) for k in kinds]))
    return out


def build_jepa_extractor(kind, args, mask, budget):
    """The cross-orbit JEPA as a feature source, trained or untrained.

    Two things here are not boilerplate. The basis and geometry come from the
    **checkpoint**, not from `--basis`, because the encoder's input is the
    canonical pre-projection it was trained on and feeding it a different one
    silently scores a mismatched model -- the stale-cache failure `orbitcon`
    already hit. And `jepa-random` rebuilds the architecture from the
    checkpoint's own ``arch`` dict, so the control cannot drift from the model it
    controls (the `xrot` bug, which inflated a margin to +0.370 against a true
    +0.214).

    Because of the identity initialisation, that control *is* `lr-cca:k`. It is
    therefore a real baseline rather than a random projection, and the
    trained-minus-untrained margin is a margin over the best linear corpus arm.
    """
    from deepmreye.orbitjepa import load_checkpoint
    from deepmreye.unsupervised import load_basis

    ckpt_path = Path(args.jepa_checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"[!] no Orbit-JEPA checkpoint at {ckpt_path}. Run: "
            f"python scripts/train_orbitjepa.py --out {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path, untrained=kind.endswith("-random"))
    basis_mask, bases, _ = load_basis(ckpt["basis"])
    if basis_mask.shape != mask.shape or int(basis_mask.sum()) != int(mask.sum()):
        raise SystemExit(
            f"[!] {kind}: the checkpoint's basis mask ({int(basis_mask.sum())} "
            f"voxels) does not match the probe mask ({int(mask.sum())})")

    arch, k = ckpt["arch"], ckpt["arch"]["latent_dim"]
    width = 2 * k if ckpt["head"] == "concat" else k
    tag = "untrained control (== lr-cca:%d)" % k if kind.endswith("-random") else (
        "best epoch %s, val %.5f, nonlinear share %.3f" % (
            ckpt["meta"].get("best_epoch", "?"),
            ckpt["meta"].get("best_val_loss", float("nan")),
            ckpt["meta"].get("nonlinear_share", float("nan"))))
    print(f"    {kind}: {width} features (k={k}, M={ckpt['m']}, head={ckpt['head']}, "
          f"motion-regressed={ckpt['regress_motion']}), {tag}")

    return JepaExtractor(kind, mask, bases["lr-cca"], ckpt["weights"],
                         m=ckpt["m"], head=ckpt["head"],
                         regress_motion=ckpt["regress_motion"],
                         n_components=budget)


def load_bases_for(fold, args, data_dir):
    """Bases for one fold, honouring a ``{fold}`` placeholder in ``--basis``.

    A per-fold basis is how ``--include-labeled`` stays honest: the labeled
    datasets' voxels go into the fit, but the held-out one does not, so the
    number is still leave-one-dataset-out.
    """
    from deepmreye.unsupervised import corpus_mask, load_basis

    kinds = {k for spec in args.features for k, _budget in parse_spec(spec)}
    # `fold-shrunk-pca` shrinks toward the corpus covariance and the JEPA encodes
    # the corpus canonical pre-projection, so both need the basis file even
    # though neither is one of CORPUS_KINDS.
    if not (kinds & set(CORPUS_KINDS + HYBRID_KINDS + JEPA_KINDS)):
        return (corpus_mask(data_dir) if kinds - {"raw"} else None), {}

    path = Path(str(args.basis).replace("{fold}", fold))
    if not path.exists():
        raise SystemExit(
            f"[!] no basis at {path}. Run: python scripts/fit_corpus_basis.py "
            f"--out {path}")
    mask, bases, meta = load_basis(path)
    scope = "labeled voxels IN" if meta.get("include_labeled") else "unlabeled only"
    excluded = meta.get("excluded_datasets") or []
    print(f"    basis {path.name}: {meta['n_subjects']} subjects, "
          f"{meta['datasets']} datasets, {scope}"
          + (f", excluding {excluded}" if excluded else ""))
    if meta.get("include_labeled") and fold not in excluded:
        print(f"    [!] TRANSDUCTIVE: the basis saw {fold}'s own voxels. "
              f"Not a leave-one-dataset-out number.")
    return mask, bases


def cap(ds, max_windows):
    """Evenly-spaced subsample, so every subject keeps a share of its windows
    rather than some subjects vanishing entirely."""
    if not max_windows or len(ds) <= max_windows:
        return ds
    idx = np.linspace(0, len(ds) - 1, max_windows).astype(int)
    return Subset(ds, np.unique(idx).tolist())


def dataset_pairs(datasets):
    """Ordered ``(source, target)`` pairs: train on one dataset, test on another.

    This is the protocol the *published* single-dataset DeepMReye checkpoints
    were trained under (`dataset<N>_*.h5` on OSF), so it is the only way to
    compare against them like for like. The leave-one-dataset-out table trains
    on five and cannot be set against a model trained on one.
    """
    out = []
    for source in sorted(datasets):
        for target in sorted(datasets):
            if source != target:
                out.append((f"{source} -> {target}", (source, target)))
    return out


def make_splits(protocol, holdout, data_dir, args, allowed=None):
    common = dict(labeled_data_dir=data_dir, window_size=args.window_size)
    if allowed is not None:
        common["datasets"] = set(allowed)
    if protocol == "within":
        kw = dict(split_by="time", gap=args.gap)
    elif protocol == "subject":
        kw = dict(split_by="subject")
    elif protocol == "pair":
        # Restricting the corpus to the two datasets makes the existing
        # leave-one-out branch yield exactly source for train, target for test.
        source, target = holdout
        kw = dict(datasets={source, target}, holdout={target})
        common.pop("datasets", None)
    else:
        kw = dict(holdout=holdout)
    return (ProbeDataset(split="train", **kw, **common),
            ProbeDataset(split="test", **kw, **common))


def run_fold(fold, holdout, data_dir, args, mask, bases, allowed=None):
    if "{fold}" in str(args.basis):
        mask, bases = load_bases_for(fold, args, data_dir)
    train_ds, test_ds = make_splits(args.protocol, holdout, data_dir, args,
                                    allowed=allowed)
    if not len(train_ds) or not len(test_ds):
        print(f"  [!] {fold}: empty split (train {len(train_ds)}, test {len(test_ds)}) -- skipped")
        return {}

    train_ds, test_ds = cap(train_ds, args.max_windows), cap(test_ds, args.max_windows)
    # A separate cap on the *training* side only, so the labeled-data budget can
    # be swept while the test split -- and therefore the metric -- stays fixed.
    # This is the regime where an unsupervised basis is supposed to earn its
    # keep: a fold-local PCA has to be estimated from whatever labels you have,
    # while a corpus basis was already paid for.
    train_ds = cap(train_ds, args.max_train_windows)
    print(f"    train {len(train_ds)} windows, test {len(test_ds)} windows")

    extractors = build_extractors(args, mask, bases)
    # Fold-local bases are fitted on the training split only, before either
    # split is read for features: a basis that had seen the test windows would
    # be a different (and unearned) arm.
    fit_fold_bases(extractors, train_ds, args)

    n_t = args.window_size // args.temp_patch_size
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers),
        "test": DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers),
    }
    tr = extract(loaders["train"], n_t, extractors, f"{fold} train")
    te = extract(loaders["test"], n_t, extractors, f"{fold} test")
    if tr is None or te is None:
        return {}

    results = {}
    for feature in extractors:
        x_tr0, y_tr, ds_tr, sub_tr = flatten_valid_groups(
            tr[0][feature], tr[1], tr[2], tr[3])
        x_te0, y_te, ds_te, sub_te = flatten_valid_groups(
            te[0][feature], te[1], te[2], te[3])
        if len(x_tr0) < 3 or len(x_te0) < 2:
            continue

        # R^2 is measured against the *training* mean gaze. Against the test
        # mean it would flatter a model that has only learned where this
        # dataset's gaze sits on average.
        # Put every training dataset's gaze on one scale before pooling them.
        #
        # `--protocol dataset` fits ONE readout over the pooled training folds,
        # so the squared-error loss is dominated by whichever dataset has the
        # largest target variance. That was harmless while every labeled set was
        # in degrees of visual angle, and stopped being harmless the moment the
        # corpus gained datasets recorded in screen pixels: the per-fold
        # Euclidean scale now spans 21 (dsL01) to 595 (dsL12), a factor of 28,
        # and the fit simply follows the big-numbered ones.
        #
        # Standardising per dataset is the minimal correction. It uses training
        # data only, and Pearson r -- the headline metric -- is invariant to it,
        # so the predictions land in z-units and every reported correlation is
        # unchanged in meaning. R^2 against the training mean stays comparable
        # because `baseline` is recomputed on the same scale.
        if args.standardize_targets == "dataset":
            y_tr = y_tr.astype(np.float64).copy()
            for d in np.unique(ds_tr):
                m = ds_tr == d
                sd = y_tr[m].std(axis=0)
                sd[sd < 1e-9] = 1.0
                y_tr[m] = (y_tr[m] - y_tr[m].mean(axis=0)) / sd

        baseline = y_tr.mean(axis=0)

        # Alignment is a transform of already-extracted features, so every
        # method shares one pass over the data rather than re-reading 50 GB.
        g_tr = sub_tr if args.align_by == "subject" else ds_tr
        g_te = sub_te if args.align_by == "subject" else ds_te

        for method in args.align:
            x_tr, x_te = apply_pair(x_tr0, g_tr, x_te0, g_te, method)
            arm = feature if method == "none" else f"{feature}|{method}"

            # Where each concatenated block starts, for the readouts that
            # regularise per block. `None` for a single feature source, which
            # those readouts read as one block spanning everything.
            blocks = getattr(extractors[feature], "block_widths", None)
            if args.dyadic_blocks:
                from deepmreye.evaluate.combine import dyadic_blocks

                blocks = dyadic_blocks(blocks or [x_tr.shape[1]])

            for readout in args.readouts:
                t0 = time.time()
                model = fit_readout(readout, x_tr, y_tr, args.n_components,
                                    args.seed, blocks=blocks, groups=sub_tr)
                if model is None:
                    continue
                preds = predict(model, x_te)
                results[f"{arm}/{readout}"] = {
                    "feature": arm,
                    "align": method,
                    "readout": readout,
                    "by_subject": aggregate_by_subject(y_te, preds, sub_te, baseline),
                    "pooled": compute_probe_metrics(y_te, preds, baseline),
                    "by_dataset": {
                        str(d): aggregate_by_subject(
                            y_te[ds_te == d], preds[ds_te == d],
                            sub_te[ds_te == d], baseline)
                        for d in np.unique(ds_te)
                    },
                    "seconds": round(time.time() - t0, 1),
                    "n_features": int(x_tr.shape[1]),
                    "n_train_rows": int(len(x_tr)),
                }
                # What a block readout actually decided. This is the answer to
                # "does the second block contribute", stated by the fit itself
                # rather than inferred from a score difference: a hard-shrunk
                # block (large `block_alphas`, ~0 stack weight) says redundant.
                if blocks and len(blocks) > 1:
                    fitted = results[f"{arm}/{readout}"]["fitted"] = {}
                    fitted["blocks"] = [int(b) for b in blocks]
                    for attr in ("block_alphas_", "block_weights_", "alpha_",
                                 "stack_weights_"):
                        if hasattr(model, attr):
                            fitted[attr.rstrip("_")] = np.asarray(
                                getattr(model, attr)).round(4).tolist()
    return results


def report(all_results, pooled=False, standardized=False):
    def row(res):
        m = res["pooled"] if pooled else res["by_subject"]
        return (m.get("n_subjects", 0), m.get("euclidean_error", float("nan")),
                m.get("r2_vs_baseline", float("nan")),
                m.get("pearson_r_x", float("nan")), m.get("pearson_r_y", float("nan")))

    width = 112
    print("\n" + "=" * width)
    print(f"{'fold':<22} {'feature':<24} {'readout':<10} {'dim':>5} {'subj':>5} "
          f"{'euclid':>8} {'R2':>8} {'r_x':>7} {'r_y':>7}")
    print("-" * width)
    for fold, arms in all_results.items():
        for res in arms.values():
            n_sub, euc, r2, rx, ry = row(res)
            print(f"{fold:<22} {res['feature']:<24} {res['readout']:<10} "
                  f"{res['n_features']:>5} {n_sub:>5} {euc:>8.3f} {r2:>8.3f} "
                  f"{rx:>7.3f} {ry:>7.3f}")
    print("=" * width)

    # The per-fold table is the evidence; this is the summary a feature source
    # is actually judged on, since a basis that wins one fold and loses three
    # has not helped. Median over folds of the mean of r_x and r_y.
    by_arm = {}
    for arms in all_results.values():
        for res in arms.values():
            key = (res["feature"], res["readout"])
            _, _, r2, rx, ry = row(res)
            by_arm.setdefault(key, []).append((0.5 * (rx + ry), r2))
    if len(by_arm) > 1:
        print(f"\n{'feature':<24} {'readout':<10} {'folds':>6} {'median r':>10} {'median R2':>10}")
        print("-" * 64)
        for (feature, readout), vals in sorted(
                by_arm.items(), key=lambda kv: -np.nanmedian([v[0] for v in kv[1]])):
            rs = [v[0] for v in vals]
            r2s = [v[1] for v in vals]
            print(f"{feature:<24} {readout:<10} {len(vals):>6} "
                  f"{np.nanmedian(rs):>10.3f} {np.nanmedian(r2s):>10.3f}")
    print("\nPer-subject medians. R2 is against the training-mean gaze; 0 = learned nothing."
          if not pooled else "\nPooled over all rows (mixes between-subject variance in).")
    if standardized:
        print("Targets were standardised per training dataset (--standardize-targets "
              "dataset),\nso R2 and euclid are NOT interpretable here: predictions are "
              "in z-units while\nthe test targets keep their own scale. Read the r "
              "columns only.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--protocol",
                   choices=["within", "subject", "dataset", "paradigm", "pair"],
                   default="dataset")
    p.add_argument("--readouts", nargs="+", default=list(DEFAULT_READOUTS),
                   choices=list(ALL_READOUTS))
    p.add_argument("--features", nargs="+", default=["raw"],
                   help=f"Feature sources to cross with the readouts, from "
                        f"{{{', '.join(FEATURE_KINDS)}}}. `raw` is the published "
                        f"baseline; the corpus ones need --basis. Join with '+' to "
                        f"concatenate, e.g. fold-pca+lr-cca.")
    p.add_argument("--basis", default="results/corpus_basis.npz",
                   help="Bases fitted by scripts/fit_corpus_basis.py. May contain "
                        "'{fold}', which is replaced by the held-out fold name so "
                        "each fold loads its own basis.")
    p.add_argument("--n-basis", type=int, default=None,
                   help="Keep only the top N components of each basis (default: all).")
    p.add_argument("--jepa-checkpoint", default="results/orbitjepa.pt",
                   help="Cross-orbit JEPA for `jepa`/`jepa-random`.")
    p.add_argument("--device", default="auto",
                   help="Device for torch-backed feature sources.")
    p.add_argument("--align", nargs="+", default=["none"], choices=list(ALIGN_METHODS),
                   help="Unsupervised feature alignment applied per group before "
                        "the readout; see deepmreye/evaluate/align.py. Several may "
                        "be given and share one extraction pass.")
    p.add_argument("--align-by", choices=["subject", "dataset"], default="subject",
                   help="The unit each alignment is fitted within.")
    p.add_argument("--basis-fit-windows", type=int, default=400,
                   help="Training windows used to fit a fold-local basis.")
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--temp-patch-size", type=int, default=5)
    p.add_argument("--voxel-stride", type=int, default=4)
    p.add_argument("--n-components", type=int, default=32, help="For pca-ridge and pls.")
    p.add_argument("--gap", type=int, default=0, help="TRs discarded either side of a `within` split.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-folds", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None,
                   help="Subsample each split to at most N windows, evenly spaced. "
                        "For iterating locally; leave off for a real number.")
    p.add_argument("--max-train-windows", type=int, default=None,
                   help="Cap the training split only, leaving the test split (and "
                        "so the metric) untouched. Sweeps the labeled-data budget.")
    p.add_argument("--out", default=None, help="Write full results as JSON here.")
    p.add_argument("--pooled", action="store_true", help="Print pooled instead of per-subject.")
    p.add_argument("--standardize-targets", choices=("none", "dataset"),
                   default="dataset",
                   help="Put each training dataset's gaze on one scale before "
                        "pooling. Required once the corpus mixes units: the "
                        "per-fold Euclidean scale spans 21 to 595, and a single "
                        "pooled ridge otherwise fits whichever dataset has the "
                        "biggest numbers. Pearson r is unaffected; R^2 and "
                        "Euclidean error are NOT interpretable in this mode, "
                        "because predictions come out in z-units while the test "
                        "targets stay in their own. Use 'none' to reproduce the "
                        "pre-expansion numbers.")
    p.add_argument("--shrink-lambda", type=float, default=0.5,
                   help="`fold-shrunk-pca` only: weight on the corpus covariance in "
                        "(1-lam)*C_fold + lam*C_corpus. 0 reproduces fold-pca "
                        "and 1 reproduces corpus-pca, so sweep it -- the claim "
                        "worth making is an interior optimum.")
    p.add_argument("--dyadic-blocks", action="store_true",
                   help="For the block readouts only: subdivide each feature "
                        "block into log-spaced sub-blocks (8, 8, 16, 32, ...) so "
                        "cross-validation learns a penalty *taper* down the "
                        "variance-ordered spectrum instead of the hard `:k` "
                        "truncation. Asks whether truncation is the wrong prior "
                        "rather than the wrong budget.")
    p.add_argument("--exclude-datasets", nargs="*", default=(),
                   help="Labeled datasets to keep out of the fold set. A dataset "
                        "that is mid-ingest or has not passed "
                        "scripts/verify_gaze_sync.py would otherwise join "
                        "automatically and move every number.")
    p.add_argument("--fold-name", default=None,
                   help="Filter execution to folds whose name matches this substring.")
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    print(f"[*] data {data_dir}\n[*] protocol {args.protocol}"
          f"\n[*] features {args.features}\n[*] readouts {args.readouts}")

    for spec in args.features:          # fail on a typo before reading 50 GB
        parse_spec(spec)

    # Only load the corpus bases if an arm asks for them, so the plain baseline
    # still runs on a machine that has never fitted one. With a `{fold}`
    # placeholder each fold loads its own instead, inside run_fold.
    mask, bases = None, {}
    if "{fold}" not in str(args.basis):
        mask, bases = load_bases_for("", args, data_dir)

    present = sorted({s.dataset for s in ProbeDataset(
        labeled_data_dir=data_dir, split="train", window_size=args.window_size)._discover()})
    drop = set(args.exclude_datasets or ())
    hit = sorted(drop & set(present))
    if hit:
        print(f"[*] excluding {len(hit)} dataset(s): {', '.join(hit)}")
        present = [d for d in present if d not in drop]
    print(f"[*] labeled datasets: {', '.join(present)}")

    if args.protocol == "within":
        folds = [("within-subject", None)]
    elif args.protocol == "subject":
        folds = [("held-out subjects", None)]
    elif args.protocol == "dataset":
        folds = dataset_folds(present)
    elif args.protocol == "pair":
        folds = dataset_pairs(present)
    else:
        folds = paradigm_folds(present)

    if args.fold_name:
        folds = [f for f in folds if args.fold_name in f[0]]

    if args.limit_folds:
        folds = folds[: args.limit_folds]

    all_results = {}
    for name, holdout in folds:
        print(f"\n[*] fold: {name}" + (f"  (holding out {sorted(holdout)})" if holdout else ""))
        all_results[name] = run_fold(name, holdout, data_dir, args, mask, bases,
                                     allowed=present)

    report(all_results, args.pooled,
           standardized=args.standardize_targets == "dataset")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(all_results, indent=2, default=float))
        print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
