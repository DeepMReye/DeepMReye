"""The published DeepMReye 1.0 CNN, scored on this corpus under this protocol.

The comparison uses the authors' released weights (https://osf.io/mrhk9/,
``model_weights/``) rather than a reimplementation, and the v1 source is
vendored **at run time** from the ``main`` branch with ``git show`` rather than
copied here, so this cannot silently drift from what was published.

**Two stages, two virtualenvs, and that is the point.** The weights are Keras
2.4 HDF5 and need TensorFlow, whose numpy pin fights the sklearn stack; worse,
v1's package is also called ``deepmreye``, so importing this branch's package in
the same process as the vendored one is a module-name collision waiting to
happen. So:

    # 1. predictions only -- no v2 import anywhere in this process
    .venv-tf/bin/python scripts/eval_dme1.py predict \
        --weights results/dme1/datasets_1to6.h5 \
        --datasets dsL08_studyforrest_movie dsL11_backtothefuture \
        --out results/dme1/pred_1to6.npz

    # 2. scoring, through `deepmreye.probe` and nothing else
    .venv/bin/python scripts/eval_dme1.py score --pred results/dme1/pred_1to6.npz

Stage 2 pushes v1's ``[T, 10, 2]`` output through the *same* ``_resolutions`` ->
``metrics.score`` -> ``_summarise`` path every arm in ``probe`` goes through, so
the sub-TR and 1-TR numbers are comparable to the rest of the project by
construction rather than by a matching convention someone has to maintain. v1
predicts gaze directly, so there is no ridge and no training fold -- the only
thing it shares with `lodo` is the metric.

**Which checkpoint is legitimate on which fold.** ``dsL01``--``dsL06`` *are* the
DeepMReye paper's training datasets 1-6, so:

- ``datasets_1to6.h5`` -- the model as actually shipped and used. Held out on
  ``dsL07`` (ds006833, Kling et al., a later acquisition), ``dsL08``
  (studyforrest) and ``dsL11`` (Back to the Future). Contaminated on the rest.
- ``datasets_1to5.h5`` -- additionally held out on ``dsL06``.

``--allow-contaminated`` reproduces an in-sample number deliberately; it warns
and tags the output. Nothing here may be reported as held out without appearing
in ``CLEAN_FOLDS``.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
V1_FILES = ["deepmreye/architecture.py", "deepmreye/util/util.py",
            "deepmreye/util/model_opts.py"]

# What each checkpoint has NOT seen. dsL01-dsL06 are the paper's datasets 1-6.
CLEAN_FOLDS = {
    "datasets_1to6.h5": {"dsL07_deepmreye_calib", "dsL08_studyforrest_movie",
                         "dsL11_backtothefuture"},
    "datasets_1to5.h5": {"dsL06_sequences", "dsL07_deepmreye_calib",
                         "dsL08_studyforrest_movie", "dsL11_backtothefuture"},
}


# --------------------------------------------------------------------------- #
# Stage 1: predictions (TensorFlow venv; must not import this branch's package)
# --------------------------------------------------------------------------- #

def vendor_v1(ref="main"):
    """Materialise the published v1 modules from ``ref`` into a temp package.

    Written out rather than imported from this branch: ``deepmreye`` here is v2
    and shares module names with v1, so the vendored copy must come first on
    ``sys.path`` and must not end up a partial mix of the two.
    """
    root = Path(tempfile.mkdtemp(prefix="dme1_"))
    (root / "deepmreye" / "util").mkdir(parents=True)
    # Empty __init__: v1's real one imports the whole package and we need two
    # modules out of it.
    (root / "deepmreye" / "__init__.py").write_text("")
    (root / "deepmreye" / "util" / "__init__.py").write_text("")
    for rel in V1_FILES:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{rel}"],
                             capture_output=True, check=True)
        (root / rel).write_bytes(out.stdout)
    sys.path.insert(0, str(root))
    return root


def build_inference_model(input_shape, inner_timesteps=10):
    """v1's inference graph, at the topology the released weights expect."""
    from deepmreye import architecture
    from deepmreye.util import model_opts
    # v1 asks for `keras.optimizers.legacy.Adam`, which modern Keras removed.
    # Nothing here trains, but `create_standard_model` compiles, so one must exist.
    from tensorflow.keras.optimizers import Adam

    architecture.get_adam_optimizer = lambda lr: Adam(learning_rate=lr)
    opts = model_opts.get_opts()
    opts["mc_dropout"] = False
    opts["gaussian_noise"] = 0
    opts["inner_timesteps"] = inner_timesteps
    _, model_inference = architecture.create_standard_model(input_shape, opts)
    return model_inference


def find_corpus(explicit=None):
    """The corpus directory, mirroring ``datasource.resolve``'s precedence.

    Inlined rather than imported, because importing this branch's package would
    register v2 in ``sys.modules`` under the name the vendored v1 needs.
    """
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("DEEPMREYE_DATA")
    for c in [Path(env).expanduser() if env else None, REPO / "data",
              Path.home() / ".cache" / "deepmreye"]:
        if c and any(c.glob("dsL*/*.h5")):
            return c
    raise SystemExit("[!] no corpus found; pass --data-dir")


def cmd_predict(args):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import h5py

    name = Path(args.weights).name
    clean = CLEAN_FOLDS.get(name, set())
    dirty = [d for d in args.datasets if d not in clean]
    if dirty and not args.allow_contaminated:
        raise SystemExit(
            f"[!] {name} was trained on {', '.join(dirty)} -- scoring it there is "
            f"in-sample. Held out for this checkpoint: "
            f"{', '.join(sorted(clean)) or '(none recorded)'}. "
            f"Pass --allow-contaminated to do it anyway.")

    data_dir = find_corpus(args.data_dir)
    files = [(ds, p) for ds in args.datasets
             for p in sorted((data_dir / ds).glob("*.h5"))[: args.limit or None]]
    if not files:
        raise SystemExit(f"[!] no participants under {data_dir} for {args.datasets}")

    with h5py.File(files[0][1], "r") as f:
        spatial = tuple(f["eye_block"].shape[:3])
        inner = int(f["labels"].shape[1])
    print(f"[*] {data_dir}: {len(files)} participants, input {spatial + (1,)}, "
          f"{inner} sub-TR samples", flush=True)

    root = vendor_v1(args.ref)
    print(f"[*] vendored DeepMReye v1 from '{args.ref}' -> {root}", flush=True)
    model = build_inference_model(spatial + (1,), inner)
    model.load_weights(args.weights)
    print(f"[*] loaded {args.weights}", flush=True)

    out, meta = {}, []
    for ds, path in files:
        with h5py.File(path, "r") as f:
            if "labels" not in f:
                continue
            block, labels = f["eye_block"][...], f["labels"][...]
        x = np.moveaxis(block, -1, 0)[..., None].astype(np.float32)
        n = min(len(x), len(labels))
        pred = np.asarray(model.predict(x[:n], batch_size=args.batch_size,
                                        verbose=0)[0], dtype=np.float32)
        i = len(meta)
        out[f"p/{i}"] = pred                       # [T, 10, 2]
        out[f"y/{i}"] = labels[:n].astype(np.float32)
        meta.append({"dataset": ds, "subject": path.stem})
        print(f"  {ds:<26}{path.stem:<12} {pred.shape}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, meta=np.array(json.dumps(meta)),
             weights=np.array(name),
             contaminated=np.array(json.dumps(sorted(dirty))), **out)
    print(f"[+] {len(meta)} participants -> {args.out}")


# --------------------------------------------------------------------------- #
# Stage 2: scoring (project venv; `probe` is the only implementation of this)
# --------------------------------------------------------------------------- #

def load_preds(paths):
    """Merge one or more prediction files into ``(records, contaminated)``.

    Several files because a run is usually split by checkpoint and by whether
    the fold is in-sample; the union is what gets scored. A participant may
    appear only once -- the same participant under two checkpoints is two
    different arms, not more data.
    """
    recs, dirty, weights, seen = [], set(), [], set()
    for path in paths:
        d = np.load(path, allow_pickle=False)
        weights.append(str(d["weights"]))
        dirty |= set(json.loads(str(d["contaminated"])))
        for i, m in enumerate(json.loads(str(d["meta"]))):
            key = (m["dataset"], m["subject"])
            if key in seen:
                raise SystemExit(
                    f"[!] {key} appears in more than one prediction file; the "
                    f"same participant under two checkpoints is two arms, not "
                    f"one. Score them separately.")
            seen.add(key)
            recs.append({**m, "pred": d[f"p/{i}"], "labels": d[f"y/{i}"]})
    return recs, dirty, weights


def cmd_score(args):
    sys.path.insert(0, str(REPO))
    from deepmreye import metrics, probe

    meta, dirty, weights = load_preds(args.pred)

    # Same shape `lodo` builds: predict once per participant, then calibrate each
    # against the *other* participants of its own dataset. v1 needs no training
    # fold, so the loop is over datasets only.
    rows = []
    for ds in sorted({m["dataset"] for m in meta}):
        preds = {}
        for m in (m for m in meta if m["dataset"] == ds):
            got = probe._resolutions(m["pred"].reshape(len(m["pred"]), 20),
                                     m["labels"])
            if got is not None:
                preds[m["subject"]] = got
        for sub, per_res in preds.items():
            row = {"dataset": ds, "subject": sub}
            for res, (p, t) in per_res.items():
                others = [preds[o][res] for o in preds if o != sub]
                if others:
                    gain, offset = metrics.fit_affine(
                        np.concatenate([o[0] for o in others]),
                        np.concatenate([o[1] for o in others]))
                else:
                    gain = offset = None
                row[res] = metrics.score(p, t, gain, offset)
            rows.append(row)

    res = probe._summarise(rows, sorted({m["dataset"] for m in meta}))
    probe.report(res, f"DeepMReye 1.0 ({', '.join(sorted(set(weights)))})")
    if dirty:
        print(f"\n[!] IN-SAMPLE for this checkpoint, NOT held out -- these folds "
              f"favour DeepMReye:\n    {', '.join(sorted(dirty))}")

    if args.json:
        payload = {"weights": sorted(set(weights)), "contaminated": sorted(dirty),
                   "summary": res["summary"],
                   "folds": {k: {"n": v["n"], "subtr": v["subtr"]["r"],
                                 "1tr": v["1tr"]["r"], "subtr_x": v["subtr"]["r_x"],
                                 "subtr_y": v["subtr"]["r_y"]}
                             for k, v in res["folds"].items()},
                   "participants": rows}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1, default=float))
        print(f"[+] {args.json}")


# --------------------------------------------------------------------------- #
# Stage 3: the head-to-head, restricted to folds v1 has not seen
# --------------------------------------------------------------------------- #

def cmd_compare(args):
    """v1 against this project's arms, on the folds v1 is entitled to be scored on.

    The project's 9-fold medians cannot be compared to a 3-fold one, so every
    arm is re-reduced over the *same* fold subset. That subset is whatever the
    checkpoint has not seen, which is why it is read from the scored file rather
    than chosen here.
    """
    dme1 = json.load(open(args.score))
    dirty = set(dme1["contaminated"])
    folds = sorted(dme1["folds"]) if args.include_contaminated else \
        sorted(f for f in dme1["folds"] if f not in dirty)
    if not folds:
        raise SystemExit("[!] no folds left; this checkpoint saw all of them")
    scaling = json.load(open(args.scaling))
    fp = json.load(open(args.foldpca))
    fp_full = next(r for r in fp if r["budget"] is None and r["arm"].endswith("lags1"))

    def med(per_fold, res):
        v = [per_fold[f][res] for f in folds if f in per_fold]
        return float(np.median(v)) if v else float("nan")

    arms = [(f"DeepMReye 1.0 ({dme1['weights']})",
             {f: {"subtr": dme1["folds"][f]["subtr"], "1tr": dme1["folds"][f]["1tr"]}
              for f in dme1["folds"]})]
    for label, key in (("lr-cca:32+lags1 @ N=2000", "lr-cca:32+lags1@2000"),
                       ("corpus-pca:64 @ N=2000", "corpus-pca:64@2000"),
                       ("raw+lags1 (stride-4 voxels)", "raw+lags1")):
        arms.append((label, scaling[key]["folds"]))
    arms.append(("fold-pca:64+lags1 (supervised)", fp_full["folds"]))

    scope = ("all folds (* = IN-SAMPLE for DeepMReye, which favours it)"
             if args.include_contaminated else "folds held out from DeepMReye")
    print(f"\n=== {scope} ===")
    print(f"{'arm':<40}{'sub-TR':>9}{'1-TR':>9}")
    for label, per_fold in arms:
        print(f"{label:<40}{med(per_fold, 'subtr'):>9.3f}{med(per_fold, '1tr'):>9.3f}")
    mark = {f: ("*" if f in dirty else "") for f in folds}
    head = "".join(f"{f.split('_')[0] + mark[f]:>9}" for f in folds)
    print(f"\n{'per fold (sub-TR)':<40}{head}")
    for label, per_fold in arms:
        print(f"{label:<40}" + "".join(
            f"{per_fold[f]['subtr']:>9.3f}" if f in per_fold else f"{'--':>9}"
            for f in folds))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"folds": folds, "contaminated": sorted(dirty),
             "weights": dme1["weights"],
             "arms": {label: {"subtr": med(pf, "subtr"), "1tr": med(pf, "1tr"),
                              "per_fold": {f: pf[f] for f in folds if f in pf}}
                      for label, pf in arms}}, indent=1, default=float))
        print(f"[+] {args.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="run the CNN, save raw predictions (.venv-tf)")
    p.add_argument("--weights", default="results/dme1/datasets_1to6.h5")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out", default="results/dme1/pred.npz")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="participants per dataset (0 = all). Only for the "
                        "in-sample sanity check; a reported number uses all.")
    p.add_argument("--ref", default="main", help="git ref holding the v1 source")
    p.add_argument("--allow-contaminated", action="store_true")
    p.set_defaults(fn=cmd_predict)

    s = sub.add_parser("score", help="score saved predictions through probe (.venv)")
    s.add_argument("--pred", nargs="+", default=["results/dme1/pred.npz"])
    s.add_argument("--json", default=None)
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("compare", help="v1 vs this project's arms on v1's clean folds")
    c.add_argument("--score", default="results/dme1/score_1to6.json")
    c.add_argument("--scaling", default="results/unlabeled_value/corpus_scaling.json")
    c.add_argument("--foldpca", default="results/unlabeled_value/foldpca_full.json")
    c.add_argument("--json", default=None)
    c.add_argument("--include-contaminated", action="store_true",
                   help="also show folds DeepMReye trained on. Legitimate only "
                        "as a conservative bound: the advantage is DeepMReye's, "
                        "so it strengthens a loss and proves nothing about a win.")
    c.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
