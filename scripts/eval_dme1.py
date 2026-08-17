#!/usr/bin/env python3
"""The published DeepMReye 1.0 CNN, evaluated on this corpus.

Every result on this branch is framed against the published model, and until now
that number did not exist here -- the baseline table holds sklearn readouts and a
random-feature control, nothing else. This script supplies it, and it does so
*without retraining*: the authors released model weights on OSF
(https://osf.io/mrhk9/, ``model_weights/``), so the comparison uses their
checkpoint rather than a reimplementation a reviewer would have to take on faith.

    # leave-one-dataset-out: trained on datasets 1-5, evaluated on the 6th
    python scripts/eval_dme1.py --weights results/dme1/datasets_1to5.h5 \
        --datasets dsL06_sequences

**Which checkpoints are legitimate for which fold.** The labeled datasets here
*are* the DeepMReye paper's training data, so ``datasets_1to6.h5`` has seen every
participant we would score it on and must never be reported as held out. The
usable checkpoints are:

- ``datasets_1to5.h5``  -> held out on ``dsL06`` only.
- ``dataset<N>_*.h5``   -> trained on one dataset, so held out on the other five.

Anything else is contaminated. ``--allow-contaminated`` exists to reproduce the
in-sample number deliberately; it prints a warning and tags the output.

**Environment.** The weights are Keras 2.4 HDF5 and the architecture is
TensorFlow, neither of which belongs in the project venv -- TF's numpy pin
fights the sklearn/torch stack. Use the separate one:

    uv venv .venv-tf --python 3.11
    uv pip install --python .venv-tf/bin/python tensorflow tf-keras pandas scipy scikit-learn
    .venv-tf/bin/python scripts/eval_dme1.py ...

The v1 source is vendored **at run time** from the ``main`` branch with
``git show``, rather than copied into this branch. That keeps one published
implementation rather than a fork of it, and means this script cannot silently
drift from what was actually released.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# tf-keras: the checkpoints predate Keras 3, whose HDF5 loader will not read
# them. Must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = Path(__file__).resolve().parent.parent
V1_FILES = ["deepmreye/architecture.py", "deepmreye/util/util.py",
            "deepmreye/util/model_opts.py"]
# Checkpoints that have seen every labeled participant.
CONTAMINATED = {"datasets_1to6.h5"}


def vendor_v1(ref="main"):
    """Materialise the published v1 modules from ``ref`` into a temp package.

    Written out rather than imported from this branch: ``deepmreye`` here is v2
    and shares module names with v1, so the vendored copy must come first on
    ``sys.path`` and must not be a partial mix of the two.
    """
    root = Path(tempfile.mkdtemp(prefix="dme1_"))
    (root / "deepmreye" / "util").mkdir(parents=True)
    # Empty __init__: v1's real one pulls in the whole package, and we need
    # exactly two modules from it.
    (root / "deepmreye" / "__init__.py").write_text("")
    (root / "deepmreye" / "util" / "__init__.py").write_text("")
    for rel in V1_FILES:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{rel}"],
                             capture_output=True, check=True)
        (root / rel).write_bytes(out.stdout)
    sys.path.insert(0, str(root))
    return root


def build_inference_model(input_shape, inner_timesteps=10):
    """The v1 inference graph with the published weights' topology."""
    from deepmreye import architecture
    from deepmreye.util import model_opts

    # v1 asks for `keras.optimizers.legacy.Adam` on Apple Silicon, which modern
    # Keras removed. Nothing here trains, so any optimizer will do -- but
    # `create_standard_model` compiles, so one has to exist.
    from tensorflow.keras.optimizers import Adam

    architecture.get_adam_optimizer = lambda lr: Adam(learning_rate=lr)

    opts = model_opts.get_opts()
    opts["mc_dropout"] = False
    opts["gaussian_noise"] = 0
    opts["inner_timesteps"] = inner_timesteps

    try:
        _, model_inference = architecture.create_standard_model(input_shape, opts)
    except AttributeError as e:
        # Newer Keras makes `model.metrics` read-only; v1 appends to it.
        raise SystemExit(
            f"[!] v1 architecture is incompatible with this Keras: {e}\n"
            f"    Try an older TF (uv pip install 'tensorflow==2.15.*') in .venv-tf.")
    return model_inference, opts


def find_corpus(explicit=None):
    """The corpus directory, mirroring ``datasource.resolve``'s precedence.

    Inlined rather than imported: importing v2's ``deepmreye`` here would pull in
    ``ants`` (absent from this venv) and, worse, would register v2 in
    ``sys.modules`` under the very name the vendored v1 needs. Never import the
    project package from this script.
    """
    if explicit:
        return Path(explicit).expanduser()
    for candidate in [Path(os.environ["DEEPMREYE_DATA"]).expanduser()
                      if os.environ.get("DEEPMREYE_DATA") else None,
                      REPO / "data",
                      Path.home() / ".cache" / "deepmreye"]:
        if candidate and any(candidate.glob("dsL*/*.h5")):
            return candidate
    raise SystemExit("[!] no corpus found; pass --data-dir")


def participant_files(data_dir, datasets):
    for ds in datasets:
        for p in sorted((Path(data_dir) / ds).glob("*.h5")):
            yield ds, p


def _reduce(a, bin_trs):
    """``[T, 10, 2]`` -> ``[T/bin, 2]`` by nanmean over sub-TR samples and TRs.

    This has to match ``evaluate.probe.temporal_targets`` exactly. It bins 5 TRs
    (``window_size / temp_patch_size`` = 100/20) and takes the nanmean over the
    TRs *and* their sub-TR samples together. Scoring this model at TR resolution
    against a probe number computed on 5-TR means would not be the same
    measurement -- averaging suppresses noise and lifts correlations, and it
    would do so only on our side of the comparison.
    """
    import warnings as _w

    a = np.asarray(a, dtype=np.float64)
    bin_trs = max(1, int(bin_trs))
    pad = (-a.shape[0]) % bin_trs
    if pad:
        a = np.concatenate([a, np.full((pad,) + a.shape[1:], np.nan)], axis=0)
    # [T, 10, 2] -> [T/bin, bin, 10, 2]; reduce everything but the bin and x/y.
    a = a.reshape(-1, bin_trs, *a.shape[1:])
    axes = tuple(range(1, a.ndim - 1))
    with _w.catch_warnings():
        # An all-NaN bin means no gaze was recorded there; masked downstream.
        _w.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axes)


def score_subject(pred, true, bin_trs=5):
    """Per-subject Pearson r per gaze axis, on the probe's temporal bins.

    NaN labels mark TRs with no valid gaze sample and are dropped rather than
    imputed.
    """
    p = _reduce(pred, bin_trs)
    t = _reduce(true, bin_trs)
    ok = np.isfinite(t).all(1) & np.isfinite(p).all(1)
    p, t = p[ok], t[ok]
    if len(p) < 20:
        return None
    r = [float(np.corrcoef(p[:, j], t[:, j])[0, 1]) if t[:, j].std() > 1e-9
         else np.nan for j in range(2)]
    err = float(np.nanmean(np.linalg.norm(p - t, axis=1)))
    return {"n": int(len(p)), "pearson_r_x": r[0], "pearson_r_y": r[1],
            "euclidean_error": err}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", default="results/dme1/datasets_1to5.h5")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--datasets", nargs="+", default=["dsL06_sequences"])
    p.add_argument("--out", default="results/probe_dme1.json")
    p.add_argument("--limit", type=int, default=0,
                   help="participants per dataset (0 = all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temporal-bin", type=int, default=5,
                   help="TRs per temporal bin. 5 matches eval_probe's default "
                        "window-size 100 / temp-patch-size 5; use 1 to score at "
                        "full TR resolution.")
    p.add_argument("--ref", default="main", help="git ref holding v1")
    p.add_argument("--allow-contaminated", action="store_true")
    args = p.parse_args()

    name = Path(args.weights).name
    if name in CONTAMINATED and not args.allow_contaminated:
        raise SystemExit(
            f"[!] {name} was trained on every labeled dataset, so it has seen "
            f"the participants you are about to score. Use datasets_1to5.h5 "
            f"(held out on dsL06) or a single-dataset checkpoint. Pass "
            f"--allow-contaminated to do it anyway.")

    import h5py

    data_dir = find_corpus(args.data_dir)
    files = list(participant_files(data_dir, args.datasets))
    if args.limit:
        keep, seen = [], {}
        for ds, f in files:
            seen[ds] = seen.get(ds, 0) + 1
            if seen[ds] <= args.limit:
                keep.append((ds, f))
        files = keep
    if not files:
        raise SystemExit(f"[!] no participants under {data_dir} for {args.datasets}")

    with h5py.File(files[0][1], "r") as f:
        spatial = tuple(f["eye_block"].shape[:3])
        inner = int(f["labels"].shape[1])
    print(f"[*] data {data_dir}\n[*] {len(files)} participants, "
          f"input {spatial + (1,)}, {inner} sub-TR samples")

    root = vendor_v1(args.ref)
    print(f"[*] vendored DeepMReye v1 from '{args.ref}' -> {root}")
    model, _ = build_inference_model(spatial + (1,), inner)
    model.load_weights(args.weights)
    print(f"[*] loaded {args.weights}")

    per_subject, rows = {}, []
    for ds, path in files:
        with h5py.File(path, "r") as f:
            block = f["eye_block"][...]
            labels = f["labels"][...]
        x = np.moveaxis(block, -1, 0)[..., None].astype(np.float32)
        n = min(len(x), len(labels))
        pred = model.predict(x[:n], batch_size=args.batch_size, verbose=0)[0]
        got = score_subject(np.asarray(pred), labels[:n], args.temporal_bin)
        if got is None:
            continue
        got["dataset"] = ds
        per_subject[path.stem] = got
        rows.append(got)
        print(f"  {ds:<24}{path.stem:<22} r_x {got['pearson_r_x']:+.3f}  "
              f"r_y {got['pearson_r_y']:+.3f}  err {got['euclidean_error']:.2f}",
              flush=True)

    if not rows:
        raise SystemExit("[!] no participant produced a score")

    def agg(key, fn):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(fn(v[np.isfinite(v)]))

    summary = {
        "weights": name,
        "contaminated": name in CONTAMINATED,
        "temporal_bin_trs": args.temporal_bin,
        "datasets": args.datasets,
        "n_subjects": len(rows),
        "pearson_r_x": agg("pearson_r_x", np.median),
        "pearson_r_y": agg("pearson_r_y", np.median),
        "mean_r": 0.5 * (agg("pearson_r_x", np.median)
                         + agg("pearson_r_y", np.median)),
        "euclidean_error": agg("euclidean_error", np.median),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "per_subject": per_subject}, indent=1))
    print(f"\n[*] median over {summary['n_subjects']} participants: "
          f"r_x {summary['pearson_r_x']:+.3f}, r_y {summary['pearson_r_y']:+.3f}, "
          f"mean r {summary['mean_r']:+.3f}")
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
