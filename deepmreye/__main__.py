"""DeepMReye: decode eye gaze from fMRI without an eye tracker.

    deepmreye decode sub-01_bold.nii.gz -o gaze/        gaze for one or more runs
    deepmreye features sub-01_bold.nii.gz -o feats/     the basis coordinates only
    deepmreye preprocess sub-01_bold.nii.gz -o mystudy/sub-01.h5 [--labels gaze.npy]
    deepmreye calibrate --data-dir mystudy/ -o calibrated.npz
    deepmreye train --data-dir mystudy/ -o my_decoder.npz
    deepmreye evaluate                                   leave-one-dataset-out
    deepmreye fit-basis -o basis.npz                     the basis, from unlabelled fMRI
    deepmreye download [--all]                           the published corpus

`decode` uses the pretrained decoder unless `--model` names another.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _stem(path):
    name = Path(path).name
    for suffix in (".nii.gz", ".nii", ".h5", ".mgz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _model(path):
    from deepmreye.model import Decoder

    return Decoder.pretrained() if path in (None, "pretrained") else Decoder.load(path)


def _basis(path):
    from deepmreye.model import pretrained_basis

    return pretrained_basis() if path in (None, "pretrained") else Path(path)


def write_gaze_tsv(path, gaze, tr=None):
    """One row per gaze sample: volume, sample, onset (s), x, y."""
    t_n = len(gaze)
    vol, sample = np.divmod(np.arange(t_n * 10), 10)
    onset = (vol + sample / 10.0) * tr if tr else np.full(t_n * 10, np.nan)
    xy = np.asarray(gaze).reshape(-1, 2)
    with open(path, "w") as f:
        f.write("volume\tsample\tonset\tx\ty\n")
        for v, s, o, (x, y) in zip(vol, sample, onset, xy):
            f.write(f"{v}\t{s}\t{o:.4f}\t{x:.5f}\t{y:.5f}\n")
    return path


def cmd_decode(args):
    import deepmreye

    model = _model(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        stem = _stem(run)
        print(f"[*] {run}", flush=True)
        qa = out / f"{stem}_qa.png" if args.qa else None
        if str(run).endswith(".h5"):
            res = deepmreye.decode(run, model=model, tr=args.tr)
        else:
            block, tr = deepmreye.extract_eye_block(run, qa_png=qa, tr=args.tr)
            if args.save_block:
                from deepmreye.data import write_subject

                write_subject(out / f"{stem}.h5", block,
                              attrs={"repetition_time": tr, "source": str(run)})
            res = deepmreye.decode(block, model=model, tr=tr)
        write_gaze_tsv(out / f"{stem}_gaze.tsv", res["gaze"], res["tr"])
        summary = {"source": str(run), "volumes": len(res["gaze"]), "tr": res["tr"],
                   "units": res["units"], "confidence": res["confidence"],
                   "model": {k: model.meta.get(k) for k in ("version", "trained_on", "alpha")}}
        (out / f"{stem}_decode.json").write_text(json.dumps(summary, indent=1))
        print(f"[+] {out / (stem + '_gaze.tsv')}  ({len(res['gaze'])} volumes, "
              f"confidence {res['confidence']:.2f}, units {res['units']})")


def _read_labels(path, n_volumes):
    """Gaze labels from ``.npy``: ``[T, 10, 2]``, or ``[T, 2]`` (one sample per volume,
    repeated over the ten sub-volume samples)."""
    y = np.load(path).astype(np.float32)
    if y.ndim == 2 and y.shape[1] == 2:
        y = np.repeat(y[:, None, :], 10, axis=1)
    if y.ndim != 3 or y.shape[1:] != (10, 2):
        raise SystemExit(f"[!] labels must be [T, 10, 2] or [T, 2], got {y.shape}")
    if len(y) != n_volumes:
        raise SystemExit(f"[!] {len(y)} label rows for {n_volumes} volumes")
    return y


def cmd_preprocess(args):
    from deepmreye.data import write_subject
    from deepmreye.preprocess import extract_eye_block

    out = Path(args.out)
    qa = out.with_suffix(".png") if args.qa else None
    block, tr = extract_eye_block(args.run, qa_png=qa, tr=args.tr)
    labels = None
    if args.labels:
        labels = _read_labels(args.labels, block.shape[-1])
    elif args.gaze:
        from deepmreye.data import bin_gaze, read_gaze_tsv

        times, x, y = read_gaze_tsv(args.gaze)
        labels = bin_gaze(times - args.gaze_offset, x, y, block.shape[-1], tr)
    write_subject(out, block, labels=labels, attrs={"repetition_time": tr, "source": str(args.run)})
    print(f"[+] {out}  ({block.shape[-1]} volumes, TR {tr:g} s"
          f"{', with labels' if labels is not None else ''})")


def _runs(data_dir, pattern="*", subjects=None, labelled=True):
    """``(dataset, subject, path)`` for the runs under ``data_dir``.

    Runs are ``<dataset>/<subject>.h5``; ``.h5`` files directly in ``data_dir``
    count as one dataset named after the directory. ``subjects`` keeps only the
    runs whose file name matches one of its glob patterns (``sub-01``,
    ``'sub-0[1-4]*'``); ``labelled`` keeps only runs that carry gaze.
    """
    import fnmatch

    import h5py

    from deepmreye.data import iter_subjects

    root = Path(data_dir)
    found = [(root.name, f.stem, f) for f in sorted(root.glob("*.h5"))]
    found += list(iter_subjects(root, pattern))
    for dataset, subject, path in found:
        if subjects and not any(fnmatch.fnmatch(subject, s) or fnmatch.fnmatch(subject, s + "_*")
                                for s in subjects):
            continue
        if labelled:
            with h5py.File(path, "r") as f:
                if "labels" not in f:
                    continue
        yield dataset, subject, path


def _labelled_runs(data_dir, pattern="*", subjects=None):
    from deepmreye.data import read_subject

    for dataset, subject, path in _runs(data_dir, pattern, subjects):
        block, labels, _attrs = read_subject(path)
        yield dataset, subject, block, labels


def cmd_features(args):
    import deepmreye

    model = _model(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        stem = _stem(run)
        if str(run).endswith(".h5"):
            from deepmreye.data import read_subject

            block = read_subject(run, with_labels=False)[0]
        else:
            block, _tr = deepmreye.extract_eye_block(run, tr=args.tr)
        zl, zr = model.orbits(block)
        z = np.stack([zl, zr], axis=1) if args.orbits else 0.5 * (zl + zr)
        np.save(out / f"{stem}_basis.npy", z.astype(np.float32))
        print(f"[+] {out / (stem + '_basis.npy')}  {z.shape}")


def cmd_train(args):
    from deepmreye.data import LABELLED, resolve
    from deepmreye.model import Decoder

    sources = []
    if args.data_dir:
        sources.append((Path(args.data_dir), args.pattern, args.subjects))
    if args.include_published or not args.data_dir:
        sources.append((resolve(None, patterns=LABELLED), "dsL*", None))
    model = Decoder.from_basis(_basis(args.basis), k=args.k, lags=args.lags)
    feats, labels, datasets = [], [], []
    for root, pattern, subjects in sources:
        for dataset, _subject, block, y in _labelled_runs(root, pattern, subjects):
            feats.append(model.features(block))
            labels.append(y)
            datasets.append(dataset)
    if not feats:
        raise SystemExit("[!] no labelled runs found")
    print(f"[*] {len(feats)} runs from {len(set(datasets))} datasets: "
          f"{', '.join(sorted(set(datasets)))}", flush=True)
    model.fit_features(feats, labels, datasets)
    model.save(args.out)
    print(f"[+] {args.out}  (ridge penalty {model.meta['alpha']:g})")


def cmd_calibrate(args):
    model = _model(args.model)
    blocks, labels = [], []
    for _dataset, _subject, block, y in _labelled_runs(args.data_dir, args.pattern, args.subjects):
        blocks.append(block)
        labels.append(y)
    if not blocks:
        raise SystemExit(f"[!] no labelled runs under {args.data_dir}")
    model.calibrate(blocks, labels, units=args.units)
    model.save(args.out)
    print(f"[+] {args.out}  (gain {np.round(model.gain, 3)}, offset {np.round(model.offset, 3)}, "
          f"from {len(blocks)} runs)")


def cmd_evaluate(args):
    from deepmreye import evaluate
    from deepmreye.data import LABELLED, resolve

    data_dir = resolve(args.data_dir, patterns=LABELLED)
    recs = evaluate.load_or_build(data_dir, _basis(args.basis), cache=args.cache,
                                  rebuild=args.rebuild, pattern=args.pattern)
    print(f"[*] {len(recs)} participants, {len({r['dataset'] for r in recs})} datasets")
    if args.check and not evaluate.calibrate(recs):
        raise SystemExit("[!] the protocol did not reproduce the published numbers")
    res = evaluate.lodo(recs, evaluate.features(args.k, args.lags))
    evaluate.report(res, f"lr-cca:{args.k}, +-{args.lags} TR")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))
        print(f"[+] {args.json}")


def cmd_fit_basis(args):
    from deepmreye.basis import fit_corpus
    from deepmreye.data import CORPUS, resolve

    data_dir = resolve(args.data_dir, patterns=CORPUS)
    fit_corpus(data_dir, args.out, k=args.k, n=args.n, seed=args.seed)


def cmd_download(args):
    from deepmreye.data import CORPUS, LABELLED, cache_dir, fetch

    target = Path(args.data_dir) if args.data_dir else cache_dir()
    fetch(CORPUS if args.all else LABELLED, target=target)
    print(f"[+] {target}")


def build_parser():
    parser = argparse.ArgumentParser(prog="deepmreye", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    data_help = ("directory of <dataset>/<subject>.h5 files; default $DEEPMREYE_DATA, "
                 "./data, else the published corpus (downloaded on first use)")

    p = sub.add_parser("decode", help="Decode gaze for functional runs (NIfTI or .h5).")
    p.add_argument("runs", nargs="+", help="4D NIfTI runs, or eye blocks written by preprocess")
    p.add_argument("-o", "--out", default="deepmreye_out", help="output directory")
    p.add_argument("--model", default=None, help="a saved decoder (default: pretrained)")
    p.add_argument("--tr", type=float, default=None, help="override the header's TR (s)")
    p.add_argument("--qa", action="store_true", help="write a QA image of the registration")
    p.add_argument("--save-block", action="store_true", help="also write the eye block (.h5)")
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("preprocess", help="Extract one run's eye block to .h5, with labels.")
    p.add_argument("run", help="4D NIfTI run")
    p.add_argument("-o", "--out", required=True, help="output .h5, e.g. mystudy/sub-01.h5")
    p.add_argument("--labels", default=None, help=".npy gaze, [T, 10, 2] or [T, 2]")
    p.add_argument("--gaze", default=None,
                   help="raw eye-tracker samples (.tsv/.csv with time, x, y columns; time in s "
                        "from the first volume's onset), binned to ten samples per volume")
    p.add_argument("--gaze-offset", type=float, default=0.0,
                   help="subtract this from the gaze times (s), e.g. the first trigger's time")
    p.add_argument("--tr", type=float, default=None, help="override the header's TR (s)")
    p.add_argument("--qa", action="store_true", help="write a QA image beside the .h5")
    p.set_defaults(func=cmd_preprocess)

    subjects_help = "only these runs (file names or glob patterns, e.g. sub-01 'sub-0[1-4]*')"

    p = sub.add_parser("features", help="Write the basis coordinates of runs (e.g. as regressors).")
    p.add_argument("runs", nargs="+", help="4D NIfTI runs, or eye blocks written by preprocess")
    p.add_argument("-o", "--out", default="deepmreye_out", help="output directory")
    p.add_argument("--orbits", action="store_true",
                   help="keep the two orbits apart: [T, 2, 32] instead of their average [T, 32]")
    p.add_argument("--model", default=None, help="a saved decoder whose basis to use")
    p.add_argument("--tr", type=float, default=None)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="Fit the readout on gaze-labelled runs.")
    p.add_argument("--data-dir", default=None,
                   help="your labelled runs (<dataset>/<subject>.h5, or .h5 files directly); "
                        "default: the published labelled datasets")
    p.add_argument("--pattern", default="*", help="dataset folders under --data-dir")
    p.add_argument("--subjects", nargs="+", default=None, help=subjects_help)
    p.add_argument("--include-published", action="store_true",
                   help="also train on the nine published labelled datasets")
    p.add_argument("--basis", default=None, help="basis file (default: the published basis)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("-o", "--out", required=True, help="output decoder .npz")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("calibrate", help="Fit a gain and offset per axis to degrees.")
    p.add_argument("--data-dir", required=True, help="labelled runs from the study being decoded")
    p.add_argument("--pattern", default="*")
    p.add_argument("--subjects", nargs="+", default=None, help=subjects_help)
    p.add_argument("--model", default=None, help="decoder to calibrate (default: pretrained)")
    p.add_argument("--units", default="deg")
    p.add_argument("-o", "--out", required=True, help="output decoder .npz")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("evaluate", help="Leave-one-dataset-out on the labelled datasets.")
    p.add_argument("--data-dir", default=None, help=data_help)
    p.add_argument("--pattern", default="dsL*")
    p.add_argument("--basis", default=None, help="basis file (default: the published basis)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("--cache", default=None, help="keep the projections here between runs")
    p.add_argument("--rebuild", action="store_true", help="ignore an existing cache")
    p.add_argument("--check", action="store_true",
                   help="first reproduce the published numbers (published corpus and basis only)")
    p.add_argument("--json", default=None, help="write every number to this file")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("fit-basis", help="Fit the basis on unlabelled eye blocks.")
    p.add_argument("--data-dir", default=None, help=data_help + " (all of it: 77 GB)")
    p.add_argument("--n", type=int, default=None, help="only the first N eligible participants")
    p.add_argument("--k", type=int, default=256, help="canonical pairs stored")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", required=True, help="output basis .npz")
    p.set_defaults(func=cmd_fit_basis)

    p = sub.add_parser("download", help="Download the published corpus.")
    p.add_argument("--all", action="store_true", help="the whole corpus (77 GB), not only "
                   "the labelled datasets (20 GB)")
    p.add_argument("--data-dir", default=None, help="target directory (default: the cache)")
    p.set_defaults(func=cmd_download)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
