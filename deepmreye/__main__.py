"""Single entry point for the DeepMReye 2.0 pipeline.

Run any stage with:

    python -m deepmreye <command> [options]

Commands
--------
compile     Sample a few subjects per OpenNeuro dataset for manual QA.
qa          Launch the browser labeling UI to mark eyes / no eyes.
preprocess  Download and extract every subject of approved datasets.
all         Run compile -> qa -> preprocess, pausing for QA.

fetch       Download the corpus from HuggingFace up front (stages do it lazily).

export-labels   Snapshot current QA labels from datasets.h5 to labels.csv.
restore-labels  Replay labels.csv back into datasets.h5 (e.g. after recompile).

The labeling UI also mirrors every save into labels.csv automatically, so
manual labels survive a corrupted or rebuilt registry. Both files travel
between machines with scripts/sync_labels.py.

fit-basis   Fit the unsupervised feature basis on the unlabeled corpus.
evaluate    Leave-one-dataset-out gaze decoding: r, R-squared, error in degrees.

The stages are ordered: compile produces the registry, qa approves datasets,
preprocess extracts the full data, fit-basis learns the projection from the
unlabeled half, evaluate reads gaze out of the labeled half. Nothing here is
trained on gaze except the final ridge readout.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from deepmreye.pipeline import DEFAULT_REPORT, REPORT_MODES


def cmd_compile(args):
    from compile_openneuro import run_compile
    limit = None if str(args.limit).lower() == "none" else int(args.limit)
    run_compile(args.data_dir, limit=limit, workers=getattr(args, "workers", 4),
                force=getattr(args, "force", False),
                report=getattr(args, "report", DEFAULT_REPORT))


def cmd_merge_registry(args):
    from deepmreye import registry
    n = registry.merge_pending(args.data_dir)
    print(f"Merged {n} pending subject records into {args.data_dir}/datasets.h5")


def cmd_qa(args):
    from label_datasets import run_labeler
    run_labeler(data_dir=args.data_dir, port=args.port,
                no_download=getattr(args, 'no_download', False))


def cmd_preprocess(args):
    from download_and_preprocess import run_preprocess
    run_preprocess(args.data_dir, force=args.force,
                   report=getattr(args, "report", DEFAULT_REPORT))


def cmd_export_labels(args):
    from pathlib import Path
    from deepmreye.labels import export_labels
    data_dir = Path(args.data_dir).resolve()
    n = export_labels(data_dir / "datasets.h5", data_dir / "labels.csv")
    print(f"Snapshotted {n} labels to {data_dir / 'labels.csv'}")


def cmd_restore_labels(args):
    from pathlib import Path
    from deepmreye.labels import restore_labels
    data_dir = Path(args.data_dir).resolve()
    applied, missing = restore_labels(data_dir / "datasets.h5", data_dir / "labels.csv")
    print(f"Restored {applied} labels into {data_dir / 'datasets.h5'} ({missing} skipped, not in registry)")


def cmd_fetch(args):
    """Pull the corpus down eagerly, instead of letting a stage do it lazily."""
    from deepmreye.datasource import DEFAULT_REPO, REGISTRY_FILES, cache_dir, fetch

    patterns = list(REGISTRY_FILES)
    if not args.labels_only:
        patterns.append("*/*.h5")
        if args.reports:
            patterns.append("*/*/*.html")

    # No resolution here: fetch is the explicit form, so it always downloads to
    # a stated place rather than picking one.
    target = Path(args.data_dir) if args.data_dir else cache_dir()
    path = fetch(repo_id=args.repo_id or DEFAULT_REPO, target=target, patterns=patterns)
    print(f"[+] corpus at {path}")


def cmd_fit_basis(args):
    from fit_corpus_basis import main as fit_main
    sys.argv = ["fit_corpus_basis.py", "--data-dir", str(args.data_dir),
                "--out", args.out, "--k", str(args.k),
                "--trs-per-subject", str(args.trs_per_subject)]
    if args.max_subjects:
        sys.argv += ["--max-subjects", str(args.max_subjects)]
    fit_main()


def cmd_evaluate(args):
    from deepmreye import probe
    recs = probe.load_or_build(args.data_dir, args.basis, args.cache, args.m,
                               args.build_cache)
    print(f"[*] {len(recs)} participants, {len({r['dataset'] for r in recs})} datasets")
    if args.calibrate and not probe.calibrate(recs):
        raise SystemExit("[!] calibration failed -- do not trust this run")
    res = probe.lodo(recs, probe.incumbent(args.k, args.lags))
    probe.report(res, f"lr-cca:{args.k} + lags{args.lags}")
    if args.json:
        import json as _json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(_json.dumps(res, indent=2, default=float))
        print(f"[+] wrote {args.json}")


def cmd_all(args):
    cmd_compile(args)
    input(
        "\nCompile done. Next, run QA labeling in a separate terminal:\n"
        f"    python -m deepmreye qa --data-dir {args.data_dir}\n"
        "Press Enter here once labeling is finished to continue with preprocessing..."
    )
    cmd_preprocess(args)


def build_parser():
    # --data-dir lives only on the subparsers (one slot, one default) so it is
    # written after the command: `python -m deepmreye preprocess --data-dir X`.
    # Putting it on both the top-level parser and the subparsers lets the
    # subparser default silently clobber a value given before the command.
    common = argparse.ArgumentParser(add_help=False)
    # Default None, not "./data": an unset value means "work it out" --
    # $DEEPMREYE_DATA, then ./data, then the HuggingFace copy. See
    # deepmreye/datasource.py. Passing --data-dir explicitly always wins.
    common.add_argument("--data-dir", type=str, default=None,
                        help="Corpus directory. Default: $DEEPMREYE_DATA, else ./data, "
                             "else download from HuggingFace into the local cache.")
    common.add_argument("--no-download", action="store_true",
                        help="Never fetch from HuggingFace; fail if no local corpus.")

    parser = argparse.ArgumentParser(prog="python -m deepmreye", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", parents=[common], help="Sample subjects from OpenNeuro.")
    p_compile.add_argument("--limit", type=str, default="5", help="Datasets to sample. Use 'None' for all.")
    p_compile.add_argument("--workers", type=int, default=4, help="Parallel download/registration workers.")
    p_compile.add_argument("--force", action="store_true", help="Re-extract subjects already on disk.")
    p_compile.add_argument("--report", choices=REPORT_MODES, default=DEFAULT_REPORT,
                           help="QA artifact per subject: the ~20 KB thumbnail (png, default), the "
                             "~5 MB Plotly report (html), or both.")
    p_compile.set_defaults(func=cmd_compile)

    p_merge = sub.add_parser("merge-registry", parents=[common],
                             help="Fold pending worker records into datasets.h5.")
    p_merge.set_defaults(func=cmd_merge_registry)

    p_qa = sub.add_parser("qa", parents=[common], help="Launch the labeling UI.")
    p_qa.add_argument("--port", type=int, default=5050, help="Port for the labeling UI.")
    p_qa.set_defaults(func=cmd_qa)

    p_pre = sub.add_parser("preprocess", parents=[common], help="Extract all subjects of approved datasets.")
    p_pre.add_argument("--force", action="store_true", help="Reprocess and overwrite existing extractions.")
    p_pre.add_argument("--report", choices=REPORT_MODES, default=DEFAULT_REPORT,
                       help="QA artifact per subject: the ~20 KB thumbnail (png, default), the "
                             "~5 MB Plotly report (html), or both.")
    p_pre.set_defaults(func=cmd_preprocess)

    p_export = sub.add_parser("export-labels", parents=[common], help="Snapshot current QA labels to labels.csv.")
    p_export.set_defaults(func=cmd_export_labels)

    p_restore = sub.add_parser("restore-labels", parents=[common], help="Replay labels.csv back into datasets.h5.")
    p_restore.set_defaults(func=cmd_restore_labels)

    p_fetch = sub.add_parser("fetch", parents=[common],
                             help="Download the corpus from HuggingFace up front.")
    p_fetch.add_argument("--repo-id", default=None, help="Defaults to $DEEPMREYE_HF_REPO.")
    p_fetch.add_argument("--reports", action="store_true",
                         help="Also pull the QA report HTML (~8 GB). Only needed to "
                              "label offline; the UI otherwise fetches them per dataset.")
    p_fetch.add_argument("--labels-only", action="store_true",
                         help="Registry, label backup and index only (a few MB).")
    p_fetch.set_defaults(func=cmd_fetch)

    p_basis = sub.add_parser("fit-basis", parents=[common],
                             help="Fit the unsupervised basis on the unlabeled corpus.")
    p_basis.add_argument("--out", default="results/basis.npz")
    p_basis.add_argument("--k", type=int, default=256, help="Components kept per basis.")
    p_basis.add_argument("--trs-per-subject", type=int, default=48)
    p_basis.add_argument("--max-subjects", type=int, default=None)
    p_basis.set_defaults(func=cmd_fit_basis)

    p_eval = sub.add_parser("evaluate", parents=[common],
                            help="Leave-one-dataset-out gaze decoding.")
    p_eval.add_argument("--basis", default="results/basis.npz")
    p_eval.add_argument("--cache", default="results/labeled_cache.npz")
    p_eval.add_argument("--m", type=int, default=256, help="Directions kept in the cache.")
    p_eval.add_argument("--k", type=int, default=32, help="Directions the readout uses.")
    p_eval.add_argument("--lags", type=int, default=1)
    p_eval.add_argument("--build-cache", action="store_true")
    p_eval.add_argument("--calibrate", action="store_true",
                        help="Reproduce the known headline numbers before reporting.")
    p_eval.add_argument("--json", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_all = sub.add_parser("all", parents=[common], help="Run the whole pipeline, pausing for QA.")
    p_all.add_argument("--limit", type=str, default="5", help="Datasets to sample in compile.")
    p_all.add_argument("--force", action="store_true", help="Force reprocessing in preprocess.")
    p_all.set_defaults(func=cmd_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Every stage works on a resolved local directory, so the commands are
    # identical on the cluster (data on scratch) and on a laptop (data pulled
    # from HuggingFace on first use).
    if hasattr(args, "data_dir") and args.command != "fetch":
        from deepmreye.datasource import resolve
        # Stages that *produce* the corpus must never try to download it, and
        # must not fail when it does not exist yet -- they are what creates it.
        creates_data = args.command in ("compile", "preprocess", "all", "fit-basis")
        if creates_data and args.data_dir is None:
            try:
                args.data_dir = str(resolve(None, download=False, quiet=True))
            except FileNotFoundError:
                args.data_dir = "./data"
        else:
            from deepmreye.datasource import STAGE_PATTERNS
            args.data_dir = str(resolve(
                args.data_dir,
                download=not getattr(args, "no_download", False),
                patterns=STAGE_PATTERNS.get(args.command),
            ))

    args.func(args)


if __name__ == "__main__":
    main()
