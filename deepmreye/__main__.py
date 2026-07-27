"""Single entry point for the DeepMReye 2.0 pipeline.

Run any stage with:

    python -m deepmreye <command> [options]

Commands
--------
compile     Sample a few subjects per OpenNeuro dataset for manual QA.
qa          Launch the browser labeling UI to mark eyes / no eyes.
preprocess  Download and extract every subject of approved datasets.
train       Train the JEPA model on the approved data.
all         Run compile -> qa -> preprocess -> train, pausing for QA.

fetch       Download the corpus from HuggingFace up front (stages do it lazily).

export-labels   Snapshot current QA labels from datasets.h5 to labels.csv.
restore-labels  Replay labels.csv back into datasets.h5 (e.g. after recompile).

The labeling UI also mirrors every save into labels.csv automatically, so
manual labels survive a corrupted or rebuilt registry. Both files travel
between machines with scripts/sync_labels.py.

The stages are ordered: compile produces the registry, qa approves datasets,
preprocess extracts the full data, and train consumes it.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def cmd_compile(args):
    from compile_openneuro import run_compile
    limit = None if str(args.limit).lower() == "none" else int(args.limit)
    run_compile(args.data_dir, limit=limit, workers=getattr(args, "workers", 4),
                force=getattr(args, "force", False))


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
    run_preprocess(args.data_dir, force=args.force)


def cmd_train(args):
    import runpy
    # train_jepa parses its own args; forward the remaining CLI verbatim.
    sys.argv = ["train_jepa.py", "--data_dir", args.data_dir] + args.train_args
    runpy.run_path(str(SCRIPTS_DIR / "train_jepa.py"), run_name="__main__")


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


def cmd_all(args):
    cmd_compile(args)
    input(
        "\nCompile done. Next, run QA labeling in a separate terminal:\n"
        f"    python -m deepmreye qa --data-dir {args.data_dir}\n"
        "Press Enter here once labeling is finished to continue with preprocessing..."
    )
    cmd_preprocess(args)
    cmd_train(args)


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
    p_compile.set_defaults(func=cmd_compile)

    p_merge = sub.add_parser("merge-registry", parents=[common],
                             help="Fold pending worker records into datasets.h5.")
    p_merge.set_defaults(func=cmd_merge_registry)

    p_qa = sub.add_parser("qa", parents=[common], help="Launch the labeling UI.")
    p_qa.add_argument("--port", type=int, default=5050, help="Port for the labeling UI.")
    p_qa.set_defaults(func=cmd_qa)

    p_pre = sub.add_parser("preprocess", parents=[common], help="Extract all subjects of approved datasets.")
    p_pre.add_argument("--force", action="store_true", help="Reprocess and overwrite existing extractions.")
    p_pre.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train", parents=[common], help="Train the JEPA model.")
    p_train.add_argument("train_args", nargs=argparse.REMAINDER, help="Args forwarded to train_jepa.py.")
    p_train.set_defaults(func=cmd_train)

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

    p_all = sub.add_parser("all", parents=[common], help="Run the whole pipeline, pausing for QA.")
    p_all.add_argument("--limit", type=str, default="5", help="Datasets to sample in compile.")
    p_all.add_argument("--force", action="store_true", help="Force reprocessing in preprocess.")
    p_all.add_argument("train_args", nargs=argparse.REMAINDER, help="Args forwarded to train_jepa.py.")
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
        creates_data = args.command in ("compile", "preprocess", "all")
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
