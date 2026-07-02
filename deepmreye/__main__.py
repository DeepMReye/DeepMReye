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

export-labels   Snapshot current QA labels from datasets.h5 to labels.csv.
restore-labels  Replay labels.csv back into datasets.h5 (e.g. after recompile).

The labeling UI also mirrors every save into labels.csv automatically, so
manual labels survive a corrupted or rebuilt registry. Commit labels.csv to
git to version your QA effort.

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
    run_compile(args.data_dir, limit=limit)


def cmd_qa(args):
    from label_datasets import run_labeler
    run_labeler(data_dir=args.data_dir, port=args.port)


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
    common.add_argument("--data-dir", type=str, default="./data", help="Central data storage directory (default: ./data).")

    parser = argparse.ArgumentParser(prog="python -m deepmreye", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", parents=[common], help="Sample subjects from OpenNeuro.")
    p_compile.add_argument("--limit", type=str, default="5", help="Datasets to sample. Use 'None' for all.")
    p_compile.set_defaults(func=cmd_compile)

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

    p_all = sub.add_parser("all", parents=[common], help="Run the whole pipeline, pausing for QA.")
    p_all.add_argument("--limit", type=str, default="5", help="Datasets to sample in compile.")
    p_all.add_argument("--force", action="store_true", help="Force reprocessing in preprocess.")
    p_all.add_argument("train_args", nargs=argparse.REMAINDER, help="Args forwarded to train_jepa.py.")
    p_all.set_defaults(func=cmd_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
