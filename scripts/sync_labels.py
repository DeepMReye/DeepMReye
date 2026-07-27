#!/usr/bin/env python3
"""Move QA labels between machines without moving the corpus.

The labels are the expensive, irreplaceable part of this project and they are
also tiny: `datasets.h5` is a few MB of metadata and `labels.csv` is text, next
to ~29 GB of eye blocks. So they get their own round trip -- label on a laptop,
push the labels, pull them on the cluster, and run the full extraction there.

    python scripts/sync_labels.py push --data-dir DATA --repo-id ORG/REPO
    python scripts/sync_labels.py pull --data-dir DATA --repo-id ORG/REPO

Pull merges rather than overwrites: a label already in the local registry is
kept unless the remote one is newer *and* the local slot is still unlabeled.
That way a pull can never silently undo labeling done on the other machine.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABEL_FILES = ["datasets.h5", "labels.csv"]


def push(data_dir, repo_id, private=True):
    from huggingface_hub import HfApi

    data_dir = Path(data_dir)
    present = [f for f in LABEL_FILES if (data_dir / f).exists()]
    if not present:
        print(f"Nothing to push: none of {LABEL_FILES} found in {data_dir}")
        return

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for name in present:
        size = (data_dir / name).stat().st_size / 1e6
        print(f"  pushing {name} ({size:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(data_dir / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"[+] labels pushed to {repo_id}")


def pull(data_dir, repo_id, force=False):
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in LABEL_FILES:
        try:
            remote = hf_hub_download(repo_id, name, repo_type="dataset")
        except Exception as e:
            print(f"  [-] {name}: not on the remote ({e.__class__.__name__})")
            continue

        local = data_dir / name
        if name == "datasets.h5" and local.exists() and not force:
            applied = _merge_registry(remote, local)
            print(f"  merged {applied} labels from remote into {name}")
        else:
            # Keep a copy of what we are replacing; labels are irreplaceable.
            if local.exists():
                shutil.copy2(local, local.with_suffix(local.suffix + ".bak"))
            shutil.copy2(remote, local)
            print(f"  copied {name}")

    print(f"[+] labels pulled from {repo_id} into {data_dir}")


def _merge_registry(remote_path, local_path):
    """Copy `approved` attributes from the remote registry into the local one.

    Only fills slots that are still unlabeled locally, so pulling can never
    overwrite labeling done on this machine. Conflicts are reported, not
    silently resolved.
    """
    import h5py

    applied, conflicts = 0, []
    with h5py.File(remote_path, "r") as src, h5py.File(local_path, "a") as dst:
        for ds_name in src.keys():
            if ds_name not in dst:
                continue
            r_ds, l_ds = src[ds_name], dst[ds_name]

            r_lbl = r_ds.attrs.get("approved", -1)
            if r_lbl != -1 and l_ds.attrs.get("approved", -1) == -1:
                l_ds.attrs["approved"] = r_lbl
                applied += 1

            for sub in r_ds.keys():
                if sub not in l_ds:
                    continue
                remote_lbl = r_ds[sub].attrs.get("approved", -1)
                local_lbl = l_ds[sub].attrs.get("approved", -1)
                if remote_lbl == -1 or remote_lbl == local_lbl:
                    continue
                if local_lbl == -1:
                    l_ds[sub].attrs["approved"] = remote_lbl
                    applied += 1
                else:
                    conflicts.append((ds_name, sub, local_lbl, remote_lbl))

    if conflicts:
        print(f"  [!] {len(conflicts)} subjects labeled differently on both sides; "
              f"kept the local value:")
        for ds, sub, loc, rem in conflicts[:10]:
            print(f"      {ds}/{sub}: local={loc} remote={rem}")
        if len(conflicts) > 10:
            print(f"      ... and {len(conflicts) - 10} more")
    return applied


def main():
    parser = argparse.ArgumentParser(description="Sync QA labels via HuggingFace.")
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--data-dir", default=None,
                        help="Corpus directory. Defaults to the usual resolution order.")
    parser.add_argument("--repo-id", default=None, help="Defaults to $DEEPMREYE_HF_REPO.")
    parser.add_argument("--public", action="store_true", help="Create the repo public on push.")
    parser.add_argument("--force", action="store_true",
                        help="On pull, overwrite the local registry instead of merging.")
    args = parser.parse_args()

    from deepmreye.datasource import DEFAULT_REPO, resolve
    data_dir = resolve(args.data_dir, download=False, quiet=True) if args.data_dir is None \
        else Path(args.data_dir)
    repo_id = args.repo_id or DEFAULT_REPO

    if args.action == "push":
        push(data_dir, repo_id, private=not args.public)
    else:
        pull(data_dir, repo_id, force=args.force)


if __name__ == "__main__":
    main()
