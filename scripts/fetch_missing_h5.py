"""Fetch participant `.h5` files present on the Hub but absent locally, one at a time.

`snapshot_download` is the wrong tool on a Leonardo login node for the reason `CLAUDE.md`
documents: the 32 GB cgroup is shared across the whole session and **page cache from your own
writes counts against it**, so a large download is killed with tens of GB of cache and ~0 GB
RSS -- a failure that reads like a memory leak while the process is doing nothing of the sort.

So this downloads file by file and calls `posix_fadvise(POSIX_FADV_DONTNEED)` on each one as
soon as it lands. The files are read back later on a compute node, so evicting them costs
nothing. Memory stays flat in the number of files: no dict of futures, no accumulated
listing, one file in flight.

Resumable by construction -- it re-lists what is missing every run, so an interrupted fetch is
restarted by running it again.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def drop_cache(path):
    """Evict a just-written file from the page cache."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=os.environ.get("DEEPMREYE_DATA"))
    p.add_argument("--repo-id", default="DeepMReye/eyeballs")
    p.add_argument("--limit", type=int, default=0, help="0 = all missing files.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    target = Path(args.data_dir)
    if not target.is_dir():
        raise SystemExit(f"[!] {target} is not a directory")

    api = HfApi()
    repo = {f for f in api.list_repo_files(args.repo_id, repo_type="dataset")
            if f.endswith(".h5") and f.count("/") == 1}
    local = {f"{q.parent.name}/{q.name}" for q in target.glob("*/*.h5")}
    missing = sorted(repo - local)
    if args.limit:
        missing = missing[:args.limit]

    print(f"[*] repo {len(repo)}  local {len(local)}  missing {len(missing)}", flush=True)
    if not missing:
        print("[+] nothing to do")
        return
    by_ds = {}
    for m in missing:
        by_ds.setdefault(m.split("/")[0], 0)
        by_ds[m.split("/")[0]] += 1
    for ds, n in sorted(by_ds.items(), key=lambda kv: -kv[1]):
        print(f"    {ds:<22} {n}")
    if args.dry_run:
        return

    t0, done = time.time(), 0
    for i, rel in enumerate(missing, 1):
        try:
            got = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=rel,
                                  local_dir=str(target))
        except Exception as e:                      # one bad file must not end the run
            print(f"  [!] {rel}: {type(e).__name__} {e}", flush=True)
            continue
        drop_cache(got)
        done += 1
        if i % 10 == 0 or i == len(missing):
            print(f"  {i}/{len(missing)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[+] {done} of {len(missing)} downloaded", flush=True)


if __name__ == "__main__":
    main()
