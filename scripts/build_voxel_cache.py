"""Build the voxel memmap caches the voxelnet trainers read.

Both caches were originally built ad hoc from an interactive session, which is not something
a second machine can reproduce. This is that step as a command.

    # labeled probe set: 337 participants, 9 datasets, ~405k TRs, ~11 GB fp16
    python scripts/build_voxel_cache.py --out results/subtr/voxels

    # unlabeled pretraining corpus: every participant WITHOUT labels, ~29 GB
    python scripts/build_voxel_cache.py --out results/subtr/voxels_unlabeled --unlabeled

    # the frozen canonical coordinates the incumbent reads (small, but needed by the trainer)
    python scripts/build_voxel_cache.py --out results/subtr/voxels --z-only

`--z-only` writes just `z_cca_k<K>.npy` beside an existing cache; the trainer computes it on
first use anyway, so this only exists to keep that cost off the first training job.

The unlabeled build asserts that no `dsL*` participant enters it. That assertion is the whole
point of a separate cache: one labeled participant leaking into pretraining would put the same
person on both sides of a leave-one-dataset-out split, and nothing downstream would notice.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepmreye.voxelnet import build_voxel_cache, cca_matrix, load_voxel_cache


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="Cache directory to write.")
    p.add_argument("--data-dir", default=None,
                   help="Corpus root. Default: DEEPMREYE_DATA / ./data / HuggingFace cache.")
    p.add_argument("--unlabeled", action="store_true",
                   help="Build the pretraining cache (participants with no `labels`).")
    p.add_argument("--max-participants", type=int, default=None)
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--z-only", action="store_true",
                   help="Skip the voxel build; only write z_cca_k<K>.npy for an existing cache.")
    p.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    args = p.parse_args()

    from deepmreye.datasource import resolve
    from deepmreye.unsupervised import corpus_mask, load_basis

    data_dir = resolve(args.data_dir, download=False, quiet=False)
    mask = corpus_mask(data_dir)
    print(f"[*] corpus {data_dir}")
    print(f"[*] mask {mask.shape}, {int(mask.sum())} voxels")

    if not args.z_only:
        meta = build_voxel_cache(data_dir, mask, args.out,
                                 dtype=np.dtype(args.dtype).type,
                                 labeled=not args.unlabeled,
                                 max_participants=args.max_participants)
        n_ds = len({q["dataset"] for q in meta["parts"]})
        print(f"[+] {len(meta['parts'])} participants across {n_ds} datasets")

    # The incumbent's frozen projection, precomputed for every row. The trainer will build
    # this on demand, but a 400k x 14236 pass is not something to pay for inside a GPU job.
    zc = Path(args.out) / f"z_cca_k{args.k}.npy"
    if zc.exists():
        print(f"[=] {zc} already present")
        return
    if args.unlabeled:
        print("[=] skipping z_cca for the unlabeled cache (the incumbent needs labels)")
        return
    _m, bases, _meta = load_basis(Path(args.basis))
    w_cca, mu = cca_matrix(bases["lr-cca"], k=args.k)
    vox, _lab, meta = load_voxel_cache(args.out, mask)
    z = np.empty((meta["n_rows"], args.k), dtype=np.float32)
    step = 20000
    for i in range(0, meta["n_rows"], step):
        j = min(i + step, meta["n_rows"])
        z[i:j] = (vox[i:j].astype(np.float32) - mu.astype(np.float32)) @ w_cca.astype(np.float32)
        if (i // step) % 5 == 0:
            print(f"    {j}/{meta['n_rows']} rows", flush=True)
    np.save(zc, z)
    print(f"[+] {zc}  {z.shape}")


if __name__ == "__main__":
    main()
