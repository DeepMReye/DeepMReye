#!/usr/bin/env python3
"""Self-supervised pretraining of the voxel encoder on the unlabeled corpus.

Every arm so far has been supervised on 337 labeled participants. This is the one direction
that does not need the residual to be predictable from labels: 3,318 unlabeled participants
across 903 accessions, none of which can appear in the probe set (`build_voxel_cache` asserts
no `dsL*` path enters).

**The objective is chosen to dodge a failure this corpus has already measured.** Plain next-TR
prediction learns the *nuisance*: over corpus-PCA coordinates the next TR is predictable at
R^2 0.32, concentrated in components 0-8 (38% of variance, R^2 0.59) against 128-256 (R^2 0.09),
and those leading components are global signal, motion and drift. An objective dominated by
what is predictable optimises drift, which is why the `ar-gru` arm scored *below* its own
untrained control.

So: **band-matched temporal contrast.** Anchor at `t`, positive at `t +- {1, 2}`, and negatives
drawn from the **same run** at `|delta| in [3, 20]` TRs. Same-run negatives hold participant,
anatomy, scanner and slow drift constant by construction, so none of those can separate anchor
from negative -- the only thing that varies on that timescale is where the eyes are. Far or
cross-run negatives would make drift discriminative again, which is the CPC failure mode and
the next-TR result in contrastive clothing.

Registered kill signal: if the learned features' lag-1 autocorrelation sits in the corpus
nuisance band (0.83-0.87) rather than the gaze band (~0.85 at dsL02, but with structure at
the eye-movement timescale), the objective found drift anyway.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepmreye.voxelnet import build_net, cca_matrix, load_voxel_cache, shift_augment  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", default="results/subtr/voxels_unlabeled")
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--encoder", default="cnn", choices=("cnn", "lowrank"))
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--batch", type=int, default=24, help="anchors per step")
    p.add_argument("--negatives", type=int, default=8)
    p.add_argument("--pos-max", type=int, default=2, help="positive offset, TRs")
    p.add_argument("--neg-min", type=int, default=3)
    p.add_argument("--neg-max", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--shift", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/subtr/pretrained_encoder.pt")
    args = p.parse_args()

    import torch
    import torch.nn.functional as F

    from deepmreye.datasource import resolve
    from deepmreye.unsupervised import corpus_mask, load_basis

    dev = (torch.device("mps") if args.device == "auto" and torch.backends.mps.is_available()
           else torch.device(args.device if args.device != "auto" else "cpu"))
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    _m, bases, _meta = load_basis(Path(args.basis))
    w_cca, mu = cca_matrix(bases["lr-cca"], k=32)

    vox, _lab, meta = load_voxel_cache(args.voxels, mask)
    parts = [q for q in meta["parts"] if q["n"] >= args.neg_max + args.pos_max + 2]
    print(f"[*] {len(parts)} unlabeled participants, {meta['n_rows']:,} TRs, device {dev}",
          flush=True)

    mask_idx = np.flatnonzero(mask.reshape(-1))
    net = build_net(w_cca, mu, lags=1, encoder=args.encoder, rank=args.rank, width=args.width,
                    dropout=0.0, grid_shape=tuple(meta["mask_shape"]), mask_idx=mask_idx,
                    seed=args.seed).to(dev)
    mask_t = torch.as_tensor(mask_idx, dtype=torch.long, device=dev)
    enc_params = [q for n, q in net.named_parameters()
                  if n.startswith(("conv", "enc"))]
    opt = torch.optim.AdamW(enc_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    gen = np.random.default_rng(args.seed)

    def gather(rows):
        return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=dev)

    def views(x):
        """Two independently augmented views of the same volumes."""
        out = []
        for _ in range(2):
            v = shift_augment(x[None], mask_t, tuple(meta["mask_shape"]), args.shift, gen)[0]
            if args.noise > 0:
                v = v + args.noise * torch.randn_like(v)
            out.append(v)
        return out

    history = []
    t0 = time.time()
    for step in range(args.steps):
        anchors, positives, negatives = [], [], []
        for _ in range(args.batch):
            m = parts[int(gen.integers(len(parts)))]
            lo = m["start"] + args.neg_max
            hi = m["start"] + m["n"] - args.neg_max - 1
            if hi <= lo:
                continue
            t = int(gen.integers(lo, hi))
            d_pos = int(gen.choice([-2, -1, 1, 2][: 2 * args.pos_max]))
            anchors.append(np.asarray(vox[t], dtype=np.float32))
            positives.append(np.asarray(vox[t + d_pos], dtype=np.float32))
            for _ in range(args.negatives):
                d = int(gen.integers(args.neg_min, args.neg_max + 1)) * (1 if gen.random() < 0.5 else -1)
                negatives.append(np.asarray(vox[t + d], dtype=np.float32))
        if len(anchors) < 4:
            continue

        b = len(anchors)
        xa, xp = views(gather(anchors))[0], views(gather(positives))[1]
        xn = views(gather(negatives))[0]

        net.train()
        za = F.normalize(net.encode(xa[None])[0], dim=-1)
        zp = F.normalize(net.encode(xp[None])[0], dim=-1)
        zn = F.normalize(net.encode(xn[None])[0], dim=-1).view(b, args.negatives, -1)

        pos = (za * zp).sum(-1, keepdim=True) / args.temperature      # [B, 1]
        neg = torch.einsum("bd,bkd->bk", za, zn) / args.temperature   # [B, K]
        logits = torch.cat([pos, neg], dim=1)
        loss = F.cross_entropy(logits, torch.zeros(b, dtype=torch.long, device=dev))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_params, 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0 or step == args.steps - 1:
            acc = float((logits.argmax(1) == 0).float().mean())
            history.append({"step": step, "loss": float(loss.detach()), "acc": acc})
            print(f"    step {step:>5} loss {float(loss.detach()):.4f}  "
                  f"pos-rank-1 {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": {k: v.cpu() for k, v in net.state_dict().items()
                            if k.startswith(("conv", "enc"))},
                "arch": net.arch(), "args": vars(args), "history": history}, args.out)
    print(f"[+] {args.out}")


if __name__ == "__main__":
    main()
