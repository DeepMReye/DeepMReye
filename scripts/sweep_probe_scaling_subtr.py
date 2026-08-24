"""Does the unlabeled corpus still buy anything between N=1039 and N=2009?

`CLAUDE.md` records `lr-cca` gaining **+0.150** from N=25 to N=800 and then *saturating*
between N=800 and N=1039 -- with an explicit warning not to extrapolate, because an earlier
straight-line projection to parity at N~1800 was wrong. The corpus has since grown to 2009
eligible participants and the headline basis is `basis_n2000`, so "is it still rising" has
never actually been measured. This script measures it.

Two things it does that `sweep_probe_scaling.py` cannot:

- **It scores at sub-TR resolution.** That script goes through `eval_probe`, which
  `temporal_probe`'s own docstring establishes cannot score sub-TR gaze at all -- every number
  it produces is 1-TR mean gaze at 5-TR bins. The sub-TR figure is the headline for this
  project, so a scaling curve that cannot see it answers the wrong question. Both resolutions
  are reported here, from the single audited `lodo_subtr`.
- **It rebuilds the labeled cache per basis.** The cached canonical coordinates are a
  *projection through a specific basis*; reusing one basis's cache to score another silently
  measures the wrong thing, which is why `load_subtr_cache` refuses a mismatched basis path.

`k` is fixed at the value `sweep_k_at_n2000.py` confirmed optimal rather than retuned per
checkpoint: retuning it per point would make the curve a comparison of *tuned* arms, which is
a different and more flattering question than "does the same method improve with data".
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepmreye.temporal_probe import (build_subtr_cache, cca_avg, corpus_fingerprint,
                                      load_subtr_cache, lodo_subtr, make_lags,
                                      save_subtr_cache)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis-dir", default="results/scaling_ext")
    p.add_argument("--cache-dir", default="results/subtr/scaling_caches")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lags-subtr", type=int, default=1)
    p.add_argument("--lags-1tr", type=int, default=0)
    p.add_argument("--out", default="results/scaling_ext/probe_curve_subtr.json")
    args = p.parse_args()

    from deepmreye.datasource import resolve
    from deepmreye.unsupervised import corpus_mask, load_basis

    data_dir = Path(args.data_dir) if args.data_dir else resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(args.basis_dir).glob("basis_n*.npz"),
                   key=lambda q: int(re.search(r"basis_n(\d+)", q.name).group(1)))
    if not paths:
        raise SystemExit(f"[!] no basis_n*.npz under {args.basis_dir}")
    sizes = [int(re.search(r"basis_n(\d+)", q.name).group(1)) for q in paths]
    print(f"[*] {len(paths)} checkpoints: {', '.join(str(v) for v in sizes)}", flush=True)

    rows, fp0 = [], None
    for path, n in zip(paths, sizes):
        cache = Path(args.cache_dir) / f"subtr_n{n}.npz"
        if cache.exists():
            recs = load_subtr_cache(cache, path, args.m, False)
        else:
            _m, bases, _meta = load_basis(path)
            recs = build_subtr_cache(data_dir, mask, bases["lr-cca"], m=args.m)
            save_subtr_cache(cache, recs, path, args.m, False)

        # Every checkpoint must score the SAME labeled participants; a curve whose corpus
        # drifts between points is measuring two things at once.
        fp = corpus_fingerprint(recs)
        if fp0 is None:
            fp0 = fp
        elif fp != fp0:
            raise SystemExit(f"[!] N={n} scores a different labeled corpus ({fp[:12]} vs "
                             f"{fp0[:12]}) -- the curve would be uninterpretable")

        sub = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, args.k), args.lags_subtr))
        one = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, args.k), args.lags_1tr))
        rows.append({"n": n, "subtr": sub["median_subtr"], "1tr": one["median_1tr"],
                     "subtr_folds": sub.get("subtr", {}), "1tr_folds": one.get("1tr", {})})
        print(f"  N={n:<5} sub-TR {sub['median_subtr']:.4f}   1-TR {one['median_1tr']:.4f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"k": args.k, "m": args.m, "lags_subtr": args.lags_subtr, "lags_1tr": args.lags_1tr,
         "fingerprint": fp0, "rows": rows}, indent=2))

    print(f"\n{'N':>6} {'sub-TR':>9} {'d vs prev':>11} {'1-TR':>9} {'d vs prev':>11}")
    for i, r in enumerate(rows):
        ds = f"{r['subtr'] - rows[i-1]['subtr']:+.4f}" if i else "     -"
        do = f"{r['1tr'] - rows[i-1]['1tr']:+.4f}" if i else "     -"
        print(f"{r['n']:>6} {r['subtr']:>9.4f} {ds:>11} {r['1tr']:>9.4f} {do:>11}")
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
