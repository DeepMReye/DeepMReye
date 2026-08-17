#!/usr/bin/env python3
"""The unlabeled-corpus scaling figure: `lr-cca` vs `corpus-pca` vs `fold-pca`.

Reads the per-N probe results written by `sweep_probe_scaling.py` and draws the
one comparison this project needs to make honestly. Four panels, because the
single headline number ("does the corpus basis beat the fold-local one?") is
*no*, and the interesting result is a different question ("does it improve with
unlabeled participants?"), which is *yes*.

- **A** median r against unlabeled corpus size. `fold-pca:64` is refitted per
  fold and cannot depend on N, so it is a horizontal reference, not a curve.
  Its own run-to-run spread (+-0.02, measured -- see STATE.md) is shaded, because
  every comparison here has to be read against it.
- **B** the gap to `fold-pca`, which is what "closing in" means quantitatively.
  It closes and then **saturates**; the earlier straight-line extrapolation to
  parity was wrong and the flat last segment is why.
- **C** the optimal component count *falls* as the corpus grows -- the reverse of
  the obvious guess, and the reason every `k` conclusion is conditional on N.
- **D** per fold at the full corpus, because the medians hide that the ranking is
  not uniform: `lr-cca` wins some folds outright and loses `dsL04`/`dsL06` badly.

    python scripts/plot_corpus_scaling.py
    python scripts/plot_corpus_scaling.py --out media/figures/corpus_scaling.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SIZES = [25, 50, 100, 200, 400, 800, 1039]

# Measured run-to-run variation of a 7-fold median (STATE.md): `fold-pca:64`
# reported 0.847 at 1000 training windows and 0.828 with all of them, a method
# that cannot get worse with more labels. Anything inside this band is a tie.
NOISE = 0.02

COLORS = {
    "lr-cca:64": "#c0392b",
    "lr-cca:32": "#e67e22",
    "corpus-pca:64": "#2471a3",
    "band-pca:64": "#7d3c98",
    "fold-pca:64": "#212121",
    "gev-slow:64": "#95a5a6",
}


def mean_r(entry):
    """Per-subject median r, averaged over the two gaze axes.

    Per participant first, then median across participants -- pooling every row
    of every subject rewards a model that only predicts *which subject this is*.
    """
    ps = entry["by_subject"]["per_subject"]
    if not ps:
        return float("nan")
    rx = np.median([v["pearson_r_x"] for v in ps.values()])
    ry = np.median([v["pearson_r_y"] for v in ps.values()])
    return float(np.mean([rx, ry]))


def load(results_dir, sizes):
    """{feature: {N: {fold: r}}}"""
    table = {}
    for n in sizes:
        path = Path(results_dir) / f"probe_n{n}.json"
        if not path.exists():
            print(f"[!] missing {path}, skipping N={n}")
            continue
        data = json.loads(path.read_text())
        for fold, arms in data.items():
            for key, entry in arms.items():
                feat = key.split("/")[0]
                table.setdefault(feat, {}).setdefault(n, {})[fold] = mean_r(entry)
    return table


def med(table, feat, n):
    vals = [v for v in table.get(feat, {}).get(n, {}).values() if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", default="results/scaling")
    p.add_argument("--out", default="media/figures/corpus_scaling.png")
    p.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = load(args.results_dir, args.sizes)
    sizes = [n for n in args.sizes if any(n in table.get(f, {}) for f in table)]
    ref = med(table, "fold-pca:64", sizes[-1])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    fig.suptitle("Does the unlabeled corpus help? Yes for lr-cca — and it still "
                 "does not beat a fold-local PCA",
                 fontsize=13, fontweight="bold", y=0.98)

    # ---- A: scaling curves -------------------------------------------------
    ax = axes[0, 0]
    ax.axhspan(ref - NOISE, ref + NOISE, color=COLORS["fold-pca:64"], alpha=0.10,
               zorder=0)
    ax.axhline(ref, color=COLORS["fold-pca:64"], ls="--", lw=1.8, zorder=1,
               label=f"fold-pca:64 = {ref:.3f}  (needs labels, no corpus)")
    for feat, marker in [("lr-cca:64", "o"), ("corpus-pca:64", "s"),
                         ("band-pca:64", "^"), ("gev-slow:64", "v")]:
        xs = [n for n in sizes if n in table.get(feat, {})]
        ys = [med(table, feat, n) for n in xs]
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, ms=6, lw=2, color=COLORS[feat],
                label=feat, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("unlabeled participants in the basis fit (N)")
    ax.set_ylabel("median r across 7 held-out datasets")
    ax.set_title("A  Scaling with unlabeled participants", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="center right", framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.055, "shaded band = ±0.02 noise floor (measured)",
            transform=ax.transAxes, fontsize=7.5, color="#555", va="bottom")
    ax.annotate("the control:\nWORSE with data",
                xy=(150, med(table, "gev-slow:64", 100)), xytext=(27, 0.40),
                fontsize=8, color="#5d6d7e", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=1.1))

    # ---- B: gap to the fold-local reference --------------------------------
    ax = axes[0, 1]
    ax.axhline(0, color=COLORS["fold-pca:64"], ls="--", lw=1.8,
               label="fold-pca:64 (parity)")
    ax.axhspan(-NOISE, NOISE, color=COLORS["fold-pca:64"], alpha=0.10)
    for feat, marker in [("lr-cca:64", "o"), ("corpus-pca:64", "s"),
                         ("band-pca:64", "^")]:
        xs = [n for n in sizes if n in table.get(feat, {})]
        ys = [med(table, feat, n) - ref for n in xs]
        ax.plot(xs, ys, marker=marker, ms=6, lw=2, color=COLORS[feat], label=feat)
    lo = med(table, "lr-cca:64", sizes[0]) - ref
    hi = med(table, "lr-cca:64", sizes[-1]) - ref
    ax.annotate(f"gap closes {lo:+.3f} → {hi:+.3f}\nthen saturates (N=800→1039)",
                xy=(sizes[-1], hi), xytext=(90, -0.155), fontsize=8.5,
                color=COLORS["lr-cca:64"],
                arrowprops=dict(arrowstyle="->", color=COLORS["lr-cca:64"], lw=1.2))
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("unlabeled participants in the basis fit (N)")
    ax.set_ylabel("median r − fold-pca:64")
    ax.set_title("B  The gap closes, but does not cross zero", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)

    # ---- C: the optimal k falls as the corpus grows ------------------------
    ax = axes[1, 0]
    ks = [32, 64, 128, 256]
    for n, color, marker in [(sizes[0], "#a9cce3", "o"),
                             (800 if 800 in sizes else sizes[-1], "#1a5276", "s")]:
        ys = [med(table, f"corpus-pca:{k}", n) for k in ks]
        ax.plot(ks, ys, marker=marker, ms=7, lw=2, color=color,
                label=f"corpus-pca, N={n}")
        best = int(np.nanargmax(ys))
        ax.plot(ks[best], ys[best], marker="*", ms=17, color=color, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("components kept (k)")
    ax.set_ylabel("median r")
    ax.set_title("C  The best k *falls* as the corpus grows", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.text(0.30, 0.36,
            "★ = optimum, and it MOVES: k=256 at N=25,\n"
            "k=64 at N=800. More unlabeled data buys a\n"
            "smaller representation, not a richer one —\n"
            "a well-estimated basis is compact.\n"
            "Retune k whenever the corpus changes.",
            transform=ax.transAxes, fontsize=8, va="top", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

    # ---- D: per fold at the full corpus -----------------------------------
    ax = axes[1, 1]
    n = sizes[-1]
    arms = ["lr-cca:32", "corpus-pca:64", "fold-pca:64"]
    arms = [a for a in arms if n in table.get(a, {})]
    folds = sorted(table[arms[-1]][n])
    x = np.arange(len(folds))
    w = 0.8 / len(arms)
    for j, feat in enumerate(arms):
        ys = [table[feat][n].get(f, np.nan) for f in folds]
        ax.bar(x + j * w - 0.4 + w / 2, ys, w, label=f"{feat} ({med(table, feat, n):.3f})",
               color=COLORS.get(feat, "#888"), alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("dsL", "").replace("_", "\n") for f in folds],
                       fontsize=7.5)
    ax.set_ylabel("r (held-out dataset)")
    ax.set_title(f"D  Per fold at the full corpus (N={n})", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, 1.06)
    ax.annotate("resolution limit:\ngaze outruns the TR.\n"
                "No method scores here.",
                xy=(2, 0.21), xytext=(1.55, 0.42), fontsize=7.5, color="#555",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.1))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"[*] -> {out}")

    print("\nmedian r by corpus size")
    hdr = f"{'feature':<16}" + "".join(f"{('N=' + str(n)):>9}" for n in sizes)
    print(hdr)
    print("-" * len(hdr))
    for feat in ["lr-cca:64", "corpus-pca:64", "band-pca:64", "fold-pca:64",
                 "gev-slow:64"]:
        if feat not in table:
            continue
        print(f"{feat:<16}" + "".join(
            f"{med(table, feat, n):>9.3f}" if n in table[feat] else f"{'--':>9}"
            for n in sizes))


if __name__ == "__main__":
    main()
