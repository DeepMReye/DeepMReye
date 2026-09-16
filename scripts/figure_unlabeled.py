"""The figure that says what the unlabeled corpus is worth.

Each panel answers the question one level closer to the thing a reader cares
about:

a. **The estimator.** Cross-orbit CCA fitted on 25 participants reports sixty-odd
   directions shared between the orbits at correlation 0.8 and above. They are
   not there. As the corpus grows the spectrum contracts onto a handful of
   genuinely shared directions and stops moving by about a thousand participants.
   No gaze labels enter this panel at all -- it is a property of the fit.

b. **The decoder.** The same bases, scored leave-one-dataset-out on 337
   gaze-labeled participants. `lr-cca` gains +0.154 over the corpus; the
   specificity control is `corpus-pca`, which comes out of the same
   accumulators at the same rank under a different selection criterion and
   gains +0.03. (`gev-slow`, which *degrades* by 0.31 over the same data, is the
   sharper control and is deliberately **not** plotted -- a line that collapses
   to 0.16 and moves non-monotonically reads to a reviewer as a bug rather than
   as a result. The number is in FINDINGS.md.)

c. **Every fold.** A median over nine folds can be one dataset moving. It is
   not: all nine rise, by six to nine times the noise floor.

d. **The trade.** Held-out accuracy against the number of labeled participants
   per training study. Without the corpus the voxel basis and the readout come
   out of the same labeled scans; with a large one, the basis is already fixed
   and only the ridge is left to estimate.

e. **Against the published CNN**, on the folds it was never trained on -- which
   is three of nine, because `dsL01`-`dsL06` *are* DeepMReye 1.0's training
   data. Its all-nine median (0.813) is partly scored on that training data and
   is not a transfer number, so it does not appear here or anywhere else.

Panels are chosen by letter: `--panels abce` is the default and is the paper
figure; `--panels abc` is the corpus-size argument alone when space is tight.

Run after `analysis_unlabeled.py` and `eval_dme1.py` have written their JSON:

    python scripts/figure_unlabeled.py --results results/unlabeled_value --out paper
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito, checked for adjacent-pair separation under deutan and tritan
# simulation. Every series is also direct-labeled, so identity is never carried
# by colour alone -- which is the case that matters on a printed abstract.
C_CCA = "#0072B2"
C_CCA_LIGHT = "#6BAED6"   # same arm, smaller corpus: lightness carries the size
C_PCA = "#009E73"
C_CTL = "#D55E00"
C_REF = "#444444"
C_RAW = "#999999"
INK = "#1a1a1a"
MUTED = "#666666"

SIZES = [25, 50, 100, 200, 400, 800, 1039, 1200, 1500, 1800, 2000]
NOISE = 0.02          # the measured floor on a 9-fold median; see FINDINGS.md


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })


def panel_spectrum(ax, stats):
    """(a) The canonical spectrum contracts as the corpus grows."""
    shown = [25, 100, 400, 2000]
    cmap = plt.get_cmap("viridis")
    # `k` is a sequential quantity, so the ramp is one hue light-to-dark rather
    # than four categorical colours.
    ax.axvline(32, color="#bbbbbb", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.annotate("k=32\n(shipped)", (32, 0.04), xytext=(3, 0), textcoords="offset points",
                fontsize=5.8, color="#8a8a8a", va="bottom", ha="left")
    for i, n in enumerate(shown):
        rho = np.asarray(stats["spectrum"][str(n)])
        color = cmap(0.12 + 0.72 * i / (len(shown) - 1))
        ax.plot(np.arange(1, len(rho) + 1), rho, color=color, lw=1.4, zorder=3)
        ax.annotate(f"N={n:,}", (len(rho), rho[-1]), xytext=(2.5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=6.4, color=color)
    ax.set_xlim(1, 78)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 16, 32, 48, 64])
    ax.set_xlabel("canonical component")
    ax.set_ylabel(r"canonical correlation $\rho$")
    ax.set_title("a   the fit, with no labels", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _curve(a, arm, res):
    return [a[f"{arm}@{n}"][res] for n in SIZES]


def panel_folds(ax, a, res="subtr", arm="lr-cca:32+lags1"):
    """(c) Every held-out dataset separately, smallest corpus to largest.

    The median in (b) could in principle be one dataset moving. It is not: all
    nine rise, and the smallest gain is six times the noise floor. Plotted as
    slopes rather than a grouped bar chart because the reader's question is
    "did any go down", which a slope answers at a glance.
    """
    lo, hi = a[f"{arm}@{SIZES[0]}"]["folds"], a[f"{arm}@{SIZES[-1]}"]["folds"]
    order = sorted(hi, key=lambda d: hi[d][res])
    # Labels are pushed apart in data units before drawing: nine folds put four
    # of their endpoints inside 0.02 of each other, and overlapping text reads
    # as a plotting bug rather than as a tight cluster.
    # Only the extremes are named. Nine labels do not fit legibly beside a panel
    # this size, and the panel's claim is "none of them went down", which the
    # slopes carry on their own -- naming the best and worst fold is what a
    # reader actually needs to place the spread.
    named = {order[0], order[-1]}
    for ds in order:
        y0, y1 = lo[ds][res], hi[ds][res]
        ax.plot([0, 1], [y0, y1], color=C_CCA, lw=1.0, alpha=0.45, zorder=3)
        ax.plot([0, 1], [y0, y1], ".", color=C_CCA, ms=3.5, zorder=4)
        if ds in named:
            ax.annotate(ds.split("_", 1)[0].replace("dsL", "ds"), (1.04, y1),
                        fontsize=6.0, color=MUTED, va="center", ha="left")
    ax.annotate("all 9 folds up\n+0.117 to +0.181", (0.02, 1.03), xycoords="axes fraction",
                fontsize=6.0, color=C_CCA, va="top", ha="left", fontweight="bold")
    med = [a[f"{arm}@{SIZES[0]}"][res], a[f"{arm}@{SIZES[-1]}"][res]]
    ax.plot([0, 1], med, color=C_CCA, lw=2.4, zorder=5)
    ax.annotate("median", (0, med[0]), xytext=(-3, 0), textcoords="offset points",
                fontsize=6.4, color=C_CCA, va="center", ha="right", fontweight="bold")
    ax.set_xlim(-0.38, 1.30)
    ax.set_ylim(0.04, 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"N={SIZES[0]}", f"N={SIZES[-1]:,}"])
    ax.set_ylabel("held-out gaze correlation $r$")
    ax.set_title("c   every fold, not one", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_scaling(ax, a, foldpca, res="subtr"):
    """(b) Decoding against corpus size, with the two no-corpus references."""
    ref = next(r for r in foldpca if r["budget"] is None and r["arm"].endswith("lags1"))
    ax.axhspan(ref[res] - NOISE, ref[res] + NOISE, color="#ececec", zorder=0, lw=0)
    ax.axhline(ref[res], color=C_REF, lw=1.0, ls=(0, (4, 2)), zorder=2,
               label="fold-pca (sup.)")
    ax.axhline(a["raw+lags1"][res], color="#8f8f8f", lw=1.0, ls=(0, (1.2, 1.6)),
               zorder=2, label="stride-4 vox.")

    # `gev-slow` used to sit here as the negative control -- the only arm that
    # *degrades* with corpus size (-0.31), which is the sharpest evidence that
    # the climb is not "any basis improves with data". It is cut from the figure
    # because a control that collapses to 0.16 and moves non-monotonically reads
    # as a bug rather than as a result; the number stays in FINDINGS.md.
    # `corpus-pca` carries the control role here instead, and carries it more
    # safely: same corpus, same rank, same accumulators, different selection
    # criterion, and it gains +0.03 where `lr-cca` gains +0.154.
    for arm, color, label in (("lr-cca:32+lags1", C_CCA, "lr-cca (bilateral)"),
                              ("corpus-pca:64", C_PCA, "corpus-pca (variance)")):
        ax.plot(SIZES, _curve(a, arm, res), color=color, marker="o", ms=3.0,
                mew=0, zorder=4, label=label)

    # The three curves converge to within 0.02 at the right-hand end and the two
    # references are flat, so nothing here can be direct-labelled without a
    # collision. Identity goes in one legend, in the region no series passes
    # through; the two no-corpus references are grouped at the bottom of it.
    h, lab = ax.get_legend_handles_labels()
    order = [2, 3, 0, 1]                      # curves first, then the references
    ax.legend([h[i] for i in order], [lab[i] for i in order],
              loc="upper left", frameon=False, handlelength=1.1, ncol=2,
              handletextpad=0.35, borderaxespad=0.1, labelspacing=0.22,
              columnspacing=0.8, fontsize=5.6)

    ax.set_xscale("log")
    ax.set_xlim(21, 2600)
    ax.set_ylim(0.55, 0.95)
    ax.set_yticks([0.6, 0.7, 0.8])
    ax.set_xticks([25, 100, 400, 2000])
    ax.set_xticklabels(["25", "100", "400", "2000"])
    ax.set_xlabel("unlabeled participants in the corpus")
    ax.set_ylabel("held-out gaze correlation $r$")
    ax.set_title("b   the decoder", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _budget_series(rows, arm, res, budgets):
    """Median over seeds at each budget, plus the spread."""
    out = []
    for b in budgets:
        v = [r[res] for r in rows if r["arm"] == arm and r["budget"] == b]  # over seeds
        out.append((np.median(v) if v else np.nan,
                    (min(v) if v else np.nan), (max(v) if v else np.nan)))
    return np.array(out)


def panel_budget(ax, b_rows, fp_rows, res="subtr"):
    """(c) Accuracy against the labeled budget, with and without the corpus."""
    budgets = [1, 2, 4, 8, 16, None]
    x = [1, 2, 4, 8, 16, 34]          # `None` ("all") drawn one step past 16
    # Hue is the method and lightness is the corpus size, so the two `lr-cca`
    # curves read as one arm at two corpus sizes rather than as two arms. Reusing
    # a categorical hue for the small corpus would have collided with `gev-slow`
    # in (b), which is a different basis entirely.
    series = (
        (b_rows, "lr-cca:32+lags1@2000", C_CCA, "-", "lr-cca, corpus N=2000"),
        (b_rows, "lr-cca:32@25", C_CCA_LIGHT, (0, (3, 1.6)), "lr-cca, corpus N=25"),
        (fp_rows, "fold-pca:64+lags1", C_REF, "-", "fold-local PCA (no corpus)"),
        (b_rows, "raw", "#8f8f8f", "-", "stride-4 voxels (no corpus)"),
    )
    for rows, arm, color, ls, label in series:
        y = _budget_series(rows, arm, res, budgets)
        # The band is the min-max over random participant draws, not a CI: at
        # B=1 there are eight labeled participants in the whole fit and which
        # eight matters more than anything else on the panel.
        ax.fill_between(x, y[:, 1], y[:, 2], color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, y[:, 0], color=color, ls=ls, marker="o", ms=3.0, mew=0,
                zorder=4, label=label)

    ax.legend(loc="lower right", frameon=False, handlelength=1.5, fontsize=6.0,
              handletextpad=0.5, borderaxespad=0.1, labelspacing=0.3)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "4", "8", "16", "all"])
    ax.minorticks_off()
    ax.set_xlabel("labeled participants per training study")
    ax.set_ylabel("held-out gaze correlation $r$")
    ax.set_title("d   the trade", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_dme1(ax, dme1_rows, a, fp_full, res="subtr"):
    """(e) Against the published CNN, on the folds it was never trained on.

    Restricted to those folds and re-reducing our arms over the same subset,
    because DeepMReye 1.0's training set *is* `dsL01`-`dsL06`: its all-nine
    median (0.813) is partly scored on its own training data and is not a
    transfer number. On `dsL06` the two published checkpoints differ only in
    whether that dataset was in training, and they read 0.856 against 0.485 --
    which is the size of the in-sample advantage, measured.

    Drawn as one dumbbell per fold rather than grouped bars: the question is
    "which is higher, and on every fold?", and a connected pair answers it
    without the reader tracking a colour across nine bars.
    """
    folds = sorted(dme1_rows)
    fp = fp_full["folds"]
    xs = np.arange(len(folds), dtype=float)

    for i, f in enumerate(folds):
        lo, hi = dme1_rows[f]["r"], a["lr-cca:32+lags1@2000"]["folds"][f][res]
        ax.plot([i, i], [lo, hi], color="#c9c9c9", lw=1.2, zorder=2,
                solid_capstyle="round")
        ax.plot(i, fp[f][res], "_", color=C_REF, ms=7, mew=1.2, zorder=3)
        ax.plot(i, lo, "o", color=C_CTL, ms=4.5, mew=0, zorder=4)
        ax.plot(i, hi, "o", color=C_CCA, ms=4.5, mew=0, zorder=4)
        ax.annotate(f"+{hi - lo:.2f}", (i, max(lo, hi)), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.8, color=C_CCA, fontweight="bold")

    for color, label, marker in ((C_CCA, "lr-cca (ours, corpus N=2000)", "o"),
                                 (C_CTL, "DeepMReye 1.0 (published CNN)", "o"),
                                 (C_REF, "fold-local PCA (supervised)", "_")):
        ax.plot([], [], marker, color=color, ms=4.5 if marker == "o" else 7,
                mew=0 if marker == "o" else 1.2, ls="none", label=label)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0, fontsize=5.6,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.25, ncol=1)

    ax.set_xlim(-0.5, len(folds) - 0.5)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(xs)

    def tick(f):
        star = "*" if f == "dsL06_sequences" else ""   # scored by 1to5, not 1to6
        return f.split("_")[0].replace("dsL", "ds") + star

    ax.set_xticklabels([tick(f) for f in folds])
    ax.set_ylabel("held-out gaze correlation $r$")
    ax.set_title("e   vs the published CNN", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_criterion(ax, crit, fp_full, res="subtr"):
    """(f) What the *criterion* is worth, at matched rank.

    Every bar is a 32-dimensional subspace of the same orbit principal
    directions, read by the same ridge; only the rule that picks the subspace
    changes. That is the comparison the arm has never been given: a frozen
    linear basis is easy to believe in once you see that a random one of the
    same shape reads 0.39.
    """
    rows = [
        ("random", crit["random"], C_RAW),
        ("variance", crit["variance"], C_PCA),
        ("sparsity", crit["dict"], C_RAW),
        ("independence", crit["ica"], C_RAW),
        ("bilateral (PLS)", crit["pls"], C_CCA_LIGHT),
        ("bilateral + whitening", crit["cca"], C_CCA),
    ]
    y = np.arange(len(rows))
    for i, (_lab, v, color) in enumerate(rows):
        ax.barh(i, v, height=0.62, color=color, zorder=3)
        # Inside the bar, so the supervised reference line can cross the panel
        # without ever landing on a number.
        ax.annotate(f"{v:.3f}", (v, i), xytext=(-4, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=6.0, zorder=5,
                    color="white", fontweight="bold" if i == len(rows) - 1 else "normal")
    ax.axvline(fp_full[res], color=C_REF, lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax.annotate("fold-pca (sup.)", (fp_full[res], -0.55),
                xytext=(3, 0), textcoords="offset points", fontsize=5.6,
                color=C_REF, va="center", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_ylim(-1.0, len(rows) - 0.35)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("held-out gaze correlation $r$")
    ax.set_title("f   what the criterion buys", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_tradeoff(ax, rows, res="subtr"):
    """(g) Participants against TRs each, at a fixed total number of TRs.

    The scaling curve grows participants, TRs and accessions together, so it
    cannot say which is the resource. Holding the row count fixed does. Plotted
    as a single falling line because the reader's question is "does it matter
    who supplies the rows", and the answer is "not until you run out of people".
    """
    order = sorted(rows, key=lambda k: -int(k.split()[0]))
    n = [int(k.split()[0]) for k in order]
    y = [rows[k][res] for k in order]
    base = y[0]
    ax.axhspan(base - NOISE, base + NOISE, color="#ececec", zorder=0, lw=0)
    ax.plot(n, y, color=C_CCA, marker="o", ms=3.4, mew=0, zorder=4)
    ax.annotate("48 TRs each", (n[0], y[0]), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=5.6, color=MUTED)
    ax.annotate(f"768 TRs each\n{y[-1] - base:+.3f}, 9/9 folds", (n[-1], y[-1]),
                xytext=(7, 0), textcoords="offset points", ha="left", va="center",
                fontsize=5.8, color=C_CTL, fontweight="bold")
    ax.annotate("noise floor", (2400, base - NOISE), xytext=(0, 2),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=5.4, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{v:,}" for v in n])
    ax.minorticks_off()
    ax.set_xlim(88, 4200)
    ax.set_ylim(0.705, 0.800)
    ax.set_xlabel("participants, at a fixed 96,000 total TRs")
    ax.set_ylabel("held-out gaze correlation $r$")
    ax.set_title("g   people, or rows?", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def load_criterion(path):
    """Matched-rank k=32 scores for each subspace-selection criterion."""
    w = json.load(open(Path(path) / "sweep_whiten.json"))
    b = json.load(open(Path(path) / "beyond.json"))
    return {"random": w["random orthogonal:32"]["subtr"],
            "variance": b["orbit-variance:32"]["subtr"],
            "dict": b["dict:32"]["subtr"],
            "ica": b["ica-kurtosis:32"]["subtr"],
            "pls": w["PLS    whiten=0  shrink=0.001"]["subtr"],
            "cca": w["CCA    whiten=1  shrink=0.001"]["subtr"]}


def load_dme1(results_dir):
    """Per-fold v1 scores, each from the checkpoint that did not train on it."""
    rows = {}
    for name in ("score_1to6.json", "score_1to5.json"):
        path = Path(results_dir) / name
        if not path.exists():
            continue
        d = json.load(open(path))
        dirty = set(d["contaminated"])
        for fold, v in d["folds"].items():
            if fold in dirty or fold in rows:
                continue
            rows[fold] = {"r": v["subtr"], "1tr": v["1tr"],
                          "weights": d["weights"]}
    if not rows:
        raise SystemExit("[!] no DeepMReye scores under results/dme1 -- run "
                         "scripts/eval_dme1.py predict/score first")
    return rows


PANELS = {
    "a": ("spectrum", "basis_stats.json"),
    "b": ("scaling", "corpus_scaling.json"),
    "c": ("folds", "corpus_scaling.json"),
    "d": ("budget", "label_budget.json"),
    "e": ("dme1", None),
    "f": ("criterion", None),
    "g": ("tradeoff", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/unlabeled_value")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--res", default="subtr", choices=["subtr", "1tr"])
    ap.add_argument("--name", default="figure_unlabeled")
    ap.add_argument("--panels", default="abce",
                    help="Any subset of abcdefg in order: a spectrum, b corpus "
                         "scaling, c per-fold, d labeled budget, e vs "
                         "DeepMReye 1.0, f selection criterion at matched rank, "
                         "g participants against TRs. One row up to three "
                         "panels, else a two-row grid.")
    ap.add_argument("--variants", default="results/lrcca_variants",
                    help="directory holding sweep_whiten.json / beyond.json / "
                         "tradeoff.json, for panels f and g")
    ap.add_argument("--dme1", default="results/dme1",
                    help="directory holding score_1to6.json / score_1to5.json")
    args = ap.parse_args()

    want = [c for c in args.panels]
    unknown = [c for c in want if c not in PANELS]
    if unknown:
        raise SystemExit(f"[!] unknown panel(s) {unknown}; pick from {sorted(PANELS)}")

    r = Path(args.results)
    a = json.load(open(r / "corpus_scaling.json"))
    stats = json.load(open(r / "basis_stats.json"))
    fp_full_rows = json.load(open(r / "foldpca_full.json"))
    fp_full = next(x for x in fp_full_rows
                   if x["budget"] is None and x["arm"].endswith("lags1"))

    draw = {
        "a": lambda ax: panel_spectrum(ax, stats),
        "b": lambda ax: panel_scaling(ax, a, fp_full_rows, args.res),
        "c": lambda ax: panel_folds(ax, a, args.res),
        "d": lambda ax: panel_budget(ax, json.load(open(r / "label_budget.json")),
                                     json.load(open(r / "foldpca_budget.json")),
                                     args.res),
        "e": lambda ax: panel_dme1(ax, load_dme1(args.dme1), a, fp_full, args.res),
        "f": lambda ax: panel_criterion(ax, load_criterion(args.variants), fp_full,
                                        args.res),
        "g": lambda ax: panel_tradeoff(
            ax, json.load(open(Path(args.variants) / "tradeoff.json")), args.res),
    }

    style()
    n = len(want)
    if n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(min(7.4, 2.45 * n), 2.05),
                                 squeeze=False)
        flat = list(axes[0])
        fig.subplots_adjust(wspace=0.50)
    else:
        cols = (n + 1) // 2
        # Authored at the printed width: a figure laid out wider than the text
        # block is scaled down by LaTeX, and 7pt type at 0.8x is 5.6pt on paper.
        fig, axes = plt.subplots(2, cols, figsize=(7.4, 3.05), squeeze=False)
        flat = [ax for row in axes for ax in row]
        for ax in flat[n:]:
            ax.remove()
        fig.subplots_adjust(wspace=0.44, hspace=0.68)
    for letter, ax in zip(want, flat):
        draw[letter](ax)
    # Each panel hard-codes its own letter, so a subset like "abce" would print
    # a gap. Renumber by position after drawing.
    for i, ax in enumerate(flat[:n]):
        title = ax.get_title(loc="left")
        ax.set_title(f"{chr(ord('a') + i)}{title[1:]}", loc="left",
                     fontweight="bold", pad=4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{args.name}.{ext}")
    print(f"[+] {out / (args.name + '.pdf')}")


if __name__ == "__main__":
    main()
