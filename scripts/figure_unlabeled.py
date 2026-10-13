"""The paper figure: the frozen basis against supervision, and a confidence
without labels. Four panels, one row, authored at the printed width; a-c share
one y axis, c and d have their own; every panel is gaze r, mean over datasets.

a. **More unlabelled data.** Leave-one-dataset-out accuracy against the number
   of unlabelled participants the basis is fitted on, from 100, against a
   supervised decoder with the *same* ridge readout on voxel block means (the
   better of 2x2x2 and 3x3x3). Every size below the full corpus is 20 random
   draws of that many participants: lines at their mean, bands (ours) over the
   99% confidence interval of that mean. Dashed: both again on each dataset's
   80% most reliable participants by orbit agreement -- the same participants
   for both.
b. **Fewer labelled datasets.** Against how many labelled datasets the readout
   trains on: below eight, the mean of 20 random draws of which datasets, with
   the 99% CI of that mean.
c. **Datasets or participants.** The same number of labelled participants
   (8, 16, 32) drawn from 1-8 datasets: mean of 10 draws, ours with the 99% CI
   of that mean.
d. **A confidence without labels.** Accuracy on each dataset's participants
   kept by orbit agreement, against ranking by true r (an oracle), the same
   agreement through a random basis, and random selection.

Panel b marks the lead over supervision at one and eight labelled datasets;
no other differences are printed on the panels, they are in the text.

What is deliberately not drawn, and why, is in ``docs/COSYNE_FRAMING.md``:
corpus PCA (one sentence of text), the within-dataset readout (its fair
counterpart is a supervised decoder fitted there too, which is level or
ahead), and the 1-TR curves (supervision's best there is within the noise floor).

    python scripts/figure_unlabeled.py --results results/unlabeled_value --out paper
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unlabeled.common import keep_most_confident  # noqa: E402

# Okabe-Ito, checked for adjacent-pair separation under deutan and tritan
# simulation; every series is also labelled, so identity never rests on colour.
C_OURS = "#0072B2"
C_OURS_LIGHT = "#6BAED6"
C_SUP = "#D55E00"
C_REF = "#444444"
C_RAW = "#999999"
INK = "#1a1a1a"
MUTED = "#666666"
DASHED = (0, (3, 1.6))

OURS = "lr-cca:32+lags1"
SAME_READOUT = ("voxels:2+lags1", "voxels:3+lags1")     # the ridge ours uses, on voxel block means
ALL_EIGHT = "other datasets, all eight"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": "#888888",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
    })


def _grid(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _by_participant(schemes, arm, res="subtr"):
    """``{"dataset/subject": r}`` for one arm under leave-one-dataset-out."""
    return {f"{r['dataset']}/{r['subject']}": r[res] for r in schemes
            if r["scheme"] == ALL_EIGHT and r["arm"] == arm and np.isfinite(r[res])}


MEDIAN, MEAN = 0, 1    # keep_most_confident returns [median, mean] over datasets


def _kept(per_participant, agreement, frac, stat=MEAN):
    """Mean over datasets of the median r of each dataset's ``frac`` most
    agreeing participants (``frac = 1``: all of them)."""
    people = [{"dataset": k.split("/")[0], "agreement": agreement[k], "r": v}
              for k, v in per_participant.items() if k in agreement and np.isfinite(v)]
    return keep_most_confident(people, "agreement", frac)[stat]


def _mean_ci(values, level=0.99):
    """Mean over random draws and its ``level`` t confidence interval."""
    from scipy.stats import t

    v = np.asarray(values, dtype=float)
    m, n = v.mean(axis=0), len(v)
    half = t.ppf(0.5 + level / 2, n - 1) * v.std(axis=0, ddof=1) / np.sqrt(n)
    return m, m - half, m + half


def panel_corpus(ax, scaling, schemes, reliable, draws, frac=0.8, min_n=100, level=0.99):
    """(a) Accuracy against unlabelled corpus size from ``min_n`` participants,
    ours and the same ridge on voxels, on all participants and on each
    dataset's ``frac`` most reliable. Below the full corpus every size is many
    random draws: lines at their mean, bands (ours) over the ``level``
    confidence interval of that mean. Returns the y range drawn."""
    full = max(int(k.split("@")[1]) for k in scaling if k.startswith(f"{OURS}@"))
    sizes = sorted({int(n) for row in draws.values() for n in row if int(n) >= min_n})
    seeds = sorted(draws, key=int)
    supervised = {arm: _by_participant(schemes, arm) for arm in SAME_READOUT}
    every = {k: 1.0 for k in next(iter(supervised.values()))}
    sup = max(_kept(supervised[arm], every, 1.0) for arm in SAME_READOUT)

    def theirs(agreement):
        return max(_kept(supervised[arm], agreement, frac) for arm in SAME_READOUT)

    per_draw = {
        "all": [[draws[s][str(n)][OURS]["subtr_mean"] for n in sizes] for s in seeds],
        "kept": [[_kept(draws[s][str(n)]["participants"], draws[s][str(n)]["agreement"], frac)
                  for n in sizes] for s in seeds],
        "sup_kept": [[theirs(draws[s][str(n)]["agreement"]) for n in sizes] for s in seeds]}
    at_full = {"all": scaling[f"{OURS}@{full}"]["subtr_mean"],
               "kept": _kept(reliable[f"{OURS}@{full}"], reliable[f"agreement@{full}"], frac),
               "sup_kept": theirs(reliable[f"agreement@{full}"])}
    xs = sizes + [full]
    curves = {}
    for key, rows in per_draw.items():
        m, lo, hi = _mean_ci(rows, level)
        curves[key] = (np.r_[m, at_full[key]], lo, hi)

    ax.plot([xs[0], full], [sup, sup], color=C_SUP, lw=1.3, zorder=2)
    for key, alpha in (("all", 0.25), ("kept", 0.15)):
        _m, lo, hi = curves[key]
        ax.fill_between(sizes, lo, hi, color=C_OURS, alpha=alpha, lw=0, zorder=1)
    for key, colour, ls, lw, z in (("sup_kept", C_SUP, DASHED, 1.1, 2),
                                   ("kept", C_OURS, DASHED, 1.3, 3),
                                   ("all", C_OURS, "-", 1.4, 4)):
        ax.plot(xs, curves[key][0], color=colour, lw=lw, ls=ls, zorder=z,
                **({"marker": "o", "ms": 3.0, "mew": 0} if key == "all" else {}))
    ours = curves["all"][0]
    above = np.nonzero(ours > sup)[0]
    if len(above) and above[0] > 0:
        i, logn = int(above[0]), np.log(xs)
        cross = float(np.exp(logn[i - 1] + (sup - ours[i - 1]) / (ours[i] - ours[i - 1])
                             * (logn[i] - logn[i - 1])))
        ax.plot([cross], [sup], "o", ms=4.5, mfc="white", mec=C_SUP, mew=1.1, zorder=5)
        ax.annotate(f"N ≈ {int(round(cross / 50) * 50)}", (cross, sup), xytext=(4, -4),
                    textcoords="offset points", ha="left", va="top", fontsize=5.8, color=C_SUP)
    ax.legend(handles=[Line2D([], [], color=C_OURS, marker="o", ms=3.0, mew=0, label="ours"),
                       Line2D([], [], color=C_SUP, label="supervised"),
                       Line2D([], [], color=MUTED, lw=1.2, ls=DASHED,
                              label=f"{int(frac * 100)}% most reliable"),
                       Patch(color=C_OURS, alpha=0.25, lw=0,
                             label=f"{int(level * 100)}% CI, {len(seeds)} draws")],
              loc="lower right", frameon=False, fontsize=5.6, handlelength=1.6, borderaxespad=0.2,
              labelspacing=0.2)
    ax.set_xscale("log")
    ax.set_xlim(xs[0] * 0.8, full * 1.3)
    ticks = [t for t in (100, 400) if t in sizes] + [full]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("unlabelled participants")
    ax.set_title("a   unlabelled data", loc="left", fontweight="bold", pad=4)
    _grid(ax)
    lows = [sup, *[c[1].min() for c in curves.values()]]
    highs = [*[c[0].max() for c in curves.values()], *[c[2].max() for c in curves.values()]]
    return min(lows), max(highs)


def panel_labelled(ax, ours_rows, sup_rows, level=0.99, min_m=1):
    """(b) Mean over datasets against the number of labelled datasets. Below
    all eight, many draws of which datasets: lines at their mean, bands over
    the ``level`` confidence interval of that mean. Returns the y range."""
    arm = next(r["arm"] for r in ours_rows if r["arm"].startswith(f"{OURS}@"))
    ms = sorted({r["datasets"] for r in ours_rows if r["datasets"] >= min_m})
    draws = {"ours": [[r["subtr_mean"] for r in ours_rows if r["arm"] == arm and r["datasets"] == m] for m in ms],
             "sup": [[r["subtr_mean"] for r in sup_rows if r["arm"] == "voxels:3+lags1" and r["datasets"] == m]
                     for m in ms]}
    lows, highs, ends = [], [], {}
    # Every other supervised decoder, in grey. They were run on fewer draws than
    # the same ridge (2-4 against 20), and the draw matters more than the decoder,
    # so each is drawn as the same ridge's curve plus its paired difference from
    # it on the draws it shares: comparable to the coloured lines, not to its
    # own raw mean.
    def by_seed(name, m):
        return {r["seed"]: r["subtr_mean"] for r in sup_rows if r["arm"] == name and r["datasets"] == m}

    others, drawn = sorted({r["arm"] for r in sup_rows} - {"voxels:3+lags1"}), 0
    for name in others:
        ys = []
        for m in ms:
            mine, ridge = by_seed(name, m), by_seed("voxels:3+lags1", m)
            shared = sorted(set(mine) & set(ridge))
            if not shared:
                break
            ys.append(np.mean(list(ridge.values())) + np.mean([mine[k] - ridge[k] for k in shared]))
        if len(ys) < len(ms):
            continue
        drawn += 1
        ax.plot(ms, ys, color=C_RAW, lw=0.8, alpha=0.8, marker="o", ms=1.8, mew=0, zorder=2)
        lows += [min(ys)]
    for name, colour, alpha, z in (("sup", C_SUP, 0.22, 3), ("ours", C_OURS, 0.25, 4)):
        means = [float(np.mean(d)) for d in draws[name]]
        ends[name] = (means[0], means[-1])
        sampled = [(m, d) for m, d in zip(ms, draws[name]) if len(d) > 1]
        _m, lo, hi = (np.array(v) for v in zip(*[_mean_ci(d, level) for _, d in sampled]))
        ax.fill_between([m for m, _ in sampled], lo, hi, color=colour, alpha=alpha, lw=0, zorder=1)
        ax.plot(ms, means, color=colour, marker="o", ms=3.0, mew=0, zorder=z)
        lows += list(lo) + means
        highs += list(hi) + means
    if drawn:
        ax.legend(handles=[Line2D([], [], color=C_RAW, lw=0.8, marker="o", ms=1.8, mew=0,
                                  label=f"other supervised ({drawn})")],
                  loc="lower right", frameon=False, fontsize=5.6, handlelength=1.4, borderaxespad=0.2)
    # The lead at both ends: a bracket beside each end point, its label in the free
    # space above the first (nothing is up there) and below the last.
    for i, (m, side) in enumerate(((ms[0], -1), (ms[-1], 1))):
        top, bottom = ends["ours"][i], ends["sup"][i]
        x = m * 2.0 ** (0.14 * side)
        ax.annotate("", xy=(x, top), xytext=(x, bottom),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.18,widthB=0.18", lw=0.7, color=C_OURS,
                                    shrinkA=0, shrinkB=0))
        ax.annotate(f"+{top - bottom:.2f}", (x, top),
                    xytext=(-3, 14) if side < 0 else (0, 3), textcoords="offset points",
                    ha="left" if side < 0 else "center", va="bottom",
                    fontsize=6.0, color=C_OURS,
                    fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.minorticks_off()
    ax.set_xlim(ms[0] * 0.78, ms[-1] * 1.3)
    ax.set_xlabel("labelled datasets")
    ax.set_title("b   labelled datasets", loc="left", fontweight="bold", pad=4)
    _grid(ax)
    return min(lows), max(highs)


def panel_spread(ax, rows, totals=(8, 16, 32), level=0.99):
    """(c) A fixed number of labelled participants, drawn from 1..8 datasets:
    mean over datasets, averaged over draws; ours with the ``level`` CI of
    that average. Returns the y range."""
    totals = tuple(t for t in totals if any(r["total"] == t for r in rows))
    ms = sorted({r["datasets"] for r in rows})
    ours = next(r["arm"] for r in rows if r["arm"].startswith(f"{OURS}@"))
    shade = dict(zip(totals, (0.40, 0.70, 1.0)[-len(totals):]))
    dodge = dict(zip(totals, np.linspace(-0.08, 0.08, len(totals))))   # log2 units, so bars do not overlap
    lows, highs = [], []
    for arm, colour, z in (("voxels:3+lags1", C_SUP, 2), (ours, C_OURS, 3)):
        for T in totals:
            draws = [[r["subtr_mean"] for r in rows if r["arm"] == arm and r["total"] == T
                      and r["datasets"] == m] for m in ms]
            m_, lo, hi = (np.array(v) for v in zip(*[_mean_ci(d, level) for d in draws]))
            x = np.asarray(ms) * 2.0 ** dodge[T]
            ax.plot(x, m_, color=colour, alpha=shade[T], marker="o", ms=2.6, mew=0, lw=1.2, zorder=z)
            lows += list(m_)
            highs += list(m_)
            if arm == ours:
                ax.errorbar(x, m_, yerr=[m_ - lo, hi - m_], fmt="none", ecolor=colour, alpha=shade[T],
                            elinewidth=0.7, capsize=1.2, capthick=0.7, zorder=z)
                lows += list(lo)
                highs += list(hi)
    ax.legend(handles=[Line2D([], [], color=C_OURS, alpha=shade[T], marker="o", ms=2.6, mew=0, lw=1.2,
                              label=str(T)) for T in totals],
              title="labelled participants", title_fontsize=5.4, loc="lower right", ncol=len(totals),
              frameon=False, fontsize=5.4, handlelength=1.1, handletextpad=0.3, columnspacing=0.7,
              borderaxespad=0.1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.minorticks_off()
    ax.set_xlim(0.8, ms[-1] * 1.25)
    ax.set_xlabel("datasets they come from")
    ax.set_title("c   datasets or participants", loc="left", fontweight="bold", pad=4)
    _grid(ax)
    return min(lows), max(highs)


def panel_confidence(ax, conf, fracs=np.round(np.arange(1.0, 0.49, -0.05), 2), n_draws=200, stat=MEAN):
    """(d) Each dataset's most confident participants, kept by orbit agreement.
    Returns the y range drawn."""
    rows = conf["participants"]
    pct = 100 * fracs

    def curve(score):
        return np.array([keep_most_confident(rows, score, f)[stat] for f in fracs])

    rng, draws = np.random.default_rng(0), []
    for _ in range(n_draws):
        for q in rows:
            q["_draw"] = rng.random()
        draws.append(curve("_draw"))
    for q in rows:
        del q["_draw"]
    ours, oracle, chance = curve("agreement"), curve("r"), np.mean(draws, axis=0)
    for label, ys, colour, ls, lw, z in (
            ("orbit agreement", ours, C_OURS, "-", 1.6, 4),
            ("oracle: true $r$", oracle, C_REF, (0, (2.5, 1.5)), 1.0, 5),
            ("random basis", curve("random-basis agreement"), C_OURS_LIGHT, "-", 1.0, 3),
            ("random", chance, C_RAW, "-", 1.0, 2)):
        ax.plot(pct, ys, color=colour, ls=ls, lw=lw, zorder=z, label=label)
    i = int(np.argmin(np.abs(pct - 80)))
    ax.axvline(80, color="#cfcfcf", lw=0.6, ls=":", zorder=0)
    ax.plot([80], [ours[i]], "o", ms=3.6, color=C_OURS, mec="white", mew=0.6, zorder=6)
    ax.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.6, borderaxespad=0.1,
              labelspacing=0.2)
    ax.set_xlim(102, 48)
    ax.set_xticks([100, 90, 80, 70, 60, 50])
    ax.set_xlabel("participants kept (%)")
    ax.set_title("d   confidence without labels", loc="left", fontweight="bold", pad=4)
    _grid(ax)
    return min(chance.min(), ours.min()), max(oracle.max(), ours.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results/unlabeled_value")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--name", default="figure_cosyne")
    a = ap.parse_args()

    r = Path(a.results)
    load = lambda name: json.load(open(r / name))  # noqa: E731

    style()
    fig = plt.figure(figsize=(7.35, 2.0))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.24, 1, 0.24, 1], wspace=0.12)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    ax_c, ax_d = fig.add_subplot(gs[3]), fig.add_subplot(gs[5])

    def limits(ax, span, top_pad=0.005, step=50, bottom_pad=0.01):
        lo, hi = span
        ax.set_ylim(np.floor((lo - bottom_pad) * step + 1e-6) / step, np.ceil((hi + top_pad) * step - 1e-6) / step)

    ab = [panel_corpus(ax_a, load("corpus_scaling.json"), load("scheme_participants.json"),
                       load("reliable_participants.json"), load("scaling_draws.json")),
          panel_labelled(ax_b, load("label_datasets.json"), load("label_datasets_voxels.json"))]
    limits(ax_a, (min(s[0] for s in ab), max(s[1] for s in ab)), step=20, bottom_pad=0.005)
    ax_a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax_a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax_b.tick_params(labelleft=False)
    limits(ax_c, panel_spread(ax_c, load("label_spread.json")), top_pad=0.01, step=20)
    limits(ax_d, panel_confidence(ax_d, load("confidence.json")), top_pad=0.015, step=100)
    ax_a.set_ylabel("gaze $r$, mean over datasets")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{a.name}.{ext}")
    print(f"[+] {out / (a.name + '.pdf')}")


if __name__ == "__main__":
    main()
