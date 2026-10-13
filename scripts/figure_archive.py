"""A second figure, in draft: what the frozen decoder recovers from archived
fMRI that was never eye-tracked for release.

a. **Guided saccades and antisaccades, same runs.** ``ds000119``: decoded
   horizontal gaze around target onset, split by the target's side, for trials
   where the participant had to look at the target and trials where they had
   to look away from it.
b. **Trial by trial.** The change in decoded horizontal gaze against the
   target's position, every correct trial of ``ds000119`` and ``ds000121``.
c. **The confidence tracks the QA labels.** Orbit agreement of every
   unlabelled participant whose eye mask is covered, by whether its dataset
   passed QA. (An uncovered participant decodes to a flat trace whose orbits
   agree trivially; 52 such in failed datasets sit at 1.0.)

``ds000120`` decodes with the mirrored relation in every participant (a
left-right flip of display or image, unresolved) and is left out of a and b.

    python scripts/figure_archive.py --results paper/data/unlabeled_value --out paper
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from figure_unlabeled import C_OURS, C_RAW, C_SUP, MUTED, _grid, style  # noqa: E402

C_GUIDED, C_ANTI = "#009E73", "#CC79A7"


def panel_traces(ax, sac, dataset="ds000119"):
    """Decoded gaze towards the target's side: (right-target - left-target) / 2."""
    lags = np.asarray(sac["lags"])
    for kind, colour, label in (("guided", C_GUIDED, "look at the target"),
                                ("anti", C_ANTI, "look away from it")):
        keys = [f"{dataset}|{kind}|CORRECT|{side}" for side in ("left", "right")]
        if not all(k in sac["traces"] for k in keys):
            continue
        (ml, mr), (el, er) = ([np.asarray(sac[f][k]) for k in keys] for f in ("traces", "trace_sem"))
        m, e = (mr - ml) / 2, np.sqrt(el ** 2 + er ** 2) / 2
        n = sum(sac["trace_counts"][k] for k in keys)
        ax.fill_between(lags, m - e, m + e, color=colour, alpha=0.18, lw=0)
        ax.plot(lags, m, color=colour, lw=1.5, label=f"{label} ({n} trials)")
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axvspan(*sac["response"], color="#f2f2f2", zorder=-1)
    ax.legend(loc="lower left", frameon=False, fontsize=5.6, handlelength=1.4, labelspacing=0.15,
              borderaxespad=0.1)
    ax.set_xlim(lags[0], lags[-1])
    ax.set_xlabel("time from target onset (s)")
    ax.set_ylabel("decoded gaze towards target (z)")
    ax.set_title(f"a   saccades, {dataset}", loc="left", fontweight="bold", pad=4)
    _grid(ax)


def panel_trials(ax, sac):
    trials = [t for t in sac["trials"] if t["dataset"] in ("ds000119", "ds000121")]
    for kind, colour, label in (("guided", C_GUIDED, "look at target"), ("anti", C_ANTI, "look away")):
        pts = [t for t in trials if t["type"] == kind and t["outcome"] == "CORRECT"]
        x, y = np.array([t["target"] for t in pts]), np.array([t["shift"] for t in pts])
        jitter = np.random.default_rng(0).normal(0, 6, len(x))
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(x + jitter, y, "o", ms=2.0, color=colour, alpha=0.55, mew=0,
                label=f"{label}: r = {r:+.2f} ({len(pts)} trials)")
    errors = [t for t in trials if t["type"] == "anti" and t["outcome"] == "INCORRECT"]
    if errors:
        ax.plot([t["target"] for t in errors], [t["shift"] for t in errors], "x", ms=3.5, color=C_SUP,
                mew=0.9, label=f"look-away errors ({len(errors)})")
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=5.2, handlelength=0.8, labelspacing=0.15,
              borderaxespad=0.1)
    ax.set_xlabel("target position (screen px from centre)")
    ax.set_ylabel("change in decoded gaze (z)")
    ax.set_title("b   trial by trial", loc="left", fontweight="bold", pad=4)
    _grid(ax)


def panel_qa(ax, meta):
    groups = [("pass", "passed QA", C_OURS), ("fail", "failed QA", C_SUP),
              ("not labelled", "not QA-labelled", C_RAW)]
    bins = np.linspace(-0.2, 1.0, 25)
    for qa, label, colour in groups:
        v = np.array([m["agreement"] for m in meta
                      if not m["labelled"] and m["qa"] == qa and m.get("covered", True)])
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.3, color=colour,
                label=f"{label} (n = {len(v):,}), median {np.median(v):.2f}")
    ax.legend(loc="upper left", frameon=False, fontsize=5.2, handlelength=1.2, labelspacing=0.15,
              borderaxespad=0.1)
    ax.set_xlabel("orbit agreement (no labels)")
    ax.set_ylabel("density")
    ax.set_title("c   confidence and QA labels", loc="left", fontweight="bold", pad=4)
    ax.tick_params(axis="y", colors=MUTED)
    _grid(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="paper/data/unlabeled_value")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--name", default="figure_archive")
    a = ap.parse_args()
    r = Path(a.results)
    sac, meta = json.load(open(r / "saccades.json")), json.load(open(r / "corpus_meta.json"))
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.15))
    fig.subplots_adjust(wspace=0.45)
    panel_traces(axes[0], sac)
    panel_trials(axes[1], sac)
    panel_qa(axes[2], meta)
    for ext in ("pdf", "png"):
        fig.savefig(Path(a.out) / f"{a.name}.{ext}")
    print(f"[+] {Path(a.out) / (a.name + '.pdf')}")


if __name__ == "__main__":
    main()
