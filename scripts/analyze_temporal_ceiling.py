#!/usr/bin/env python3
"""Decodability is set by how fast the gaze moves, not by the decoder.

Two folds on this corpus look like model failures. ``dsL03_pursuit`` decodes at
r ~ 0.20 under every feature source and readout tried; ``dsL06_sequences``
decodes its horizontal axis at 0.95 and its vertical at 0.34 (and the *published*
DeepMReye 1.0 CNN gets -0.05 there on its own held-out fold, so it is not this
pipeline). Both were chased as representation or transfer problems. Neither is.

Across the 12 (dataset, axis) cells of the labeled corpus, the lag-1
autocorrelation of the gaze trace predicts the decoded correlation at
**Pearson r ~ 0.98**. Gaze that changes faster than the acquisition samples it
cannot be recovered, and every apparent failure here sits at the bottom of that
range.

**The load-bearing evidence is ``dsL06``'s two axes**, not the overall trend.
A between-dataset correlation confounds everything that differs between
acquisitions -- TR, scanner, subjects, paradigm, registration. dsL06 dissociates
*within the same scans*: identical TR, subjects, preprocessing and model, one
axis at lag-1 0.76 decoding at 0.95 and the other at 0.25 decoding at 0.34. It is
the only dataset whose axes differ (ratio 0.33; every other sits at 0.98-1.27),
and it is what turns a correlation into a mechanism.

Report the Spearman alongside the Pearson. With three low cells against nine
high ones the Pearson is flattered by the gap between the clusters, and the two
axes of a dataset are not independent samples -- so the effective n is well
below 12.

    python scripts/analyze_temporal_ceiling.py
    python scripts/analyze_temporal_ceiling.py --probe results/probe_k64_unlab.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve  # noqa: E402

OUT = Path("media/visualizations/08_temporal_ceiling.png")
AXES = ("x", "y")


def gaze_dynamics(path, min_valid=20):
    """``(lag1, sd)`` per gaze axis for one participant, NaNs dropped.

    Sub-TR samples are averaged first so the series is at acquisition
    resolution -- the whole question is what the acquisition can resolve.
    """
    with h5py.File(path, "r") as f:
        labels = f["labels"][...]
        tr = float(f.attrs.get("repetition_time", np.nan))
    y = np.nanmean(labels, axis=1)
    v = y[np.isfinite(y).all(1)]
    if len(v) < min_valid:
        return None
    lag, sd = [], []
    for j in range(2):
        d = v[:, j] - v[:, j].mean()
        lag.append(np.corrcoef(d[:-1], d[1:])[0, 1] if d.std() > 1e-9 else np.nan)
        sd.append(float(d.std()))
    return np.array(lag), np.array(sd), tr


def decoded(probe, dataset, arm):
    b = probe[dataset][arm]["by_subject"]
    return np.array([b["pearson_r_x"], b["pearson_r_y"]])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--probe", default="results/probe_k64_unlab.json")
    p.add_argument("--arm", default="fold-pca:64/ridge-cv")
    p.add_argument("--dme1", default="results/probe_dme1_dsL06_bin5.json",
                   help="published-CNN result, plotted as a second marker")
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    probe = json.loads(Path(args.probe).read_text())

    rows = []
    for ds in sorted(d.name for d in Path(data_dir).glob("dsL*")):
        if ds not in probe:
            continue
        got = [gaze_dynamics(f) for f in sorted((Path(data_dir) / ds).glob("*.h5"))]
        got = [g for g in got if g is not None]
        if not got:
            continue
        lag = np.nanmean([g[0] for g in got], axis=0)
        sd = np.nanmean([g[1] for g in got], axis=0)
        tr = np.nanmedian([g[2] for g in got])
        dec = decoded(probe, ds, args.arm)
        for j in range(2):
            rows.append({"dataset": ds, "axis": AXES[j], "n": len(got),
                         "lag1": float(lag[j]), "sd": float(sd[j]),
                         "tr": float(tr), "decoded_r": float(dec[j])})

    lag = np.array([r["lag1"] for r in rows])
    dec = np.array([r["decoded_r"] for r in rows])
    pr, pp = pearsonr(lag, dec)
    sr, sp = spearmanr(lag, dec)

    print(f"[*] data {data_dir}\n[*] arm {args.arm}\n")
    print(f"  {'cell':<12}{'n':>4}{'TR':>6}{'lag1':>8}{'gaze SD':>9}{'decoded r':>11}")
    print("  " + "-" * 50)
    for r in sorted(rows, key=lambda r: r["lag1"]):
        print(f"  {r['dataset'].split('_')[0] + '.' + r['axis']:<12}{r['n']:>4}"
              f"{r['tr']:>6.2f}{r['lag1']:>8.3f}{r['sd']:>9.2f}"
              f"{r['decoded_r']:>11.3f}")

    print(f"\n  Pearson  r   = {pr:+.3f}  p = {pp:.6f}   (n={len(rows)})")
    print(f"  Spearman rho = {sr:+.3f}  p = {sp:.6f}   <- the conservative one")

    # The within-scan dissociation, which is the actual argument.
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["axis"]] = r
    print("\n  within-dataset axis ratio (lag1 y / lag1 x):")
    for ds, d in sorted(by_ds.items(), key=lambda kv: kv[1]["y"]["lag1"] / kv[1]["x"]["lag1"]):
        ratio = d["y"]["lag1"] / d["x"]["lag1"]
        flag = "   <- the only dissociation" if ratio < 0.6 or ratio > 1.7 else ""
        print(f"    {ds:<24}{ratio:>6.2f}{flag}")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    palette = {ds: c for ds, c in zip(sorted(by_ds), plt.cm.tab10.colors)}
    for r in rows:
        ax.scatter(r["lag1"], r["decoded_r"], s=130,
                   color=palette[r["dataset"]],
                   marker="o" if r["axis"] == "x" else "^",
                   edgecolors="k", linewidths=0.8, zorder=3)
        ax.annotate(f"{r['dataset'].split('_')[0]}.{r['axis']}",
                    (r["lag1"], r["decoded_r"]), textcoords="offset points",
                    xytext=(7, -3), fontsize=7.5)

    fit = np.polyfit(lag, dec, 1)
    xs = np.linspace(lag.min() - 0.05, lag.max() + 0.05, 50)
    ax.plot(xs, np.polyval(fit, xs), color="0.5", ls="--", lw=1.2, zorder=1)

    if Path(args.dme1).exists():
        s = json.loads(Path(args.dme1).read_text())["summary"]
        d6 = by_ds.get("dsL06_sequences")
        if d6:
            for j, axis in enumerate(AXES):
                ax.scatter(d6[axis]["lag1"], s[f"pearson_r_{axis}"], s=130,
                           facecolors="none", edgecolors="crimson",
                           linewidths=1.8, zorder=4,
                           marker="o" if axis == "x" else "^")
            ax.annotate("published CNN\n(same two axes)",
                        (d6["y"]["lag1"], s["pearson_r_y"]),
                        textcoords="offset points", xytext=(14, -6),
                        fontsize=8, color="crimson")

    ax.set_xlabel("lag-1 autocorrelation of the gaze trace (at acquisition resolution)")
    ax.set_ylabel(f"decoded Pearson r  ({args.arm.split('/')[0]})")
    ax.set_title("Decodability is bounded by how fast the gaze moves\n"
                 f"Pearson r = {pr:+.2f}, Spearman $\\rho$ = {sr:+.2f} "
                 f"(n={len(rows)} dataset x axis cells)", fontsize=11)
    ax.axhline(0, color="0.85", lw=1, zorder=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="0.4", label="x axis"),
               Line2D([], [], marker="^", ls="", color="0.4", label="y axis"),
               Line2D([], [], marker="o", ls="", markerfacecolor="none",
                      markeredgecolor="crimson", label="DeepMReye 1.0")]
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[+] Saved {OUT}")

    out = Path("results/temporal_ceiling.json")
    out.write_text(json.dumps(
        {"arm": args.arm, "pearson_r": pr, "pearson_p": pp,
         "spearman_rho": sr, "spearman_p": sp, "cells": rows}, indent=1))
    print(f"[+] Wrote {out}")


if __name__ == "__main__":
    main()
