#!/usr/bin/env python3
"""Where do the 270 labeled participants sit inside the 1773 unlabeled ones?

Every unsupervised arm tried on this corpus has tied or lost to a basis fitted
on the labeled training fold alone, and the standing explanation is *domain
mismatch*: a corpus basis orders its components by variance in OpenNeuro scans
whose scanners and protocols differ from the gaze datasets. That explanation has
never been measured -- it was inferred from the probe numbers going the wrong
way, which is exactly the kind of story that survives because nobody checked it.

This script checks it, following the standard batch-effect protocol for
multi-site neuroimaging (subject-level embeddings; PCA/t-SNE to look; a domain
classifier to quantify):

- **descriptors** -- one vector per participant, no gaze labels anywhere.
  ``cov``: per-participant log-variance and Fisher-z correlation of the corpus-PCA
  coordinates. This is the *operative* descriptor -- second-order structure over
  the components is literally what ridge reads and what EA/CORAL rewrite.
  ``sd``: per-voxel temporal standard deviation over the mask. Anatomy and
  registration rather than dynamics. Deliberately a different view, so a finding
  that appears in both is not an artifact of either.

  Note the temporal *mean* is not an option: blocks are per-voxel z-scored, so a
  participant's mean image is flat noise (0.06 SD against 0.50 for the variance
  map). Same reason ``thumbnail.from_block`` uses the SD.

- **proxy A-distance** ``d_A = 2(1 - 2 eps)`` from a held-out domain classifier.
  0 means the two groups are indistinguishable, 2 means perfectly separable. The
  point of the metric here is that it is *directional evidence*: if labeled and
  unlabeled participants are trivially separable, "domain mismatch" is a
  measurement rather than a story; if they are not, the explanation is wrong and
  the corpus basis fails for some other reason.

  Folds are grouped by dataset wherever both sides have >= 2 datasets, so the
  classifier cannot win by memorising one acquisition. Between two single
  datasets that is impossible and the number is an upper bound -- flagged in the
  output rather than quietly reported.

- **clustering** -- k-means, scored by adjusted Rand index against dataset
  identity. High ARI means the embedding organises by acquisition, which is the
  batch effect stated as a clustering fact.

Writes ``media/visualizations/06_*`` and ``07_*``, and -- if the geometry
supports it -- ``results/domain_matched_subjects.json``, the unlabeled
participants nearest the labeled cloud. That file is the actionable output: if
mismatch is really the problem, a basis fitted on those should beat one fitted
on all 1773.

    python scripts/visualize_corpus_embedding.py --k 64
    python scripts/visualize_corpus_embedding.py --cache-only   # re-plot, no reads
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
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import corpus_mask, load_basis  # noqa: E402

CACHE = Path("results/corpus_embedding.npz")
OUT_DIR = Path("media/visualizations")

# Below this many participants a held-out domain classifier is fitting noise --
# dsL02 and dsL06 contribute 5 and 6 fully-covered participants, so their rows
# are reported but must not be read as measurements.
MIN_N_FOR_DA = 10


# ---------------------------------------------------------------- descriptors

def _slab_indices(n_trs, per_slab, n_slabs):
    """Evenly spaced contiguous slabs -- eye position and scanner state drift
    over a run, so one block from the middle samples one moment of it."""
    per = min(per_slab, n_trs)
    if n_trs <= per:
        return [(0, n_trs)]
    starts = np.linspace(0, n_trs - per, n_slabs).astype(int)
    return [(int(s), int(s + per)) for s in sorted(set(starts.tolist()))]


def describe(path, mask, comps, mean, k, per_slab=24, n_slabs=3, min_trs=32):
    """``(cov_descriptor, sd_descriptor)`` for one participant, or ``None``.

    Returns nothing that could encode gaze: only second-order structure of the
    voxels. ``labels`` is never opened.
    """
    with h5py.File(path, "r") as f:
        n_trs = int(f["eye_block"].shape[-1])
        if n_trs < min_trs:
            return None
        chunks = [f["eye_block"][..., a:b] for a, b in
                  _slab_indices(n_trs, per_slab, n_slabs)]
    block = np.concatenate(chunks, axis=-1)                      # [X, Y, Z, T]
    x = block[mask].T.astype(np.float64)                         # [T, D]
    if x.shape[0] < 8:
        return None
    # Partial coverage would make the SD descriptor a map of the crop rather
    # than of the participant, so require the mask to be genuinely filled.
    if (np.abs(x).sum(0) > 0).mean() < 0.98:
        return None

    coords = (x - mean) @ comps[:, :k]                           # [T, k]
    sd = coords.std(0)
    corr = np.corrcoef(coords.T)
    corr = np.nan_to_num(corr, nan=0.0)
    iu = np.triu_indices(k, 1)
    # Fisher z: correlations near +-1 are otherwise compressed, and it is the
    # differences between participants that this whole plot is about.
    cov_desc = np.concatenate([np.log(sd + 1e-8),
                               np.arctanh(np.clip(corr[iu], -0.999, 0.999))])
    return cov_desc.astype(np.float32), x.std(0).astype(np.float32)


def build(data_dir, k, args):
    mask, bases, _ = load_basis(args.basis)
    if mask.sum() != int(corpus_mask(data_dir).sum()):
        print("[!] basis mask differs from this corpus's mask", file=sys.stderr)
    comps = bases["corpus-pca"]["components"].astype(np.float64)
    mean = bases["corpus-pca"]["mean"].astype(np.float64)

    paths = sorted(Path(data_dir).glob("*/*.h5"))
    drop = set(getattr(args, "exclude_datasets", ()) or ())
    if drop:
        before = len(paths)
        paths = [p for p in paths if p.parent.name not in drop]
        print(f"[*] excluding {before - len(paths)} participants from "
              f"{', '.join(sorted(drop))}")
    cov, sd, names, dsets = [], [], [], []
    for i, p in enumerate(paths):
        if i % 100 == 0:
            print(f"  [{i:>5}/{len(paths)}] {p.parent.name}", flush=True)
        try:
            got = describe(p, mask, comps, mean, k,
                           per_slab=args.per_slab, n_slabs=args.n_slabs)
        except Exception as e:                                # noqa: BLE001
            print(f"  [!] {p}: {e}", file=sys.stderr)
            continue
        if got is None:
            continue
        cov.append(got[0])
        sd.append(got[1])
        names.append(p.stem)
        dsets.append(p.parent.name)

    out = dict(cov=np.stack(cov), sd=np.stack(sd),
               subject=np.array(names), dataset=np.array(dsets), k=np.array([k]))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"[+] cached {len(names)} participants -> {CACHE}")
    return out


# ------------------------------------------------------------------- metrics

def proxy_a_distance(x, y, groups, seed=0, n_splits=5):
    """``d_A = 2(1 - 2 eps)`` with ``eps`` the held-out balanced error.

    Grouped by dataset when possible: a classifier that separates two groups by
    recognising one acquisition has measured nothing about the domains.
    """
    x = StandardScaler().fit_transform(x)
    n_groups = min(len(np.unique(groups[y == c])) for c in (0, 1))
    grouped = n_groups >= 2
    splits = min(n_splits, n_groups) if grouped else n_splits
    if splits < 2:
        return np.nan, False
    cv = (GroupKFold(n_splits=splits) if grouped
          else StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed))
    errs = []
    for tr, te in cv.split(x, y, groups if grouped else None):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=seed).fit(x[tr], y[tr])
        pred = clf.predict(x[te])
        # Balanced, because labeled vs unlabeled is 270 against 1773 and plain
        # accuracy would read 0.87 for a classifier that says "unlabeled".
        acc = np.mean([np.mean(pred[y[te] == c] == c) for c in (0, 1)])
        errs.append(1.0 - acc)
    if not errs:
        return np.nan, grouped
    return float(2 * (1 - 2 * np.mean(errs))), grouped


def neighbour_mix(x, is_lab, dsets, n_neighbors=25):
    """Per labeled dataset, the share of its nearest neighbours that are
    unlabeled corpus participants.

    The chance level is the unlabeled share of everything. Well below it means
    the dataset sits in its own pocket -- which is what a basis fitted on the
    corpus cannot describe well.
    """
    from sklearn.neighbors import NearestNeighbors

    z = StandardScaler().fit_transform(x)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(z)
    _, idx = nn.kneighbors(z)
    idx = idx[:, 1:]
    share = (~is_lab[idx]).mean(1)
    return {d: float(share[(dsets == d) & is_lab].mean())
            for d in sorted(set(dsets[is_lab]))}


def probe_deltas(path, labeled, corpus="corpus-pca:64/ridge-cv",
                 fold="fold-pca:64/ridge-cv"):
    """Per fold, ``corpus-pca:64`` minus ``fold-pca:64`` in mean Pearson r.

    Negative means the frozen corpus basis lost on that fold. Correlating this
    against distance-from-the-corpus is the test of the standing explanation for
    why the unlabeled half does not pay: if mismatch is the mechanism, the
    distant folds are the ones that must lose.
    """
    try:
        res = json.loads(Path(path).read_text())
    except Exception as e:                                    # noqa: BLE001
        print(f"[!] no probe deltas ({e}); skipping that panel")
        return None
    out = {}
    for d in labeled:
        arms = res.get(d, {})
        if corpus not in arms or fold not in arms:
            continue

        def mean_r(arm):
            b = arms[arm]["by_subject"]
            return 0.5 * (b["pearson_r_x"] + b["pearson_r_y"])

        out[d] = mean_r(corpus) - mean_r(fold)
    return out or None


def nearest_unlabeled(x, is_lab, n_keep):
    """Unlabeled participants closest to the labeled centroid, by cosine
    similarity in the standardised descriptor space."""
    z = StandardScaler().fit_transform(x)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    centre = z[is_lab].mean(0)
    centre /= np.linalg.norm(centre) + 1e-9
    sim = z @ centre
    order = np.argsort(-np.where(is_lab, -np.inf, sim))
    return order[:n_keep], sim


# --------------------------------------------------------------------- plots

def _scatter(ax, emb, is_lab, dsets, palette, title):
    ax.scatter(emb[~is_lab, 0], emb[~is_lab, 1], s=6, c="0.78",
               alpha=0.55, linewidths=0, rasterized=True, label="unlabeled")
    for d in sorted(set(dsets[is_lab])):
        m = (dsets == d) & is_lab
        ax.scatter(emb[m, 0], emb[m, 1], s=16, color=palette[d],
                   alpha=0.9, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_embedding(data, is_lab, dsets, palette, embeds, mix, dmat, groups,
                     counts, deltas=None, d_cov=np.nan, d_sd=np.nan):
    fig = plt.figure(figsize=(16.5, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.22)

    for j, (name, title) in enumerate([
            ("cov_tsne", "t-SNE - component covariance (dynamics)"),
            ("sd_tsne", "t-SNE - per-voxel temporal SD (anatomy)")]):
        _scatter(fig.add_subplot(gs[0, j]), embeds[name], is_lab, dsets,
                 palette, title)

    # The panel the whole exercise exists to produce: if "domain mismatch"
    # explains why a corpus basis loses, distance from the corpus has to predict
    # where it loses. Plotting it is the only way to see that it does not.
    ax = fig.add_subplot(gs[0, 2])
    if deltas:
        ks = [d for d in deltas if d in mix]
        xs = [dmat[groups.index(d), groups.index("unlabeled")] for d in ks]
        ys = [deltas[d] for d in ks]
        ax.axhline(0, color="0.6", lw=1)
        for d, xv, yv in zip(ks, xs, ys):
            ax.scatter(xv, yv, s=90 if counts[d] >= MIN_N_FOR_DA else 35,
                       color=palette[d],
                       edgecolors="k" if counts[d] >= MIN_N_FOR_DA else "none",
                       linewidths=0.8, zorder=3)
            ax.annotate(f"{d.split('_')[0]} (n={counts[d]})", (xv, yv),
                        textcoords="offset points", xytext=(6, 5), fontsize=7)
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        ax.set_xlabel("proxy A-distance from the unlabeled corpus", fontsize=8)
        ax.set_ylabel("corpus-pca:64 $-$ fold-pca:64  (mean r)", fontsize=8)
        ax.set_title(f"Does distance predict the loss?\nSpearman "
                     f"$\\rho$={rho:+.2f}, p={p:.2f}  (n={len(ks)}) - no",
                     fontsize=10)
    else:
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ks = list(mix)
    chance = float((~is_lab).mean())
    ax.barh(range(len(ks)), [mix[d] for d in ks],
            color=[palette[d] for d in ks])
    ax.axvline(chance, color="k", ls="--", lw=1.2)
    ax.text(chance, len(ks) - 0.35, f" chance {chance:.2f}", fontsize=8, va="top")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"{d.replace('_', chr(10))}\n(n={counts[d]})"
                        for d in ks], fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of 25 nearest neighbours that are unlabeled", fontsize=8)
    ax.set_title("How embedded in the corpus is each labeled set?\n"
                 "at or above chance = indistinguishable from the corpus",
                 fontsize=10)

    ax = fig.add_subplot(gs[1, 1:])
    im = ax.imshow(dmat, cmap="magma", vmin=0, vmax=2)
    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(groups)))
    lab = [g.replace("_", " ") for g in groups]
    ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(lab, fontsize=7)
    weak = {i for i, g in enumerate(groups)
            if counts.get(g, 10 ** 9) < MIN_N_FOR_DA}
    for a in range(len(groups)):
        for b in range(len(groups)):
            if not np.isfinite(dmat[a, b]):
                continue
            low = a in weak or b in weak
            ax.text(b, a, f"{dmat[a, b]:.2f}" + ("*" if low else ""),
                    ha="center", va="center", fontsize=7,
                    style="italic" if low else "normal",
                    color="w" if dmat[a, b] < 1.3 else "k")
    ax.set_title("Proxy A-distance between groups  (0 = indistinguishable, "
                 f"2 = perfectly separable)\n* = a side has < {MIN_N_FOR_DA} "
                 "participants; not a measurement", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    handles = [Line2D([], [], marker="o", ls="", color="0.78", label="unlabeled")]
    handles += [Line2D([], [], marker="o", ls="", color=palette[d],
                       label=f"{d} (n={counts[d]})")
                for d in sorted(set(dsets[is_lab]))]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Labeled gaze participants inside the unlabeled corpus "
        f"({int((~is_lab).sum())} unlabeled, {int(is_lab.sum())} labeled)\n"
        f"separable by dynamics ($d_A$={d_cov:.2f}) but not by anatomy "
        f"($d_A$={d_sd:.2f}) - and neither predicts where the corpus basis loses",
        fontsize=13)
    out = OUT_DIR / "06_corpus_embedding.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out}")


def figure_clusters(labels, is_lab, dsets, palette, ari, embeds, sim):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    n_c = labels.max() + 1
    ax = axes[0]
    ax.scatter(embeds["cov_tsne"][:, 0], embeds["cov_tsne"][:, 1], s=6,
               c=labels, cmap="tab20", alpha=0.7, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"k-means, k={n_c}\nARI vs dataset identity = {ari:.3f}",
                 fontsize=10)

    ax = axes[1]
    groups = sorted(set(dsets[is_lab]))
    bottom = np.zeros(n_c)
    frac_unlab = np.array([np.mean(~is_lab[labels == c]) for c in range(n_c)])
    for d in groups:
        share = np.array([np.mean(dsets[labels == c] == d) for c in range(n_c)])
        ax.bar(range(n_c), share, bottom=bottom, color=palette[d], label=d)
        bottom += share
    ax.bar(range(n_c), frac_unlab, bottom=bottom, color="0.78", label="unlabeled")
    ax.set_xlabel("cluster")
    ax.set_ylabel("composition")
    ax.set_title("Do labeled sets get their own clusters?", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="lower right")

    ax = axes[2]
    ax.hist(sim[~is_lab], bins=60, color="0.7", label="unlabeled", density=True)
    ax.hist(sim[is_lab], bins=60, color="crimson", alpha=0.65,
            label="labeled", density=True)
    ax.set_xlabel("cosine similarity to labeled centroid")
    ax.set_ylabel("density")
    ax.set_title("Overlap of the two populations", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "07_corpus_clusters.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out}")


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/corpus_basis.npz")
    p.add_argument("--k", type=int, default=64,
                   help="corpus-PCA components (64 is the probe's operating point)")
    p.add_argument("--per-slab", type=int, default=24)
    p.add_argument("--n-slabs", type=int, default=3)
    p.add_argument("--n-clusters", type=int, default=12)
    p.add_argument("--n-matched", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-datasets", nargs="*", default=(),
                   help="Datasets to leave out of the embedding -- a dataset "
                        "still being extracted would otherwise contribute a "
                        "handful of participants and drift between runs.")
    p.add_argument("--cache-only", action="store_true",
                   help="re-plot from results/corpus_embedding.npz")
    p.add_argument("--probe-json", default="results/probe_k64_unlab.json",
                   help="per-fold probe results, to test whether distance from "
                        "the corpus predicts where a corpus basis loses")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cache_only:
        if not CACHE.exists():
            sys.exit(f"no cache at {CACHE}; run without --cache-only first")
        data = dict(np.load(CACHE, allow_pickle=False))
        print(f"[*] cache {CACHE}: {len(data['subject'])} participants")
    else:
        data_dir = resolve(args.data_dir, download=False, quiet=True)
        print(f"[*] data {data_dir}")
        data = build(data_dir, args.k, args)

    dsets = data["dataset"]
    is_lab = np.array([d.startswith("dsL") for d in dsets])
    labeled = sorted(set(dsets[is_lab]))
    palette = {d: c for d, c in zip(labeled, plt.cm.tab10.colors)}
    counts = {d: int((dsets == d).sum()) for d in labeled}
    counts["unlabeled"] = int((~is_lab).sum())
    print(f"[*] {int(is_lab.sum())} labeled / {int((~is_lab).sum())} unlabeled, "
          f"{len(set(dsets))} datasets")
    print("[*] fully-covered participants per labeled dataset: "
          + ", ".join(f"{d.split('_')[0]}={counts[d]}" for d in labeled))
    thin = [d for d in labeled if counts[d] < MIN_N_FOR_DA]
    if thin:
        print(f"[!] {', '.join(thin)} have < {MIN_N_FOR_DA} participants; "
              "their proxy A-distances are not measurements")

    # --- embeddings -------------------------------------------------------
    embeds = {}
    for name, raw in (("cov", data["cov"]), ("sd", data["sd"])):
        z = StandardScaler().fit_transform(raw.astype(np.float64))
        pcs = PCA(n_components=min(50, z.shape[1]),
                  random_state=args.seed).fit_transform(z)
        embeds[f"{name}_pca"] = pcs[:, :2]
        embeds[f"{name}_tsne"] = TSNE(
            n_components=2, init="pca", perplexity=30,
            random_state=args.seed).fit_transform(pcs)

    # --- proxy A-distance -------------------------------------------------
    groups = labeled + ["unlabeled"]
    dmat = np.full((len(groups), len(groups)), np.nan)
    ungrouped = []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            ma = (dsets == groups[a]) if groups[a] != "unlabeled" else ~is_lab
            mb = (dsets == groups[b]) if groups[b] != "unlabeled" else ~is_lab
            sel = ma | mb
            d, grouped = proxy_a_distance(
                data["cov"][sel], mb[sel].astype(int), dsets[sel], args.seed)
            dmat[a, b] = dmat[b, a] = d
            if not grouped and np.isfinite(d):
                ungrouped.append((groups[a], groups[b]))
    np.fill_diagonal(dmat, 0.0)

    d_all, grouped_all = proxy_a_distance(
        data["cov"], is_lab.astype(int), dsets, args.seed)
    d_sd, _ = proxy_a_distance(data["sd"], is_lab.astype(int), dsets, args.seed)

    # --- clustering -------------------------------------------------------
    z = StandardScaler().fit_transform(data["cov"].astype(np.float64))
    pcs = PCA(n_components=50, random_state=args.seed).fit_transform(z)
    km = KMeans(n_clusters=args.n_clusters, n_init=10,
                random_state=args.seed).fit(pcs)
    ari = adjusted_rand_score(dsets, km.labels_)
    ari_lab = adjusted_rand_score(is_lab.astype(int), km.labels_)

    mix = neighbour_mix(data["cov"], is_lab, dsets)
    keep, sim = nearest_unlabeled(data["cov"], is_lab, args.n_matched)

    # --- does distance predict the loss? ----------------------------------
    deltas = probe_deltas(args.probe_json, labeled)

    # --- report -----------------------------------------------------------
    print("\n=== proxy A-distance: labeled vs unlabeled ===")
    print(f"  covariance descriptor : {d_all:.3f}"
          f"{'' if grouped_all else '   (ungrouped, upper bound)'}")
    print(f"  temporal-SD descriptor: {d_sd:.3f}")
    print("  0 = indistinguishable, 2 = perfectly separable")

    print("\n=== nearest-neighbour mix (chance = "
          f"{(~is_lab).mean():.3f}) ===")
    for d, v in sorted(mix.items(), key=lambda kv: kv[1]):
        print(f"  {d:<26} {v:.3f}")

    print(f"\n=== clustering (k={args.n_clusters}) ===")
    print(f"  ARI vs dataset identity   : {ari:.3f}")
    print(f"  ARI vs labeled/unlabeled  : {ari_lab:.3f}")
    pure = sum(1 for c in range(args.n_clusters)
               if is_lab[km.labels_ == c].mean() > 0.9)
    print(f"  clusters >90% labeled     : {pure}/{args.n_clusters}")

    print("\n=== pairwise proxy A-distance ===")
    print(" " * 26 + " ".join(f"{g[:9]:>10}" for g in groups))
    for a, g in enumerate(groups):
        print(f"  {g:<24}" + " ".join(
            f"{dmat[a, b]:>10.2f}" if np.isfinite(dmat[a, b]) else f"{'-':>10}"
            for b in range(len(groups))))
    if ungrouped:
        print(f"  [!] {len(ungrouped)} pairs had <2 datasets per side; those "
              "cells are upper bounds")

    if deltas:
        from scipy.stats import spearmanr

        ks = [d for d in deltas if d in mix]
        xs = [dmat[groups.index(d), groups.index("unlabeled")] for d in ks]
        print("\n=== does distance from the corpus predict the loss? ===")
        print(f"  {'fold':<24}{'n':>5}{'d_A':>7}{'nn-mix':>9}"
              f"{'corpus - fold':>15}")
        for d, xv in sorted(zip(ks, xs), key=lambda t: t[1]):
            print(f"  {d:<24}{counts[d]:>5}{xv:>7.2f}{mix[d]:>9.3f}"
                  f"{deltas[d]:>+15.3f}")
        rho, p = spearmanr(xs, [deltas[d] for d in ks])
        print(f"  spearman(d_A, delta) = {rho:+.3f}  p={p:.3f}  (n={len(ks)})")
        rho2, p2 = spearmanr([mix[d] for d in ks], [deltas[d] for d in ks])
        print(f"  spearman(nn-mix, delta) = {rho2:+.3f}  p={p2:.3f}")

    # --- actionable output ------------------------------------------------
    matched = {"n": int(len(keep)), "descriptor": "cov", "k": int(args.k),
               "subjects": [{"dataset": str(dsets[i]),
                             "subject": str(data["subject"][i]),
                             "similarity": float(sim[i])} for i in keep]}
    out = Path("results/domain_matched_subjects.json")
    out.write_text(json.dumps(matched, indent=1))
    print(f"\n[+] {len(keep)} domain-matched unlabeled participants -> {out}")

    figure_embedding(data, is_lab, dsets, palette, embeds, mix, dmat, groups,
                     counts, deltas, d_all, d_sd)
    figure_clusters(km.labels_, is_lab, dsets, palette, ari, embeds, sim)


if __name__ == "__main__":
    main()
