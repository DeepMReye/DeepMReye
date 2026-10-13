"""What is the unlabeled corpus worth? Every measurement behind FINDINGS.md and
the paper figure, written down once, all on the shipped protocol
(:func:`deepmreye.probe.lodo`: nine leave-one-dataset-out folds, 337
gaze-labelled participants, scored at sub-TR and 1-TR).

Each stage writes one JSON under ``--out`` and is skipped when that file
exists (``--force`` reruns); the slow ones resume a partial file instead. The
stages live in ``scripts/unlabeled/`` by topic:

corpus.py -- the bases themselves
  stats              basis_stats.json         canonical spectrum at each corpus size (no labels)
  scaling            corpus_scaling.json      both bases at every corpus size, with and without context
  scaling-draws      scaling_draws.json       ... on 20 random draws of each size below the full one (resumable)
  transfer           corpus_transfer.json     the same, readout trained across / pooled / within datasets
  gap                transfer_gap.json        the shipped arm under every split, ridge and MLP
  overlap            basis_overlap.json       why the two bases tie inside a study; rank sweep
  tune               tuning.json              every readout knob for both bases (resumable)

labels.py -- how much labelled data it needs
  budget             label_budget.json        labelled participants per training dataset
  foldpca            foldpca_full.json        PCA of the labelled scans, per fold (no corpus)
  foldpca-budget     foldpca_budget.json      ... and per labelled budget
  datasets           label_datasets.json      labelled *datasets* the readout may train on
  datasets-supervised label_datasets_supervised.json  ... with a PCA of those datasets' scans instead
  datasets-voxels    label_datasets_voxels.json  ... with voxels, no basis (resumable; corpus-free)
  spread             label_spread.json        datasets or participants: T labelled participants from 1..8 datasets (resumable)
  fewshot            fewshot.json             a new study's own 1..16 labelled participants
  fewshot-supervised fewshot_supervised.json  ... for decoders that saw no unlabelled scan (corpus-free)
  schemes            scheme_participants.json six decoding schemes, per participant (resumable)
  participants       lodo_participants.json   per-participant LODO r along the corpus ladder

confidence.py -- after the readout
  confidence         confidence.json          orbit agreement as a confidence; per volume; pseudo-labels
  reliable           reliable_participants.json  every arm on each dataset's most confident participants
  priors             viewing_priors.json      other viewers' decoded gaze as a prior

temporal.py -- after the readout, in time
  temporal           temporal_stages.json     interpolation and Gaussian-process stages on the lag-0 readout

readouts.py
  readout-grid       readout_grid.json        every base with every readout (resumable)
  against-dme1       dme1_table.json          the published CNN, ours and selected supervised decoders, every metric
  against-dme1-matched dme1_table_matched.json  ... with ours and the supervised decoders trained on the CNN's six datasets

data.py
  corpus-stats       corpus_stats.json        datasets screened, QA labels, acquisitions

    python scripts/analysis_unlabeled.py                       # everything not yet on disk
    python scripts/analysis_unlabeled.py --stages scaling,datasets --force
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import corpus_mask  # noqa: E402
from unlabeled import confidence, corpus, data, labels, readouts, temporal  # noqa: E402
from unlabeled.common import Context, corpus_sizes  # noqa: E402

# (stage, output, function, resumable). Run in this order: the cheap stages
# that share one projection first, then each that needs its own.
STAGES = [
    ("stats", "basis_stats.json", corpus.basis_stats, False),
    ("scaling", "corpus_scaling.json", corpus.scaling, False),
    ("scaling-draws", "scaling_draws.json", corpus.scaling_draws, True),
    ("budget", "label_budget.json", labels.budget, False),
    ("transfer", "corpus_transfer.json", corpus.transfer, False),
    ("gap", "transfer_gap.json", corpus.gap, False),
    ("datasets", "label_datasets.json", labels.datasets, True),
    ("overlap", "basis_overlap.json", corpus.overlap, False),
    ("corpus-stats", "corpus_stats.json", data.corpus_stats, False),
    ("confidence", "confidence.json", confidence.confidence, False),
    ("reliable", "reliable_participants.json", confidence.reliable, False),
    ("priors", "viewing_priors.json", confidence.priors, False),
    ("temporal", "temporal_stages.json", temporal.temporal, False),
    ("spread", "label_spread.json", labels.spread, True),
    ("fewshot", "fewshot.json", labels.fewshot, False),
    ("participants", "lodo_participants.json", labels.participants, False),
    ("schemes", "scheme_participants.json", labels.schemes, True),
    ("readout-grid", "readout_grid.json", readouts.readout_grid, True),
    ("against-dme1", "dme1_table.json", readouts.against_dme1, True),
    ("against-dme1-matched", "dme1_table_matched.json", readouts.against_dme1_matched, True),
    ("tune", "tuning.json", corpus.tune, True),
    ("datasets-supervised", "label_datasets_supervised.json", labels.datasets_supervised, False),
    ("datasets-voxels", "label_datasets_voxels.json", labels.datasets_voxels, True),
    ("fewshot-supervised", "fewshot_supervised.json", labels.fewshot_supervised, False),
    ("foldpca", "foldpca_full.json", labels.foldpca, False),
    ("foldpca-budget", "foldpca_budget.json", labels.foldpca_budget, False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--basis-dir", default="results/corpus_bases")
    ap.add_argument("--out", default="results/unlabeled_value")
    ap.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    ap.add_argument("--stages", default=",".join(s[0] for s in STAGES))
    a = ap.parse_args()

    wanted = a.stages.split(",")
    unknown = sorted(set(wanted) - {s[0] for s in STAGES})
    if unknown:
        raise SystemExit(f"[!] unknown stage(s) {unknown}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_dir) if a.data_dir else resolve(None, download=False, quiet=True)
    ctx = Context(root=root, mask=corpus_mask(root), basis_dir=Path(a.basis_dir),
                  sizes=corpus_sizes(a.basis_dir))
    print(f"[*] corpus sizes {ctx.sizes}; outputs to {out}", flush=True)
    for name, filename, fn, resumable in STAGES:
        if name not in wanted:
            continue
        path = out / filename
        if path.exists() and not a.force and not resumable:
            continue
        if a.force:
            path.unlink(missing_ok=True)
        t0 = time.time()
        print(f"[*] {name} -> {path}", flush=True)
        fn(ctx, path)
        print(f"[+] {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
