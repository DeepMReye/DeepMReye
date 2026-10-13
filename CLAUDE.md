# CLAUDE.md

Orientation for an agent picking up this project.

Read **`README.md`** first -- it is the method, the layout and how to run every
stage. Then **`FINDINGS.md`**, which is the record of what was tried, what the
controls said, and which directions are closed. Those two files are the whole
documentation set, deliberately: an earlier version of this repo carried six
markdown files totalling 6700 lines, most of it about code that no longer
exists. (`slurm/README.md` covers the cluster split; it is not about the
method. `paper/` and `docs/` are **gitignored and local to the author's
checkout** -- the COSYNE abstract, its figure, the snapshot of the figure's
input JSON, and a provenance note on which claims it makes. If they are present
in your working tree, `paper/README.md` and `docs/COSYNE_FRAMING.md` are worth
reading before touching a headline number; if they are absent, nothing in the
method depends on them.)

**`results/` is gitignored.** Every number quoted anywhere is in `FINDINGS.md`,
and `README.md` lists which script regenerates which JSON. Do not go looking for
a results file to confirm a claim -- regenerate it, or trust the record.

**The corpus is every eligible participant since 6 October 2026** (FINDINGS,
"The corpus was refitted"): `unsupervised.corpus_subjects` -- 2,060
participants of 652 datasets, never a labelled dataset's source accession. It
is **not** gated on the QA labels (they gate downloads), and there is no cap.
The headline moved 0.770 -> 0.771 (sub-TR), 0.838 -> 0.835 (1-TR). FINDINGS
sections marked *(2,060)* carry the new numbers; the rest are on the earlier
random 2,000 and were not rerun. Corpus-size curves below the full corpus are
20 random draws per size (`--stages scaling-draws`), not one nested sweep. If
you change what enters the corpus, refit with `scripts/fit_corpus_basis.py`
(and its draws), rerun `scripts/analysis_unlabeled.py`, and update
`probe.CALIBRATION`.

## Before you propose an improvement

Check `FINDINGS.md` first, and specifically the **table at the head of
"Trying to beat `lr-cca`"** -- seventeen arms, one row each, scannable in a
minute. If your idea is one of those rows it has already been measured; say
which row it is *not*.

What is closed, in one breath: eight representation-learning arms (all lost to a
linear basis, because **gaze is linearly accessible from these features**, so a
non-linear encoder in front of a linear readout has nothing to add -- the
supervised non-linear ceiling is one command and it is *below* ridge); the
estimator's own hyperparameters; every covariance the accumulator can form,
including the temporal ones; the mirror-symmetry prior; selection criteria past
second order (ICA, dictionary learning); per-participant registration; both
corpus budget axes; and putting the temporal contrast into the bilateral
criterion.

Three things in that section are *positive* and worth reusing rather than
re-deriving: the whitening dose-response (PLS -> CCA is monotone, which is the
first real evidence that correlation is the right criterion), the random-subspace
floor at matched rank (0.394 against 0.769), and the fact that the cross-orbit
criterion **fitted on the labelled scans alone loses to the corpus fit** -- which
is the cleanest statement of what the unlabelled half is for.

Two experiments are blocked on one line of ingest code each, not on ideas:
the per-participant anatomical mean (the stored block is z-scored per voxel) and
the `SliceTiming` vector. Add both the next time anything is re-extracted.

Two more constraints that bound almost any proposal:

- The **noise floor on a 9-fold median is ~0.02**. A difference under that is a
  tie no matter how suggestive its direction.
- The **temporal envelope** bounds the available headroom to roughly 0.06-0.10 r
  on two of twelve (dataset, axis) cells. The acquisition sets the scale and the
  linear readout is already at it.

Neither is a reason not to try something. Both are reasons to say up front what
result would count as a win.

**Read "What the corpus buys, and where the rest is lost" before anything
else in `FINDINGS.md`.** It relocates the headroom and corrects one comparison:

- Every *frozen* linear basis holds the same ~0.83 of gaze when the readout is
  fitted *inside* a dataset (a PCA of the study's own labelled scans reaches
  0.843), and no non-linear readout finds more. The 0.06 lost under
  leave-one-dataset-out is the feature->gaze map -- mostly its timing --
  differing between acquisitions. A proposal has to say how it makes that map
  transfer; a better representation cannot.
- With the same lag context on both, `lr-cca` leads variance PCA by 0.017 at
  the full corpus (0.771 / 0.754), inside the noise floor. Its case is the
  scaling -- `lr-cca` gains +0.12 from 10 participants (mean of 20 draws),
  variance PCA is flat from 100 -- and the matched-rank comparison (0.771
  against 0.743 at k = 32), not that margin. Below ~400 participants a corpus
  is a lottery: the 95% range of draws at N = 100 is 0.54-0.70.
- The corpus axis is flat by N = 800; the labelled-*dataset* axis is still
  rising at eight.
- **Viewing priors are measured** ("Viewing priors" in FINDINGS): only three
  datasets share a stimulus clock, and **`dsL01`'s labels are its target
  sequence**, identical for all 170 participants. Other viewers' decoded gaze
  adds ~0.02 on the mean to any decoder; feature-level alignment across
  domains or viewers hurts.
- **A label-free confidence exists, per participant only**
  (`metrics.orbit_agreement` on `probe.lodo(views=...)`): it flags each
  dataset's worst fifth at AUROC 0.93 and keeping the 80% most confident is
  within 0.002 of an oracle. It cannot rank datasets (the map error is shared
  by both orbits) or volumes (nothing can; a learned error model reaches 0.14).
  It also tracks the eyeball QA on the unlabelled corpus (median 0.47 in
  passing datasets, 0.24 in failing ones, covered eye masks only -- an
  uncovered mask decodes flat and its orbits "agree" trivially).
  **Self-training is closed**: a linear student on the decoder's own labels
  reproduces the decoder exactly, whatever volumes it is fitted on.
- **Both bases are tuned** (`--stages tune`): rank, lag context, whitening, a
  penalty grid to 1e7. Best of grid, `corpus-pca` 0.761 (k = 64, +-2 TR) against
  `lr-cca` 0.771 (shipped). Do not reopen readout-side tuning of either; and
  the readout itself does not matter (`--stages readout-grid`: least squares
  equals ridge within 0.005 on every base, the MLP is below it).
- **Against a supervised decoder** (the same ridge on every eye voxel, or PLS or
  an MLP, trained on the labelled participants only) `lr-cca` wins at every
  labelled-dataset budget: +0.035 on the fold mean with eight (8 of 9
  datasets), +0.10 with one (every one of 20 draws). At 1-TR the best supervised decoder (a ridge with
  no temporal context, 0.823) is within the noise floor of ours. Against a
  PCA of the labelled scans (no corpus, but not supervised -- its basis sees no
  labels) it wins only with few datasets: +0.05 at one or two, nothing from six.
  Call the first "supervised" and the second "no corpus"; they are different
  controls.
- Inside a study the two corpus bases are the same thing: their predictions
  correlate at 0.96 and their errors at 0.92, at every number of labelled
  participants from one up. `lr-cca`'s measurable edge is transfer at small
  rank (LODO +0.03 to +0.04 at k = 24-32), nothing else.
- Inside a *new study* with K labelled participants the corpus basis has **no
  few-label advantage**: a ridge on that study's voxels or a PCA of its labelled
  scans ties it at one participant and leads by ~0.02 on the fold mean from
  two. The `raw` (stride-4) arm is not a fair supervised baseline -- it keeps
  233 of 14,236 voxels.
- The frozen decoder is worth two to four labelled participants of a new study;
  one or two of them do not improve it.
- Eighteen further arms (deep, per-participant and site-invariant CCA; readout
  weighting; learned, generative and pretrained temporal models) are in one
  table there, measured and deleted.

## Rules that are not negotiable

- **Every self-supervised arm needs its own untrained control**, and the control
  must be built from the trained model's own attributes rather than from
  configuration. A control assembled from config drifts the moment a field is
  added -- that happened, and it inflated a reported margin by 0.15.
- **No model output may gate a dataset.** `qa_classifier` ranks and pre-selects
  in the UI; approval is manual labels. If you find yourself wiring a
  probability into an approval path, that is a deleted design coming back.
- **The corpus never contains an evaluation participant.** The `dsL*` folders
  are excluded, and so are the OpenNeuro accessions three of them were
  ingested from (`unsupervised.EVALUATION_SOURCES`) -- four of the same runs
  under another name were in the corpus until 6 October 2026. Add the source
  of any new labelled dataset there.
- **One implementation of the number.** `deepmreye/probe.py` is it. Two
  implementations of "the score" is how a 0.221 came to be compared against a
  0.847. If you need a variant protocol, add a parameter, not a second file --
  `split=`, `readout=`, `train_filter=`, `shots=` and `views=` on `probe.lodo`
  are what that looks like.
- **Fit on all the rows.** There is no training-row subsample any more and
  reintroducing one needs a stated reason.
- **Nothing in the feature path may use torch** -- LightGBM and torch each load
  their own OpenMP runtime and deadlock silently when mixed in one process.

## Cluster (Leonardo / CINECA)

Two constraints that pull in opposite directions:

- **Compute nodes have no outbound network.** Anything touching S3 or the Hub
  runs on a login node.
- **Login sessions are capped at 32 GB**, shared across all your shells on that
  node. ANTs memory does not correlate with input size, so this cannot be sized
  around -- and *page cache from your own writes* counts against the cap, which
  is why `stage_downloads.py` calls `posix_fadvise(POSIX_FADV_DONTNEED)`. A
  silent death with no traceback is almost always one of these two.

So ingestion is split: stage on login, extract on compute (`slurm/README.md`).
Everything else is one command through `slurm/run.sbatch`.

Submit under account **`AIFAC_S07_154`**, partition **`boost_usr_prod`**. Not
`EUHPC_D21_101` -- that appears in the repo path but the budget expired. sbatch
reports a wrong account and a wrong partition with the *same* error message, so
test a pair with `sbatch --test-only` rather than guessing. `boost_qos_dbg` for
smoke tests, default QoS for real runs.

## Dev

`uv` + `pyproject.toml`. Use `.venv/bin/python`. Tests: `pytest deepmreye/tests/ -q`.
