# CLAUDE.md

Orientation for an agent picking up this project.

Read **`README.md`** first -- it is the method, the layout and how to run every
stage. Then **`FINDINGS.md`**, which is the record of what was tried, what the
controls said, and which directions are closed. Those two files are the whole
documentation set, deliberately: an earlier version of this repo carried six
markdown files totalling 6700 lines, most of it about code that no longer
exists.

## Before you propose an improvement

Check `FINDINGS.md` first. Eight representation-learning arms were built and
measured here and all of them lost to a linear basis, for a reason that is
itself measured: **gaze is linearly accessible from these features**, so a
non-linear encoder in front of a linear readout has nothing to add. The
supervised non-linear ceiling is one command and it is *below* ridge.

Two more constraints that bound almost any proposal:

- The **noise floor on a 9-fold median is ~0.02**. A difference under that is a
  tie no matter how suggestive its direction.
- The **temporal envelope** bounds the available headroom to roughly 0.06-0.10 r
  on two of twelve (dataset, axis) cells. The acquisition sets the scale and the
  linear readout is already at it.

Neither is a reason not to try something. Both are reasons to say up front what
result would count as a win.

## Rules that are not negotiable

- **Every self-supervised arm needs its own untrained control**, and the control
  must be built from the trained model's own attributes rather than from
  configuration. A control assembled from config drifts the moment a field is
  added -- that happened, and it inflated a reported margin by 0.15.
- **No model output may gate a dataset.** `qa_classifier` ranks and pre-selects
  in the UI; approval is manual labels. If you find yourself wiring a
  probability into an approval path, that is a deleted design coming back.
- **One implementation of the number.** `deepmreye/probe.py` is it. Two
  implementations of "the score" is how a 0.221 came to be compared against a
  0.847. If you need a variant protocol, add a parameter, not a second file.
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
