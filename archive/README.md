# archive/ — closed arms, kept as the reproducibility record

Moved here on **2026-08-14**, not deleted. Everything in this directory is the
evidence behind a *negative* result written up in `STATE.md`; the repo is
focused on the four arms that remain live (`fold-pca`, `corpus-pca`, `lr-cca`,
`jepa` — see `RESEARCH.md`).

**This code is not importable from here.** `deepmreye/` package paths were not
preserved as an importable tree, on purpose: nothing should depend on an
archived arm by accident. To re-run one, move the module back to
`deepmreye/`, its tests back to `deepmreye/tests/`, and re-add its feature kind
and builder to `deepmreye/evaluate/features.py` and `scripts/eval_probe.py`
(both were stripped — see git history for the exact wiring).

## What is here and why it is closed

| module | arm | verdict, and where it is written up |
|---|---|---|
| `deepmreye/temporal.py` | `ar-gru` — causal next-TR prediction | Trained held-out R² +0.230 but probe **0.530 against its own untrained control's 0.686**. Training helped on 0/6 folds. The predictable part of an eye block is drift, motion and global signal, so a predictive objective evicts gaze. |
| `deepmreye/crossorbit.py` | `xorb` — soft-argmax position bottleneck | 0.389 against an untrained control at 0.273, i.e. only 30% of the score is learned. Never approaches `lr-cca`, the *linear* form of the same cross-orbit constraint. |
| `deepmreye/orbitrot.py` | `xrot` — 2-DOF rotation bottleneck | The better *learner* (0.422 at matched width, 82% of it earned, 6/6 folds over control) and still far below `fold-pca`. |
| `deepmreye/orbitcon.py` | `ocon` — cross-orbit VICReg | Training helps (+0.08 to +0.14 over control) but the probe **peaks at 200 pretraining runs and falls** while the objective improves monotonically to 884. What the two orbits share is dominated by motion and drift. |
| `deepmreye/models/composite_net.py` | `composite-net` | Loses to `fold-pca:64` on 7 of 8 folds. |
| `deepmreye/models/contrastive_net.py` | `contrastive-net` | Its negative result is **void**, not a finding: no untrained control, 188 s of training, a temporal objective already closed by the next-TR result, and `--exclude-datasets` not wired to the CLI. `VICRegLoss` from here was used by `orbitcon`. |

All six are superseded by one measurement:
`scripts/analyze_nonlinear_ceiling.py` shows that **no supervised non-linear
readout beats linear ridge** on these features, so a non-linear encoder in front
of a linear readout has nothing to add. See `RESEARCH.md` §3.1.

## Superseded evaluation scripts

`scripts/eval_corpus_*`, `eval_checkpoints_*`, `track_epoch_saturation.py` —
one-off suites replaced by `sweep_corpus_scaling.py` / `sweep_probe_scaling.py`.
Several of them also evaluate on **`dsL11_backtothefuture`**, which failed
`verify_gaze_sync.py` (two of four participants peak a full TR off, per-subject
rather than dataset-level) and is not a valid fold. Numbers from those scripts
should not be quoted.
