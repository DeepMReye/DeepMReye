# Cluster ingestion (SLURM)

Everything in this folder is cluster-specific. Nothing outside it needs SLURM,
and nothing here is needed to label, train, or publish — those run anywhere
(see the repo `README.md`).

The split exists because of two constraints on Leonardo that pull in opposite
directions: **compute nodes have no outbound network**, and **login sessions are
capped at 32 GB**. So the download half must run on a login node and the
memory-hungry registration half must run on compute. `python -m deepmreye
compile` / `preprocess`, which do both in one process, cannot work at scale
here. Details in `../CLAUDE.md`.

## Files

| file | runs on | does |
|---|---|---|
| `stage_downloads.py` | login node | resolve subjects from OpenNeuro S3, download `.nii.gz` into staging, write `manifest.jsonl` |
| `extract_staged.py` | compute node | ANTs registration, eye extraction, normalization, per-participant HDF5 write. No network. |
| `extract_array.sbatch` | compute node | SLURM array wrapper around the above |
| `submit_extraction.sh` | login node | sizes the array from the manifest and submits it |

## Usage

```bash
export DATA=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data
export STAGING=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/staging

# 1. LOGIN NODE — download (~2 s/subject, low memory)
#    --sample 2 does the QA sampling pass; omit it to stage every subject
#    of the approved datasets.
setsid nohup .venv/bin/python slurm/stage_downloads.py \
    --data-dir $DATA --staging-dir $STAGING --discover all --sample 2 --workers 3 \
    > logs/stage.log 2>&1 < /dev/null &

# 2. COMPUTE — register and extract (~42-110 s/subject)
SLURM_PARTITION=boost_usr_prod ./slurm/submit_extraction.sh

# 3. LOGIN NODE — fold worker records into the registry
.venv/bin/python -m deepmreye merge-registry --data-dir $DATA
```

Both stages are resumable and safe to re-run: staging skips files already
downloaded, extraction skips participants already extracted.

## Things that will bite you

- **Account and partition.** Submit under `AIFAC_S07_154` on `boost_usr_prod`.
  `sbatch` reports *both* a wrong account and an account-without-allocation as
  "invalid account or expired budget", so test a pair with
  `sbatch --test-only -A <acct> -p <part> --wrap=true` rather than guessing.
  `submit_extraction.sh` does this automatically.
- **Memory.** ANTs `SyNAggro` diverges unpredictably — input size does not
  predict it. Each registration runs in a forked child with an RSS watchdog
  (`--mem-limit-gb`, default 100 under a 120 G task) so a runaway costs one
  subject rather than the whole array task. Do not replace this with
  `RLIMIT_AS`; that caps virtual address space, not resident memory, and mass
  `itkImportImageContainer` errors are the symptom.
- **Long login-node jobs** need `setsid nohup … &`; plain `nohup` gets killed
  when the session is recycled. Check liveness with
  `ps -u $USER | grep "[s]tage_downloads"`, never `pgrep -f` inside a watcher
  (it matches itself).
- **Monitoring** must read `sacct` states, not just log lines: a task killed by
  the OOM killer writes no failure line.
- Slurm commands need `dangerouslyDisableSandbox: true` from the agent harness.

## Casualties

Subjects that fail are appended to `$STAGING/deferred_<task>.jsonl` — never
silently dropped. To retry them on a bigger allocation:

```bash
cat $STAGING/deferred_*.jsonl > $STAGING/retry.jsonl
MANIFEST=$STAGING/retry.jsonl ./slurm/submit_extraction.sh
```
